package com.presenceguard.yolo.detect

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.presenceguard.yolo.core.DetectionConfig
import com.presenceguard.yolo.core.FrameResult
import com.presenceguard.yolo.core.PersonCandidate
import com.presenceguard.yolo.core.Roi
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

/**
 * CameraX 分析器：单引擎（YOLO）多路线检测。
 *
 * 三条路线，按场景自动选：
 *
 *  - **整帧**：每帧对整帧跑一次 640×640 推理。覆盖近处大目标，成本固定。
 *  - **瓦片**：把画面切片，每帧只滚几片。远处小目标的补救手段，默认关闭，
 *    灵敏度拉到"灵敏"时才开一片。
 *  - **选区**：用户画了多边形之后，只在这些区域里判定，并且把选区裁出来放大送检
 *    （相当于给画面加了一台长焦）。选区存在时另外两条路线全部跳过。
 *
 * v2 相比 v1 简化了一处很关键的东西：**不再有并行线程和 CountDownLatch**。
 * v1 里姿态引擎和目标检测跑在两个线程上再汇合，是因为 ML Kit 的回调式 API 逼着你这么做；
 * v2 只有一路同步推理，直接在分析线程里跑完更简单，也顺带消掉了「位图被子线程回收」
 * 那一整类崩溃。
 */
class FrameAnalyzer(
    private val detector: YoloDetector?,
    private val fusion: PresenceFusionEngine,
    private val listener: Listener
) : ImageAnalysis.Analyzer {

    interface Listener {
        fun onResult(result: FrameResult, outcome: PresenceFusionEngine.Outcome)
        fun onError(message: String)
    }

    @Volatile
    var config: DetectionConfig = DetectionConfig()

    /** 前置摄像头时叠加层需要镜像 */
    @Volatile
    var mirrored: Boolean = false

    /**
     * 省电模式下的最小帧间隔（毫秒）。0 表示不限。
     * 判断"画面里有没有人"不需要高帧率，5fps 已经足够，功耗却能降一大截。
     */
    @Volatile
    var minFrameIntervalMs: Long = 0

    /** 瓦片网格 */
    @Volatile
    var tileGrid: TileGrid = TileGrid.DEFAULT

    /** 每帧滚几片瓦片。0 表示关闭瓦片路线。 */
    @Volatile
    var tilesPerFrame: Int = 0

    /** 用户框选的多边形选区。**非空即切换到「选区模式」**。 */
    @Volatile
    var rois: List<Roi> = emptyList()

    /** 选区模式下每帧扫几个选区 */
    @Volatile
    var roiPerFrame: Int = 1

    @Volatile
    var lastConvertMs: Long = 0
        private set

    @Volatile
    var lastDetectMs: Long = 0
        private set

    @Volatile
    var lastInferenceMs: Long = 0
        private set

    private var roiCursor = 0
    private var tileCursor = 0

    private var lastFrameAt = 0L
    private var lastAcceptedAt = 0L
    private var smoothedFps = 0f
    private var diagTick = 0

    /** 上一帧的判定结果，用来决定这一帧要不要花算力做瓦片扫描 */
    @Volatile
    private var lastPresent = false

    fun rebind(config: DetectionConfig, mirrored: Boolean) {
        this.config = config
        this.mirrored = mirrored
        lastFrameAt = 0L
        smoothedFps = 0f
    }

    // ML Kit 时代需要 android.media.Image，CameraX 把它标为实验 API；v2 只用位图，
    // 但这个注解保留着也无害，因为 ImageProxy.image 仍然依赖它。
    @OptIn(markerClass = [ExperimentalGetImage::class])
    override fun analyze(imageProxy: ImageProxy) {
        val begin = SystemClock.elapsedRealtime()

        // 省电模式：主动丢帧
        val interval = minFrameIntervalMs
        if (interval > 0 && begin - lastAcceptedAt < interval) {
            imageProxy.close()
            return
        }
        lastAcceptedAt = begin

        val rotation = imageProxy.imageInfo.rotationDegrees
        val bitmap = try {
            val raw = imageProxy.toBitmap()
            rotate(raw, rotation)
        } catch (t: Throwable) {
            Log.w(TAG, "frame conversion failed", t)
            imageProxy.close()
            return
        }
        lastConvertMs = SystemClock.elapsedRealtime() - begin

        val frameWidth = if (rotation % 180 == 0) imageProxy.width else imageProxy.height
        val frameHeight = if (rotation % 180 == 0) imageProxy.height else imageProxy.width

        val cfg = config
        val activeRois = rois
        val roiMode = activeRois.isNotEmpty()

        // 选区模式下 fusion 层要换一套门槛：用户已经用多边形声明了"这块区域值得看"，
        // 再拿"占整帧面积太小"去卡他，等于把选区这个先验扔掉了。
        val fusionCfg = if (roiMode) {
            cfg.copy(minAreaRatio = ROI_MIN_AREA_RATIO, smallTargetAreaRatio = 0f)
        } else {
            cfg
        }

        try {
            // 镜头被遮挡 / 贴着桌面时画面几乎是纯色，先挡掉：省一次推理，也避免误报卡死
            val luma = averageLuma(bitmap)
            if (luma < MIN_VALID_LUMA || luma > MAX_VALID_LUMA) {
                publish(begin, frameWidth, frameHeight, emptyList(), fusionCfg)
                return
            }

            val detections = ArrayList<PersonCandidate>(8)
            val detectBegin = SystemClock.elapsedRealtime()

            if (roiMode) {
                scanRois(bitmap, activeRois, detections, frameWidth, frameHeight)
            } else {
                detector?.detect(bitmap)?.let { detections += it }
                scanTiles(bitmap, detections, frameWidth, frameHeight)
            }

            lastDetectMs = SystemClock.elapsedRealtime() - detectBegin
            lastInferenceMs = detector?.lastInferenceMs ?: 0

            publish(begin, frameWidth, frameHeight, detections, fusionCfg)
        } finally {
            if (!bitmap.isRecycled) bitmap.recycle()
            imageProxy.close()
        }
    }

    // ------------------------------------------------------------ 路线

    /** 选区放大识别：裁出外接矩形 → 送检（等于放大）→ 按多边形过滤 */
    private fun scanRois(
        source: Bitmap,
        activeRois: List<Roi>,
        out: MutableList<PersonCandidate>,
        frameWidth: Int,
        frameHeight: Int
    ) {
        val engine = detector ?: return
        val take = min(roiPerFrame.coerceAtLeast(1), activeRois.size)
        val start = roiCursor % activeRois.size

        for (i in 0 until take) {
            val roi = activeRois[(start + i) % activeRois.size]
            val region = regionOf(roi, source.width, source.height) ?: continue
            val crop = runCatching {
                Bitmap.createBitmap(source, region.left, region.top, region.width(), region.height())
            }.onFailure { Log.w(TAG, "roi crop failed", it) }.getOrNull() ?: continue

            try {
                for (candidate in engine.detectRegion(crop, region.left, region.top)) {
                    remap(candidate, roi, frameWidth, frameHeight)?.let { out += it }
                }
            } finally {
                if (!crop.isRecycled) crop.recycle()
            }
        }
        roiCursor = (start + take) % activeRois.size
    }

    /**
     * 瓦片滚动扫描：每帧只跑几片，游标轮转，若干帧覆盖全图。
     * 只回传小目标——瓦片本来就是为了捞远处的小东西，大目标交给整帧那一路，
     * 否则同一个人会被多个瓦片各框一次。
     */
    private fun scanTiles(
        source: Bitmap,
        out: MutableList<PersonCandidate>,
        frameWidth: Int,
        frameHeight: Int
    ) {
        val engine = detector ?: return
        // 画面里已经有人时不扫瓦片：报警已触发，瓦片的使命是"找人"，
        // 稳态下省掉这几次推理，帧率能明显提升。空场景才是需要全力搜索的时候。
        val budget = if (lastPresent) 0 else tilesPerFrame
        if (budget <= 0) return

        val tiles = tileGrid.rects(source.width, source.height)
        if (tiles.isEmpty()) return

        val frameArea = (frameWidth * frameHeight).toFloat().coerceAtLeast(1f)
        val take = min(budget, tiles.size)
        for (i in 0 until take) {
            val tile = tiles[(tileCursor + i) % tiles.size]
            val crop = runCatching {
                Bitmap.createBitmap(source, tile.left, tile.top, tile.width(), tile.height())
            }.getOrNull() ?: continue
            try {
                for (candidate in engine.detectRegion(crop, tile.left, tile.top)) {
                    if (candidate.area() / frameArea <= TILE_TARGET_MAX_AREA) out += candidate
                }
            } finally {
                if (!crop.isRecycled) crop.recycle()
            }
        }
        tileCursor = (tileCursor + take) % tiles.size
    }

    // ------------------------------------------------------------ 选区几何

    /**
     * 选区的送检矩形（像素）。外扩一点，避免人站在选区边缘时框被裁掉半个身子。
     */
    private fun regionOf(roi: Roi, width: Int, height: Int): Rect? {
        val b = roi.bounds()
        val padX = (b.width() * REGION_PAD).coerceAtLeast(MIN_REGION_PAD)
        val padY = (b.height() * REGION_PAD).coerceAtLeast(MIN_REGION_PAD)

        val left = ((b.left - padX) * width).toInt().coerceIn(0, width - 1)
        val top = ((b.top - padY) * height).toInt().coerceIn(0, height - 1)
        val right = ((b.right + padX) * width).toInt().coerceIn(left + 1, width)
        val bottom = ((b.bottom + padY) * height).toInt().coerceIn(top + 1, height)

        if (right - left < MIN_REGION_PX || bottom - top < MIN_REGION_PX) return null
        return Rect(left, top, right, bottom)
    }

    /**
     * 过滤：中心点落在多边形内 就算；如果中心在外但身体明显压线（与选区外接矩形的
     * 交集 ≥ 框面积的 45%），同样收下。
     */
    private fun remap(
        candidate: PersonCandidate,
        roi: Roi,
        frameWidth: Int,
        frameHeight: Int
    ): PersonCandidate? {
        val box = candidate.box
        val cx = box.centerX() / frameWidth
        val cy = box.centerY() / frameHeight
        if (roi.contains(cx, cy)) return candidate

        val bounds = roi.boundsPx(frameWidth, frameHeight)
        val left = max(box.left, bounds.left)
        val top = max(box.top, bounds.top)
        val right = min(box.right, bounds.right)
        val bottom = min(box.bottom, bounds.bottom)
        if (right <= left || bottom <= top) return null

        val inter = (right - left) * (bottom - top)
        val own = box.width() * box.height()
        return if (own > 0f && inter >= own * ROI_OVERLAP_ACCEPT) candidate else null
    }

    // ------------------------------------------------------------ 收尾

    private fun publish(
        begin: Long,
        frameWidth: Int,
        frameHeight: Int,
        detections: List<PersonCandidate>,
        fusionCfg: DetectionConfig
    ) {
        val now = SystemClock.elapsedRealtime()
        val outcome = fusion.update(detections, frameWidth, frameHeight, now, fusionCfg)
        lastPresent = outcome.present

        if (++diagTick % 10 == 0) {
            val area = (frameWidth * frameHeight).toFloat().coerceAtLeast(1f)
            val info = detections.joinToString(" | ") { c ->
                String.format(Locale.US, "%.2f[a%.4f]", c.score, c.area() / area)
            }
            Log.d(
                TAG,
                String.format(
                    Locale.US,
                    "cv=%dms det=%dms inf=%dms src=%dx%d roi=%d(z%d) tiles=%d coords=%s " +
                        "raw=[%s] -> n=%d fs=%.2f ss=%.2f v=%d %s",
                    lastConvertMs, lastDetectMs, lastInferenceMs, frameWidth, frameHeight,
                    rois.size, roiCursor, tilesPerFrame,
                    when (detector?.normalizedCoords) {
                        true -> "norm"; false -> "px"; null -> "?"
                    },
                    info, outcome.candidates.size, outcome.frameScore, outcome.smoothedScore,
                    outcome.votes, if (outcome.present) "PRESENT" else "none"
                )
            )
        }

        val delta = now - lastFrameAt
        lastFrameAt = now
        val instantFps = if (delta in 1..2000) 1000f / delta else 0f
        smoothedFps = if (smoothedFps <= 0f) instantFps else smoothedFps * 0.8f + instantFps * 0.2f

        listener.onResult(
            FrameResult(
                candidates = outcome.candidates,
                sourceWidth = frameWidth,
                sourceHeight = frameHeight,
                mirrored = mirrored,
                processingMs = now - begin,
                fps = smoothedFps
            ),
            outcome
        )
    }

    /**
     * 采样估算平均亮度（0~255）。每 64 个像素取一个，耗时约 3~5ms。
     */
    private fun averageLuma(bitmap: Bitmap): Int {
        val w = bitmap.width
        val h = bitmap.height
        if (w <= 0 || h <= 0) return 0
        val stepX = (w / SAMPLES_PER_SIDE).coerceAtLeast(1)
        val stepY = (h / SAMPLES_PER_SIDE).coerceAtLeast(1)

        var sum = 0L
        var count = 0
        var y = 0
        while (y < h) {
            var x = 0
            while (x < w) {
                val color = bitmap.getPixel(x, y)
                sum += (299 * Color.red(color) + 587 * Color.green(color) + 114 * Color.blue(color)) / 1000
                count++
                x += stepX
            }
            y += stepY
        }
        return if (count == 0) 0 else (sum / count).toInt()
    }

    fun shutdown() = Unit

    private fun rotate(source: Bitmap, degrees: Int): Bitmap {
        val normalized = ((degrees % 360) + 360) % 360
        if (normalized == 0) return source
        val matrix = Matrix().apply { postRotate(normalized.toFloat()) }
        val rotated = Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
        if (rotated !== source && !source.isRecycled) source.recycle()
        return rotated
    }

    companion object {
        private const val TAG = "FrameAnalyzer"

        /** 采样网格边长，64×64 ≈ 4096 个采样点，足够反映整体明暗 */
        private const val SAMPLES_PER_SIDE = 64

        /** 低于这个亮度（几乎纯黑）视为镜头被挡住，不做推理 */
        private const val MIN_VALID_LUMA = 10

        /** 高于这个亮度（几乎纯白，如镜头贴纸/强光过曝）同样跳过 */
        private const val MAX_VALID_LUMA = 248

        /** 选区送检时外接矩形外扩的比例 */
        private const val REGION_PAD = 0.06f

        /** 外扩的绝对下限（归一化），小选区上按比例算出来的外扩太少 */
        private const val MIN_REGION_PAD = 0.012f

        /** 送检矩形的最小像素边长。裁出来全是模糊块时放大到 640 只是把噪声放大 */
        private const val MIN_REGION_PX = 40

        /** 框与选区的重叠达到框自身面积的这个比例，就认为人确实在选区内 */
        private const val ROI_OVERLAP_ACCEPT = 0.45f

        /**
         * 选区模式下的面积下限。整帧模式下 0.003 是为了过滤远处噪点；
         * 选区是用户亲手划的，这个先验已经足够强，面积门限可以放到极低。
         */
        private const val ROI_MIN_AREA_RATIO = 0.0002f

        /** 瓦片路线只收面积占整帧 12% 以下的目标 */
        private const val TILE_TARGET_MAX_AREA = 0.12f
    }
}
