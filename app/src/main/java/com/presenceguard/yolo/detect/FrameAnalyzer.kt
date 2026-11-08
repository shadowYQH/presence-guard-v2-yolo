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

    /**
     * 数字裁切倍数（双指缩放）。1 表示不裁。
     *
     * 这是**纯数字裁切**：不动任何相机参数，只是从画面中心裁一块出来送模型，
     * 等效增加镜头焦距。物理焦段切换是另一条路（见 DetectionService 的镜头选择），
     * 两者互不干扰，状态栏也分开显示。
     *
     * 选区模式下本参数不生效 —— 选区本身就是放大手段，再乘一次会让人莫名其妙地
     * 从视野里消失。
     */
    @Volatile
    var cropScale: Float = 1f

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
    private var roiSubCursor = 0

    /** 每块瓦片上一次的 16×16 灰度指纹，用于帧差跳过 */
    private val tilePrints = HashMap<Int, ByteArray>()

    /** 每块瓦片连续被跳过多少次，用来做"跳过太多次就强制扫一遍"的兜底 */
    private val tileSkips = HashMap<Int, Int>()

    /** 本帧因为帧差过小被跳过的瓦片数，仅用于诊断 */
    @Volatile
    var lastSkippedTiles: Int = 0
        private set

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
        lastInspectionAt = 0L
        tilePrints.clear()
        tileSkips.clear()
        inspectionFusion.reset()
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
        val frameWidth = if (rotation % 180 == 0) imageProxy.width else imageProxy.height
        val frameHeight = if (rotation % 180 == 0) imageProxy.height else imageProxy.width

        // 副摄专用：只有到点的那一帧干活，其余帧原地丢掉。
        // 注意这里不看 minFrameIntervalMs —— 那个是"省电降帧"，管的是主路线；
        // 巡检的节奏由 inspectionIntervalMs 单独控制。
        if (inspectionOnly) {
            val gap = inspectionIntervalMs
            val now = SystemClock.elapsedRealtime()
            if (gap <= 0 || now - lastInspectionAt < gap) {
                imageProxy.close()
                return
            }
            lastInspectionAt = now
            try {
                scanInspectionFrame(imageProxy, rotation, frameWidth, frameHeight)
            } finally {
                imageProxy.close()
            }
            return
        }

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

        val detections = ArrayList<PersonCandidate>(8)
        try {
            val detectBegin = SystemClock.elapsedRealtime()
            val convertBefore = lastConvertAccum
            lastConvertAccum = 0

            // 巡检优先：到点的那一帧改走整帧粗判，走完照常回到用户选定的路线。
            val now = SystemClock.elapsedRealtime()
            val interval = inspectionIntervalMs
            val inspection = interval > 0 && now - lastInspectionAt >= interval
            if (inspection) lastInspectionAt = now

            when {
                inspection -> scanInspectionFrame(imageProxy, rotation, frameWidth, frameHeight)
                roiMode -> // 选区模式：压根不转整帧。每块选区单独从 YUV 里挖出来直接转，
                    // 矩形外的像素一个都不碰 —— 这才是"选区外零计算"。
                    scanRois(imageProxy, rotation, activeRois, detections, frameWidth, frameHeight)
                else -> scanFrame(imageProxy, rotation, detections, frameWidth, frameHeight)
            }

            lastConvertMs = lastConvertAccum
            lastConvertAccum = convertBefore
            lastDetectMs = SystemClock.elapsedRealtime() - detectBegin
            lastInferenceMs = maxOf(lastInferenceMs, detector?.lastInferenceMs ?: 0)

            if (inspection) {
                // 巡检**不 publish**：它只负责"要不要响"，不改变界面上显示的主判定状态。
                // 如果把它当成一帧"没看见人"喂给主状态机，投票会被反复拉低，
                // 选区里的人反而更容易判丢。
                Log.d(
                    TAG,
                    String.format(
                        Locale.US,
                        "inspect: inf=%dms n=%d score=%.2f",
                        lastInferenceMs, inspectionCount, lastInspectionScore
                    )
                )
                return
            }

            publish(begin, frameWidth, frameHeight, detections, fusionCfg)
        } finally {
            imageProxy.close()
        }
    }

    /** 本帧累计的转换耗时（选区模式下会有多块） */
    private var lastConvertAccum: Long = 0

    // ------------------------------------------------------------ 全图巡检

    /**
     * 全图巡检间隔（毫秒）。0 表示关闭。
     *
     * 为什么要单独做这件事：选区模式下只看选区内，视野是"望远镜"；
     * 一旦有人绕到镜头视野之外或者站在选区外面，主判定完全看不见他。
     * 所以每 2 秒要退回到**当前镜头能看到的最广视野**扫一遍整帧，
     * 只做"有没有人"的粗判，不参与精细判定，也不投票。
     *
     * 注意这里不切换镜头：切镜头要 unbind/rebind，实测一次约 100ms，
     * 而且预览会黑一下 —— 每 2 秒闪一次没法用。用 max(视野) 的等价物
     * （不裁切、zoom 归位）就够了。
     */
    @Volatile
    var inspectionIntervalMs: Long = 2000L

    /** 巡检命中时的回调：参数是(是否有人, 分数)。由服务层决定要不要响 */
    @Volatile
    var onInspectionHit: ((Boolean, Float) -> Unit)? = null

    /**
     * 只做巡检，别的什么都不做。
     *
     * 这个开关是给**第二颗摄像头**用的：前置和后置同时出流时，副摄的唯一任务就是
     * 「那一面有没有人」，它不应该跑瓦片、不应该跑选区、更不应该往主状态机里投票
     * （否则主判定的人数和分数会被另一个视角的画面来回拉扯）。
     *
     * 打开后 [analyze] 只在到点时做一次整帧粗判，其余帧直接丢弃 —— 连位图都不转，
     * 30fps 的副摄流压在手上的开销就是每帧一次时间戳比较。
     */
    @Volatile
    var inspectionOnly: Boolean = false

    /** 巡检用独立的融合状态机，绝不能污染主判定 */
    private val inspectionFusion = PresenceFusionEngine()
    private var lastInspectionAt = 0L

    @Volatile
    var lastInspectionScore: Float = 0f
        private set

    @Volatile
    var inspectionCount: Int = 0
        private set

    /**
     * 巡检帧：整帧一次粗判。
     *
     * 和常规整帧路线的区别是：无视选区、无视数字裁切、不扫瓦片、结果进独立的融合引擎。
     * 目的只有一个 —— 确认"整个视野里到底有没有人"。
     */
    private fun scanInspectionFrame(
        imageProxy: ImageProxy,
        rotation: Int,
        frameWidth: Int,
        frameHeight: Int
    ) {
        val t0 = SystemClock.elapsedRealtime()
        val raw = imageProxy.toBitmap()
        val bitmap = rotate(raw, rotation)
        lastConvertAccum += SystemClock.elapsedRealtime() - t0

        try {
            if (averageLuma(bitmap) < MIN_VALID_LUMA) return
            val found = detector?.detect(bitmap) ?: emptyList()
            lastInferenceMs = detector?.lastInferenceMs ?: 0

            val outcome = inspectionFusion.update(
                found, frameWidth, frameHeight, SystemClock.elapsedRealtime(), config
            )
            lastInspectionScore = outcome.smoothedScore
            inspectionCount = found.size
            onInspectionHit?.invoke(outcome.present, outcome.smoothedScore)
        } finally {
            if (!bitmap.isRecycled) bitmap.recycle()
        }
    }

    // ------------------------------------------------------------ 路线

    /**
     * 常规路线的入口：整帧（可选数字裁切）+ 瓦片。
     *
     * 数字裁切放在这里而不是相机上，是为了让"双指缩放"和"物理焦段"彻底分开：
     * 裁切只影响送进模型的那块像素，相机参数一律不动。
     */
    private fun scanFrame(
        imageProxy: ImageProxy,
        rotation: Int,
        out: MutableList<PersonCandidate>,
        frameWidth: Int,
        frameHeight: Int
    ) {
        val scale = cropScale.coerceIn(1f, MAX_CROP_SCALE)
        var source: Bitmap? = null
        var offsetX = 0
        var offsetY = 0
        var wholeFrame = true

        try {
            if (scale > 1.01f) {
                val cropW = (frameWidth / scale).toInt().coerceIn(MIN_CROP_PX, frameWidth)
                val cropH = (frameHeight / scale).toInt().coerceIn(MIN_CROP_PX, frameHeight)
                val left = (frameWidth - cropW) / 2
                val top = (frameHeight - cropH) / 2
                val region = Rect(left, top, left + cropW, top + cropH)

                val t0 = SystemClock.elapsedRealtime()
                source = YuvCropper.crop(imageProxy, region)
                lastConvertAccum += SystemClock.elapsedRealtime() - t0
                if (source != null) {
                    offsetX = left
                    offsetY = top
                    wholeFrame = false
                }
            }

            if (source == null) {
                val t0 = SystemClock.elapsedRealtime()
                val raw = imageProxy.toBitmap()
                source = rotate(raw, rotation)
                lastConvertAccum += SystemClock.elapsedRealtime() - t0
            }

            val bitmap = source

            // 镜头被遮挡 / 贴着桌面时画面几乎是纯色，先挡掉：省一次推理，也避免误报卡死
            val luma = averageLuma(bitmap)
            if (luma < MIN_VALID_LUMA || luma > MAX_VALID_LUMA) return

            detector?.detect(bitmap, offsetX, offsetY)?.let { out += it }
            scanTiles(bitmap, out, frameWidth, frameHeight, offsetX, offsetY)

            // wholeFrame 只用来提醒自己：整帧路线的 tile 网格是相对整帧的
            if (wholeFrame) Unit
        } finally {
            source?.let { if (!it.isRecycled) it.recycle() }
        }
    }

    /**
     * 选区放大识别：**逐块**从 YUV 挖矩形（不转整帧）→ 送检（等于放大）→ 按多边形过滤。
     *
     * 选区内还要不要细分瓦片？看面积：
     *  - 选区占屏 ≤ [LARGE_ROI_AREA_RATIO]：整块送检。放大倍数已经等于"画面短边/选区短边"，
     *    再切会把上下文切碎，小选区会退化成"只看到一只鞋"，分数反而掉。
     *  - 选区更大：说明放大倍数不够，做 2×2 瓦片轮转，每帧看一块，等效再放大 2 倍。
     */
    private fun scanRois(
        imageProxy: ImageProxy,
        rotation: Int,
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
            val region = regionOf(roi, frameWidth, frameHeight) ?: continue

            val pieces = if (roi.areaRatio() > LARGE_ROI_AREA_RATIO) splitQuadrants(region) else listOf(region)
            val piece = pieces[roiSubCursor % pieces.size]
            roiSubCursor++

            val t0 = SystemClock.elapsedRealtime()
            val crop = YuvCropper.crop(imageProxy, piece) ?: fallbackCrop(imageProxy, rotation, piece)
            lastConvertAccum += SystemClock.elapsedRealtime() - t0
            if (crop == null) continue

            try {
                if (averageLuma(crop) < MIN_VALID_LUMA) continue
                for (candidate in engine.detectRegion(crop, piece.left, piece.top)) {
                    remap(candidate, roi, frameWidth, frameHeight)?.let { out += it }
                }
            } finally {
                if (!crop.isRecycled) crop.recycle()
            }
        }
        roiCursor = (start + take) % activeRois.size
    }

    /**
     * YUV 子矩形转换不可用时的兜底（例如厂商给了非 YUV_420_888 格式）。
     * 代价是转整帧，但仍然只在选区模式下发生。
     */
    private fun fallbackCrop(imageProxy: ImageProxy, rotation: Int, region: Rect): Bitmap? {
        return runCatching {
            val raw = imageProxy.toBitmap()
            val oriented = rotate(raw, rotation)
            Bitmap.createBitmap(oriented, region.left, region.top, region.width(), region.height())
                .also { if (!oriented.isRecycled) oriented.recycle() }
        }.onFailure { Log.w(TAG, "fallback crop failed", it) }.getOrNull()
    }

    /** 把一个矩形切成 2×2 四块，各自外扩一点避免人跨在接缝上 */
    private fun splitQuadrants(r: Rect): List<Rect> {
        val halfW = (r.width() / 2f).toInt()
        val halfH = (r.height() / 2f).toInt()
        val padX = (halfW * QUADRANT_PAD).toInt()
        val padY = (halfH * QUADRANT_PAD).toInt()
        val out = ArrayList<Rect>(4)
        for (qy in 0..1) {
            for (qx in 0..1) {
                val l = r.left + qx * halfW - if (qx == 1) padX else 0
                val t = r.top + qy * halfH - if (qy == 1) padY else 0
                val rr = r.left + (qx + 1) * halfW + if (qx == 0) padX else 0
                val bb = r.top + (qy + 1) * halfH + if (qy == 0) padY else 0
                out += Rect(
                    l.coerceIn(r.left, r.right - 1),
                    t.coerceIn(r.top, r.bottom - 1),
                    rr.coerceIn(r.left + 1, r.right),
                    bb.coerceIn(r.top + 1, r.bottom)
                )
            }
        }
        return out
    }

    /**
     * 瓦片滚动扫描：每帧只跑几片，游标轮转，若干帧覆盖全图。
     * 只回传小目标——瓦片本来就是为了捞远处的小东西，大目标交给整帧那一路，
     * 否则同一个人会被多个瓦片各框一次。
     *
     * 帧差跳过：送模型之前先比一下这块画面和上一轮比变了没有。
     * 完全没变就没必要再推理一遍 —— 结果必然和上次一样。海边蹲守、空走廊、
     * 夜里对着墙这类场景，绝大多数瓦片都是"什么都没发生"。
     */
    private fun scanTiles(
        source: Bitmap,
        out: MutableList<PersonCandidate>,
        frameWidth: Int,
        frameHeight: Int,
        offsetX: Int,
        offsetY: Int
    ) {
        val engine = detector ?: return
        // 画面里已经有人时不扫瓦片：报警已触发，瓦片的使命是"找人"，
        // 稳态下省掉这几次推理，帧率能明显提升。空场景才是需要全力搜索的时候。
        val budget = if (lastPresent) 0 else tilesPerFrame
        if (budget <= 0) {
            lastSkippedTiles = 0
            return
        }

        val tiles = tileGrid.rects(source.width, source.height)
        if (tiles.isEmpty()) return

        val frameArea = (frameWidth * frameHeight).toFloat().coerceAtLeast(1f)
        val take = min(budget, tiles.size)
        var skipped = 0

        for (i in 0 until take) {
            val index = (tileCursor + i) % tiles.size
            val tile = tiles[index]

            if (!shouldScanTile(source, tile, index)) {
                skipped++
                continue
            }

            val crop = runCatching {
                Bitmap.createBitmap(source, tile.left, tile.top, tile.width(), tile.height())
            }.getOrNull() ?: continue
            try {
                for (candidate in engine.detectRegion(
                    crop, tile.left + offsetX, tile.top + offsetY
                )) {
                    if (candidate.area() / frameArea <= TILE_TARGET_MAX_AREA) out += candidate
                }
            } finally {
                if (!crop.isRecycled) crop.recycle()
            }
        }
        lastSkippedTiles = skipped
        tileCursor = (tileCursor + take) % tiles.size
    }

    /**
     * 帧差判定：返回 false 表示"这块画面和上一轮几乎一样，跳过推理"。
     *
     * 兜底很重要：如果一块瓦片连续被跳过 [FORCE_RESCAN_AFTER] 次，强制扫一遍。
     * 否则一个缓慢移动的目标（或者一点点自动曝光漂移）可能永远攒不够差异，
     * 而那正是最需要发现"有人慢慢走近"的场景。
     */
    private fun shouldScanTile(source: Bitmap, tile: Rect, index: Int): Boolean {
        val current = fingerprint(source, tile)
        val previous = tilePrints[index]

        if (previous == null) {
            tilePrints[index] = current
            tileSkips[index] = 0
            return true
        }

        val skips = tileSkips[index] ?: 0
        if (skips >= FORCE_RESCAN_AFTER) {
            tilePrints[index] = current
            tileSkips[index] = 0
            return true
        }

        val diff = meanAbsDiff(previous, current)
        if (diff < MIN_TILE_MAD) {
            tileSkips[index] = skips + 1
            return false
        }

        tilePrints[index] = current
        tileSkips[index] = 0
        return true
    }

    /** 在瓦片内按 [FP_SIZE]×[FP_SIZE] 网格采样灰度，成本约 0.2ms */
    private fun fingerprint(source: Bitmap, tile: Rect): ByteArray {
        val out = ByteArray(FP_SIZE * FP_SIZE)
        var at = 0
        for (gy in 0 until FP_SIZE) {
            val y = tile.top + (tile.height() * gy / FP_SIZE)
            for (gx in 0 until FP_SIZE) {
                val x = tile.left + (tile.width() * gx / FP_SIZE)
                val c = source.getPixel(x, y)
                out[at++] = ((299 * Color.red(c) + 587 * Color.green(c) + 114 * Color.blue(c)) / 1000).toByte()
            }
        }
        return out
    }

    private fun meanAbsDiff(a: ByteArray, b: ByteArray): Int {
        var sum = 0
        for (i in a.indices) {
            sum += kotlin.math.abs((a[i].toInt() and 0xFF) - (b[i].toInt() and 0xFF))
        }
        return sum / a.size
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
                    "cv=%dms det=%dms inf=%dms src=%dx%d roi=%d(z%d) tiles=%d skip=%d crop=%.1f coords=%s " +
                        "raw=[%s] -> n=%d fs=%.2f ss=%.2f v=%d %s",
                    lastConvertMs, lastDetectMs, lastInferenceMs, frameWidth, frameHeight,
                    rois.size, roiCursor, tilesPerFrame, lastSkippedTiles, cropScale,
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

        /** 数字裁切（双指缩放）的上限。再大就只是把噪声放大，分数会掉 */
        const val MAX_CROP_SCALE = 6f

        /** 裁切后的最小边长，低于这个尺寸送进 640 模型毫无意义 */
        private const val MIN_CROP_PX = 64

        /**
         * 选区面积占比超过这个值就细分瓦片。
         * 6% 大致相当于"离镜头 3~4 米的人"所占的面积 —— 再大的选区放大倍数不够了。
         */
        private const val LARGE_ROI_AREA_RATIO = 0.06f

        /** 2×2 细分时四块各自向外扩的比例，避免人正好跨在十字接缝上 */
        private const val QUADRANT_PAD = 0.12f

        /** 帧差指纹网格边长。16×16=256 个采样点，够区分"有人走过"和"什么都没变" */
        private const val FP_SIZE = 16

        /**
         * 帧差阈值（平均绝对差，0~255）。低于它认为画面没变，跳过推理。
         * 6 是实测能滤掉传感器噪声和轻微曝光漂移、又不会漏掉一个人走过来的值。
         */
        private const val MIN_TILE_MAD = 6

        /**
         * 连续被跳过这么多次就强制扫一遍。
         * 没有这个兜底，慢慢靠近的目标可能永远攒不够帧差。
         */
        private const val FORCE_RESCAN_AFTER = 8
    }
}
