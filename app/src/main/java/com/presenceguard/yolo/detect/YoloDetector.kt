package com.presenceguard.yolo.detect

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import android.graphics.RectF
import android.util.Log
import com.presenceguard.yolo.core.DetectionSource
import com.presenceguard.yolo.core.Evidence
import com.presenceguard.yolo.core.PersonCandidate
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.Locale
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * YOLO 检测器 —— 自己驱动 TFLite Interpreter，不经过任何 Task / Support 库。
 *
 * 为什么不用 `tensorflow-lite-task-vision`：那一层是给 EfficientDet / MobileNet 这类
 * 带 metadata 的模型准备的，YOLO 的 TFLite 导出没有配套 metadata，
 * 而且 Task 库的 JNI 不接受量化模型（v1 踩过，见 docs/PERFORMANCE.md）。
 * 自己写这一层只有一百多行，换来三件事：
 *
 *  - 能把量化模型的 scale / zeroPoint 用对；
 *  - 能按输出形状自适应（v10 的端到端头和 v8 的原始头解码方式完全不同）；
 *  - 想去掉一层依赖就去掉，APK 从 56MB 降到 8MB。
 *
 * 输入约定：`[1, H, W, 3]`，letterbox 到正方形后归一化。
 * 输出约定：两种都支持
 *  - `[1, N, 6]`          YOLOv10 端到端头，每行 (x1, y1, x2, y2, score, classId)，已无重复框
 *  - `[1, 4+C, N]` / `[1, N, 4+C]`  YOLOv5/v8 原始头，需要自己做 NMS
 */
class YoloDetector(
    private val context: Context,
    private val scoreThreshold: Float = DEFAULT_SCORE_THRESHOLD,
    private val modelAsset: String = MODEL_ASSET,
    private val accel: Accel = Accel.CPU,
    private val numThreads: Int = NUM_THREADS
) {

    /** 推理后端。NNAPI 失败时自动退回 CPU，不会让整个检测挂掉。 */
    enum class Accel { CPU, CPU_BASIC, NNAPI }

    private var interpreter: Interpreter? = null

    /** 模型输入边长（正方形） */
    private var inputSize = 640

    private var inputType: DataType = DataType.FLOAT32
    private var inputScale = 0f
    private var inputZero = 0

    private var outputType: DataType = DataType.FLOAT32
    private var outputScale = 0f
    private var outputZero = 0

    /** 输出布局 */
    private var layout = Layout.END2END

    /** 端到端头：行数（最多 300 个候选） */
    private var rows = 0

    /** 原始头：通道数（4 + 类别数）与锚点/网格总数 */
    private var channels = 0
    private var anchors = 0

    /** 最近一次推理耗时 */
    @Volatile
    var lastInferenceMs: Long = 0
        private set

    /** 上次解码出来、分数最高的那个人的类别 id，用于核对模型是不是 COCO */
    @Volatile
    var lastClassId: Int = -1
        private set

    /** 坐标系推断结果：true 表示模型输出的是 0~1 归一化坐标 */
    @Volatile
    var normalizedCoords: Boolean? = null
        private set

    @Volatile
    var ready: Boolean = false
        private set

    var accelerator: String = "CPU(手写层)"
        private set

    /**
     * 写入模型前把 0~255 的通道值除以多少。
     *
     * 同一个 YOLO 权重，Ultralytics 不同导出版本对输入的期待不一样：
     *  - `255f` → 送 0~1（float 导出常见，也是训练时的真实输入分布）
     *  - `1f`   → 送 0~255（部分 integer_quant 导出把归一化折进了图里）
     * 猜错不会报错，只会安静地输出一堆无意义的低分框，所以只能实测区分。
     */
    var inputDenominator: Float = 255f

    /** 调试用：把输出表里分数最高的若干行原样吐出来 */
    fun rawDump(source: Bitmap, topN: Int = 6): String {
        val itp = interpreter ?: return "interpreter=null"
        val lb = fillInput(source)
        runCatching { itp.run(inputBuffer, outputBuffer()) }
            .onFailure { return "invoke failed: ${it.message}" }
        val sb = StringBuilder()

        // 先把所有输出张量的形状列清楚 —— 有的导出会拆成多个头
        sb.append("outputs=").append(itp.outputTensorCount)
        for (i in 0 until itp.outputTensorCount) {
            val t = itp.getOutputTensor(i)
            sb.append(" [").append(i).append("]=")
                .append(t.shape().contentToString()).append(t.dataType())
                .append("(s=").append(t.quantizationParams().scale)
                .append(",z=").append(t.quantizationParams().zeroPoint).append(')')
        }
        sb.append(" | layout=").append(layout)
            .append(" outType=").append(outputType)
            .append("(s=").append(outputScale).append(",z=").append(outputZero).append(')')
            .append(" inType=").append(inputType)
            .append("(s=").append(inputScale).append(",z=").append(inputZero).append(')')
            .append(" denom=").append(inputDenominator)
            .append(" lb=").append(lb.scale).append(',').append(lb.padX).append(',').append(lb.padY)

        // 再摊平一份，绕开对「哪一维是框、哪一维是候选」的任何假设
        val table = asFloatTable(outputBuffer())
        if (table != null) {
            val flat = ArrayList<Float>(2048)
            for (row in table[0]) for (v in row) flat += v
            sb.append("\n  flat(反量化后)[0..35]:")
            for (i in 0 until min(36, flat.size)) {
                if (i % 6 == 0) sb.append("\n   r").append(i / 6).append(':')
                sb.append(String.format(Locale.US, " %.3f", flat[i]))
            }
        }

        if (layout == Layout.END2END && table != null) {
            val top = table[0].sortedByDescending { it[4] }.take(topN)
            sb.append("\n  top scores:")
            for (r in top) {
                sb.append(String.format(Locale.US, "\n   s=%.4f cls=%.1f box=(%.3f,%.3f)-(%.3f,%.3f)", r[4], r[5], r[0], r[1], r[2], r[3]))
            }
        }
        return sb.toString()
    }

    private enum class Layout { END2END, CHANNELS_FIRST, CHANNELS_LAST }

    // 复用缓冲区，避免每次推理都分配几百 KB
    private var inputBuffer: ByteBuffer? = null
    private var pixelScratch = IntArray(0)
    private var canvasBitmap: Bitmap? = null
    private var canvas: Canvas? = null

    init {
        runCatching { load() }
            .onFailure { Log.e(TAG, "模型加载失败: $modelAsset", it) }
    }

    private fun load() {
        val buffer = mapAsset(modelAsset)
        val itp = createInterpreter(buffer) ?: error("没有任何后端能加载 $modelAsset")

        val inTensor = itp.getInputTensor(0)
        val inShape = inTensor.shape()
        inputSize = inShape[1]
        inputType = inTensor.dataType()
        inTensor.quantizationParams().let {
            inputScale = it.scale
            inputZero = it.zeroPoint
        }

        val outTensor = itp.getOutputTensor(0)
        val outShape = outTensor.shape()
        outputType = outTensor.dataType()
        outTensor.quantizationParams().let {
            outputScale = it.scale
            outputZero = it.zeroPoint
        }

        // 形状自适应：6 列且行数很大 → 端到端头；否则按通道/锚点判断
        when {
            outShape.size == 3 && outShape[2] == 6 -> {
                layout = Layout.END2END
                rows = outShape[1]
            }
            outShape.size == 3 && outShape[1] <= MAX_CHANNELS -> {
                layout = Layout.CHANNELS_FIRST
                channels = outShape[1]
                anchors = outShape[2]
            }
            else -> {
                layout = Layout.CHANNELS_LAST
                rows = outShape[1]
                channels = outShape[2]
            }
        }

        inputBuffer = ByteBuffer
            .allocateDirect(inputSize * inputSize * 3 * bytesPerElement(inputType))
            .order(ByteOrder.nativeOrder())
        pixelScratch = IntArray(inputSize * inputSize)
        canvasBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        canvas = Canvas(canvasBitmap!!)

        interpreter = itp
        ready = true
        Log.i(
            TAG,
            String.format(
                Locale.US,
                "yolo ready: accel=%s input=%dx%d %s out=%s layout=%s score>=%.2f",
                accelerator, inputSize, inputSize, inputType, outShape.contentToString(), layout, scoreThreshold
            )
        )
    }

    /**
     * 按 [accel] 依次尝试后端，谁先成功用谁，全失败就返回 null。
     *
     * 为什么必须写成回退链：NNAPI 的失败是**在 Interpreter 构造时抛异常**的
     * （不是运行时降级），一次失败就等于整个检测器不可用。而同一个模型在不同
     * ROM / 驱动上的可委派程度差别很大，只能试。
     *
     * 关于 GPU：v2 试过 `tensorflow-lite-gpu:2.14.0`，在这个混合量化模型上是
     * "Replacing 13 out of 352 node(s) ... Created 0 GPU delegate kernels"，
     * 构造直接失败 —— 2.14 的 GPU 委托对 int8 张量基本没有支持，
     * 而 YOLOv10 端到端头里的 CAST→INT64 / ADD v4 也不支持。既然一个内核都上不了，
     * 就没有理由为它背 13MB 的原生库，依赖已经移除。详见 docs/PERFORMANCE.md。
     */
    private fun createInterpreter(buffer: MappedByteBuffer): Interpreter? {
        val backends = when (accel) {
            Accel.NNAPI -> listOf("NNAPI", "CPU")
            Accel.CPU_BASIC -> listOf("BASIC")
            Accel.CPU -> listOf("CPU")
        }
        for (backend in backends) {
            val options = Interpreter.Options().apply { setNumThreads(numThreads) }
            val attempt = runCatching {
                when (backend) {
                    "NNAPI" -> {
                        options.setUseNNAPI(true)
                        options.setUseXNNPACK(true)
                    }
                    "BASIC" -> options.setUseXNNPACK(false)
                    else -> options.setUseXNNPACK(true)
                }
                Interpreter(buffer, options)
            }
            attempt.onSuccess {
                accelerator = when (backend) {
                    "NNAPI" -> "NNAPI"
                    "BASIC" -> "CPU(原生内核, 无 XNNPACK)"
                    else -> "CPU(XNNPACK)"
                }
                return it
            }.onFailure {
                Log.w(TAG, "后端 $backend 不可用，退到下一个: ${it.message?.lineSequence()?.firstOrNull()}")
            }
        }
        return null
    }

    private fun mapAsset(asset: String): MappedByteBuffer {
        val fd = context.assets.openFd(asset)
        FileInputStream(fd.fileDescriptor).use { stream ->
            stream.channel.use { channel ->
                return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    // ---------------------------------------------------------------- 对外

    /**
     * 在整张位图上检测一次。结果坐标是**原图像素坐标**。
     */
    fun detect(source: Bitmap, offsetX: Int = 0, offsetY: Int = 0): List<PersonCandidate> {
        val itp = interpreter ?: return emptyList()
        if (source.isRecycled || source.width <= 0 || source.height <= 0) return emptyList()

        val started = android.os.SystemClock.elapsedRealtime()
        val letterbox = fillInput(source)
        runCatching { itp.run(inputBuffer, outputBuffer()) }
            .onFailure { Log.w(TAG, "invoke failed", it); return emptyList() }

        val result = decode(outputBuffer(), letterbox, source.width, source.height, offsetX, offsetY)
        lastInferenceMs = android.os.SystemClock.elapsedRealtime() - started
        return result
    }

    /** 在已经裁好的区域上检测（选区放大识别用），坐标按 offset 平移回原图 */
    fun detectRegion(crop: Bitmap, offsetX: Int, offsetY: Int): List<PersonCandidate> =
        detect(crop, offsetX, offsetY)

    fun close() {
        runCatching { interpreter?.close() }
        interpreter = null
        ready = false
        canvasBitmap?.recycle()
        canvasBitmap = null
        canvas = null
    }

    // ---------------------------------------------------------------- 预处理

    private class Letterbox(val scale: Float, val padX: Int, val padY: Int)

    /**
     * 等比缩放 + 灰边填充到 inputSize×inputSize。
     *
     * 用 letterbox 而不是直接拉伸：Ultralytics 导出时训练/验证都用 letterbox，
     * 拉伸会让宽高比失真，检测框会系统性偏胖或偏瘦。
     */
    private fun fillInput(source: Bitmap): Letterbox {
        val srcW = source.width
        val srcH = source.height
        val scale = min(inputSize.toFloat() / srcW, inputSize.toFloat() / srcH)
        val drawW = (srcW * scale).roundToInt().coerceAtLeast(1)
        val drawH = (srcH * scale).roundToInt().coerceAtLeast(1)
        val padX = (inputSize - drawW) / 2
        val padY = (inputSize - drawH) / 2

        val target = canvasBitmap!!
        canvas!!.drawColor(Color.rgb(114, 114, 114))
        val srcRect = Rect(0, 0, srcW, srcH)
        val dstRect = Rect(padX, padY, padX + drawW, padY + drawH)
        canvas!!.drawBitmap(source, srcRect, dstRect, null)
        target.getPixels(pixelScratch, 0, inputSize, 0, 0, inputSize, inputSize)

        val buf = inputBuffer!!
        buf.rewind()
        val quantized = inputType != DataType.FLOAT32
        val invScale = if (quantized && inputScale > 0f) 1f / inputScale else 0f

        if (!quantized) {
            val denom = if (inputDenominator <= 0f) 255f else inputDenominator
            for (color in pixelScratch) {
                buf.putFloat(((color shr 16) and 0xFF) / denom)
                buf.putFloat(((color shr 8) and 0xFF) / denom)
                buf.putFloat((color and 0xFF) / denom)
            }
        } else {
            val denom = if (inputDenominator <= 0f) 255f else inputDenominator
            for (color in pixelScratch) {
                buf.put(quantize(((color shr 16) and 0xFF) / denom, invScale))
                buf.put(quantize(((color shr 8) and 0xFF) / denom, invScale))
                buf.put(quantize((color and 0xFF) / denom, invScale))
            }
        }
        buf.rewind()
        return Letterbox(scale, padX, padY)
    }

    /**
     * 按张量的真实 dtype 做量化写入。
     *
     * 这里曾经把两种整数类型当成一种处理：一律 `coerceIn(-128, 127)`。
     * 对 INT8 输入这是对的；对 UINT8 输入（全量化导出最常见的形式，zeroPoint=0）
     * 就是灾难 —— 值域本来是 0~255，钳到 127 相当于把所有亮度超过 50% 的像素压成同一个值，
     * 送进网络的是一张糊掉的图，输出自然全是常数。而且它不会报错，只会安静地低分。
     */
    private fun quantize(value: Float, invScale: Float): Byte {
        val raw = value * invScale + inputZero
        return if (inputType == DataType.UINT8) {
            raw.roundToInt().coerceIn(0, 255).toByte()
        } else {
            raw.roundToInt().coerceIn(MIN_Q, MAX_Q).toByte()
        }
    }

    // ---------------------------------------------------------------- 后处理

    private var cachedOutput: Any? = null
    private var cachedOutputShape: IntArray? = null

    private fun outputBuffer(): Any {
        val cached = cachedOutput
        if (cached != null) return cached

        val shape = when (layout) {
            Layout.END2END -> intArrayOf(1, rows, 6)
            Layout.CHANNELS_FIRST -> intArrayOf(1, channels, anchors)
            Layout.CHANNELS_LAST -> intArrayOf(1, rows, channels)
        }
        val created: Any = if (outputType == DataType.FLOAT32) {
            Array(1) { Array(shape[1]) { FloatArray(shape[2]) } }
        } else {
            Array(1) { Array(shape[1]) { ByteArray(shape[2]) } }
        }
        cachedOutput = created
        cachedOutputShape = shape
        return created
    }

    private fun decode(
        raw: Any,
        lb: Letterbox,
        srcW: Int,
        srcH: Int,
        offsetX: Int,
        offsetY: Int
    ): List<PersonCandidate> {
        val boxes = ArrayList<PersonCandidate>(8)
        val floatView = asFloatTable(raw) ?: return emptyList()
        when (layout) {
            Layout.END2END -> decodeEnd2End(floatView, lb, srcW, srcH, offsetX, offsetY, boxes)
            Layout.CHANNELS_FIRST -> decodeChannelsFirst(floatView, lb, srcW, srcH, offsetX, offsetY, boxes)
            Layout.CHANNELS_LAST -> decodeChannelsLast(floatView, lb, srcW, srcH, offsetX, offsetY, boxes)
        }
        val suppressed = if (layout == Layout.END2END) boxes else nonMaximumSuppression(boxes)
        return suppressed.sortedByDescending { it.score }.take(MAX_CANDIDATES)
    }

    /**
     * 把输出表统一成 float。
     *
     * 全整数量化模型的输出张量是 int8，直接按 ByteArray 读回来是量化后的整数，
     * 不反量化的话分数会变成 ±128 的整数、坐标也全是垃圾值。这里用张量自带的
     * scale / zeroPoint 做一次 (q - zeroPoint) * scale。
     */
    private fun asFloatTable(raw: Any): Array<Array<FloatArray>>? {
        @Suppress("UNCHECKED_CAST")
        if (raw is Array<*>) {
            val first = raw[0]
            if (first is Array<*> && (first.isNotEmpty()) && first[0] is FloatArray) {
                return raw as Array<Array<FloatArray>>
            }
        }
        @Suppress("UNCHECKED_CAST")
        val bytes = raw as? Array<Array<ByteArray>> ?: return null
        val scale = if (outputScale > 0f) outputScale else 1f
        val out = Array(bytes.size) { b ->
            Array(bytes[b].size) { r -> FloatArray(bytes[b][r].size) }
        }
        for (b in bytes.indices) {
            for (r in bytes[b].indices) {
                val src = bytes[b][r]
                val dst = out[b][r]
                for (c in src.indices) {
                    // INT8 是有符号的，UINT8 要按无符号展开；弄反会让零点偏移 256 个量化级
                    val q = if (outputType == DataType.INT8) src[c].toInt() else (src[c].toInt() and 0xFF)
                    dst[c] = (q - outputZero) * scale
                }
            }
        }
        return out
    }

    /** `[1, N, 6]`：每行 (x1, y1, x2, y2, score, classId)，已经是 NMS 后的结果 */
    private fun decodeEnd2End(
        raw: Any,
        lb: Letterbox,
        srcW: Int,
        srcH: Int,
        offsetX: Int,
        offsetY: Int,
        out: MutableList<PersonCandidate>
    ) {
        val table = raw as Array<Array<FloatArray>>
        var normalizedVote = true
        for (row in table[0]) {
            val score = row[4]
            if (score < scoreThreshold) continue
            val classId = row[5].roundToInt()
            if (max(max(row[0], row[1]), max(row[2], row[3])) > NORMALIZED_LIMIT) normalizedVote = false
            if (classId != PERSON_CLASS_ID) {
                lastClassId = classId
                continue
            }
            lastClassId = classId
            val candidate = toCandidate(row[0], row[1], row[2], row[3], score, lb, srcW, srcH, offsetX, offsetY)
                ?: continue
            out += candidate
        }
        normalizedCoords = normalizedVote
    }

    /** `[1, 4+C, N]`：YOLOv5/v8 原始头，通道优先 */
    private fun decodeChannelsFirst(
        raw: Any,
        lb: Letterbox,
        srcW: Int,
        srcH: Int,
        offsetX: Int,
        offsetY: Int,
        out: MutableList<PersonCandidate>
    ) {
        val table = raw as Array<Array<FloatArray>>
        var normalizedVote = true
        for (i in 0 until anchors) {
            // 只取 person 类：COCO 的 0 号；单类自定义模型也只有 0 号
            val scoreIndex = 4 + PERSON_CLASS_ID
            if (scoreIndex >= channels) continue
            val score = table[0][scoreIndex][i]
            if (score < scoreThreshold) continue

            val cx = table[0][0][i]
            val cy = table[0][1][i]
            val w = table[0][2][i]
            val h = table[0][3][i]
            if (max(max(cx, cy), max(w, h)) > NORMALIZED_LIMIT) normalizedVote = false

            val candidate = toCandidate(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f, score, lb, srcW, srcH, offsetX, offsetY)
                ?: continue
            out += candidate
        }
        normalizedCoords = normalizedVote
    }

    /** `[1, N, 4+C]`：通道在后 */
    private fun decodeChannelsLast(
        raw: Any,
        lb: Letterbox,
        srcW: Int,
        srcH: Int,
        offsetX: Int,
        offsetY: Int,
        out: MutableList<PersonCandidate>
    ) {
        val table = raw as Array<Array<FloatArray>>
        var normalizedVote = true
        val scoreIndex = 4 + PERSON_CLASS_ID
        if (scoreIndex >= channels) return
        for (row in table[0]) {
            val score = row[scoreIndex]
            if (score < scoreThreshold) continue
            val cx = row[0]; val cy = row[1]; val w = row[2]; val h = row[3]
            if (max(max(cx, cy), max(w, h)) > NORMALIZED_LIMIT) normalizedVote = false
            val candidate = toCandidate(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f, score, lb, srcW, srcH, offsetX, offsetY)
                ?: continue
            out += candidate
        }
        normalizedCoords = normalizedVote
    }

    /**
     * letterbox 坐标 → 原图像素坐标。
     *
     * 顺带处理「模型输出归一化坐标」的情况：导出配置不同，有的给 0~1，有的给 0~inputSize。
     * 靠数值量级判断（人不可能只有 1.5 像素高），并在 [normalizedCoords] 里记下来。
     */
    private fun toCandidate(
        x1: Float, y1: Float, x2: Float, y2: Float, score: Float,
        lb: Letterbox, srcW: Int, srcH: Int, offsetX: Int, offsetY: Int
    ): PersonCandidate? {
        val normalized = max(max(x1, y1), max(x2, y2)) <= NORMALIZED_LIMIT
        val mul = if (normalized) inputSize.toFloat() else 1f

        val left = ((x1 * mul) - lb.padX) / lb.scale + offsetX
        val top = ((y1 * mul) - lb.padY) / lb.scale + offsetY
        val right = ((x2 * mul) - lb.padX) / lb.scale + offsetX
        val bottom = ((y2 * mul) - lb.padY) / lb.scale + offsetY

        val box = RectF(
            left.coerceIn(0f, (srcW + offsetX).toFloat()),
            top.coerceIn(0f, (srcH + offsetY).toFloat()),
            right.coerceIn(0f, (srcW + offsetX).toFloat()),
            bottom.coerceIn(0f, (srcH + offsetY).toFloat())
        )
        if (box.width() < MIN_BOX_PX || box.height() < MIN_BOX_PX) return null
        return PersonCandidate(
            box = box,
            score = score.coerceIn(0f, 1f),
            source = DetectionSource.DETECTOR,
            evidence = Evidence(detectorScore = score, classId = PERSON_CLASS_ID)
        )
    }

    private fun nonMaximumSuppression(input: List<PersonCandidate>): List<PersonCandidate> {
        if (input.size <= 1) return input
        val sorted = input.sortedByDescending { it.score }
        val kept = ArrayList<PersonCandidate>(sorted.size)
        for (candidate in sorted) {
            if (kept.none { iou(it.box, candidate.box) > NMS_IOU }) kept += candidate
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        if (right <= left || bottom <= top) return 0f
        val inter = (right - left) * (bottom - top)
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }

    private fun bytesPerElement(type: DataType): Int = when (type) {
        DataType.FLOAT32 -> 4
        DataType.UINT8, DataType.INT8 -> 1
        else -> 4
    }

    companion object {
        private const val TAG = "YoloDetector"
        const val MODEL_ASSET = "yolov10n_int8.tflite"

        /** COCO 里 person 是第 0 类；单类自定义模型同样只有 0 号 */
        const val PERSON_CLASS_ID = 0

        private const val NUM_THREADS = 4
        private const val MAX_CANDIDATES = 16
        private const val NMS_IOU = 0.55f

        /** 低于这个分数的候选直接丢，融合层再做二次判定 */
        const val DEFAULT_SCORE_THRESHOLD = 0.25f

        /** 坐标轴上限判断：超过 1.5 就认为是像素坐标而不是归一化坐标 */
        private const val NORMALIZED_LIMIT = 1.5f

        private const val MIN_BOX_PX = 4f
        private const val MIN_Q = -128
        private const val MAX_Q = 127

        /** 通道数上限，用来区分「通道优先」和「行优先」两种布局 */
        private const val MAX_CHANNELS = 256
    }
}
