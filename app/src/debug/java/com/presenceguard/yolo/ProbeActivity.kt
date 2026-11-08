package com.presenceguard.yolo

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import com.presenceguard.yolo.detect.TileGrid
import com.presenceguard.yolo.detect.YoloDetector
import java.util.Locale

/**
 * 离屏验收探针（仅 debug 构建）。
 *
 * 用途：拿固定样图跑一遍完整流程，把结果打到 logcat。做算法改动前后可以客观对比，
 * 不靠肉眼盯着屏幕猜。正式包里不存在这个类。
 *
 * ```
 * adb shell am start -n com.presenceguard.yolo/.ProbeActivity \
 *     --es image probe_far_lake.jpg --ei tiles 1
 * ```
 *
 * [tiles]=1 时额外做一次全瓦片扫描（每片单独推理），否则只跑整帧。
 */
class ProbeActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // 必须有一个真正绘制出来的窗口：没有内容视图时系统会判定这个 Activity
        // 没有完成绘制，直接把前台让给桌面，进程随即掉出 top-app cpuset。
        setContentView(
            android.widget.TextView(this).apply {
                text = "离屏验收探针运行中…"
                textSize = 18f
                setPadding(48, 96, 48, 48)
                setBackgroundColor(android.graphics.Color.WHITE)
                setTextColor(android.graphics.Color.BLACK)
            }
        )
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val asset = intent?.getStringExtra("image") ?: "probe_far_plaza.jpg"
        val sweep = intent?.getIntExtra("tiles", 0) == 1
        Thread {
            runCatching { probe(asset, sweep) }
                .onFailure { Log.e(TAG, "probe failed", it) }
            // 不能在这里之前 finish()：Activity 一没了进程就掉进后台 cpuset，
            // 只留 0-3 四个小核，量出来的耗时能差三四倍（这个坑踩过，见 docs/PERFORMANCE.md）。
            // 而且必须延后：如果探针秒退，finish() 会在 onResume 还没走完时执行，
            // 直接抛 "did not call finish() prior to onResume() completing"。
            runOnUiThread {
                android.os.Handler(mainLooper).postDelayed({ finish() }, 800)
            }
        }.start()
    }

    private fun probe(asset: String, sweep: Boolean) {
        val source = assets.open(asset).use { BitmapFactory.decodeStream(it) }
            ?: run { Log.e(TAG, "decode failed: $asset"); return }

        val accel = when (intent?.getStringExtra("accel")) {
            "nnapi" -> YoloDetector.Accel.NNAPI
            "basic" -> YoloDetector.Accel.CPU_BASIC
            else -> YoloDetector.Accel.CPU
        }
        val detector = YoloDetector(
            applicationContext,
            accel = accel,
            numThreads = intent?.getIntExtra("threads", 4) ?: 4
        )
        if (!detector.ready) {
            Log.e(TAG, "模型未就绪")
            return
        }

        Log.i(TAG, "== $asset ${source.width}x${source.height} accel=${detector.accelerator} threads=${intent?.getIntExtra("threads", 4) ?: 4} ==")
        Log.i(TAG, "cpuset=${cpusAllowed()}")

        // 0) 输入归一化方式辨识：两种都跑一遍，看谁输出的分数像话
        if (intent?.getBooleanExtra("diag", false) == true) {
            for (denom in floatArrayOf(255f, 1f)) {
                detector.inputDenominator = denom
                Log.i(TAG, "diag denom=$denom ->\n${detector.rawDump(source)}")
            }
            detector.inputDenominator = 255f
            detector.close()
            if (!source.isRecycled) source.recycle()
            return
        }

        // 1) 整帧
        val full = detector.detect(source)
        Log.i(TAG, "整帧: ${full.size} 个候选, ${detector.lastInferenceMs}ms, " +
            "coords=${detector.normalizedCoords}, class=${detector.lastClassId}")
        dump(full)

        // 2) 全瓦片扫描
        if (sweep) {
            val grid = TileGrid(
                cols = intent?.getIntExtra("cols", 0)?.takeIf { it > 0 } ?: TileGrid.DEFAULT.cols,
                rows = intent?.getIntExtra("rows", 0)?.takeIf { it > 0 } ?: TileGrid.DEFAULT.rows
            )
            val collected = ArrayList<com.presenceguard.yolo.core.PersonCandidate>()
            var totalMs = 0L
            for (tile in grid.rects(source.width, source.height)) {
                val crop = Bitmap.createBitmap(source, tile.left, tile.top, tile.width(), tile.height())
                try {
                    collected += detector.detectRegion(crop, tile.left, tile.top)
                    totalMs += detector.lastInferenceMs
                } finally {
                    if (!crop.isRecycled) crop.recycle()
                }
            }
            Log.i(
                TAG,
                "瓦片扫描: ${grid.tileCount} 片, ${collected.size} 个候选, 累计 ${totalMs}ms, " +
                    "平均 ${totalMs / grid.tileCount}ms/片"
            )
            dump(collected.sortedByDescending { it.score }.take(10))
        }

        // 3) 单帧耗时基准（8 次取中位）
        repeat(8) { detector.detect(source) }
        val times = ArrayList<Long>(8)
        repeat(8) {
            detector.detect(source)
            times += detector.lastInferenceMs
        }
        times.sort()
        Log.i(TAG, "基准: 中位 ${times[times.size / 2]}ms, 最小 ${times.first()}ms, 最大 ${times.last()}ms")

        detector.close()
        if (!source.isRecycled) source.recycle()
        Log.i(TAG, "== 结束 ==")
    }

    private fun cpusAllowed(): String = runCatching {
        java.io.File("/proc/self/status").readLines()
            .firstOrNull { it.startsWith("Cpus_allowed_list") }
            ?.substringAfter(':')?.trim() ?: "?"
    }.getOrDefault("?")

    private fun dump(list: List<com.presenceguard.yolo.core.PersonCandidate>) {
        for (c in list) {
            Log.i(
                TAG,
                String.format(
                    Locale.US,
                    "  score=%.3f cls=%d box=(%.0f,%.0f)-(%.0f,%.0f) wh=%.0fx%.0f",
                    c.score, c.evidence.classId,
                    c.box.left, c.box.top, c.box.right, c.box.bottom,
                    c.box.width(), c.box.height()
                )
            )
        }
    }

    private companion object {
        const val TAG = "Probe"
    }
}
