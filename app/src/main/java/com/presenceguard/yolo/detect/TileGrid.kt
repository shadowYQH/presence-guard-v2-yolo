package com.presenceguard.yolo.detect

import android.graphics.Rect
import kotlin.math.roundToInt

/**
 * 瓦片网格。
 *
 * 为什么需要它：模型的输入固定为 640×640，整帧缩放后远处的人只剩十几像素，特征消失；
 * **提高采集分辨率没有用**，因为缩放会把增益抵消掉。唯一有效的手段是缩小送进模型的视野
 * ——把画面切成瓦片，人在这片里占的比例被放大约 [放大倍数] 倍。
 *
 * 放大倍数只取决于瓦片数量，与采集分辨率无关（分辨率只决定放大后是否还清晰）。
 *
 * [overlap] 是相邻瓦片的重叠比例，避免人正好跨在接缝上被切成两半。
 */
data class TileGrid(
    val cols: Int,
    val rows: Int,
    val overlap: Float = 0.25f
) {
    fun rects(width: Int, height: Int): List<Rect> {
        if (width <= 0 || height <= 0 || cols <= 0 || rows <= 0) return emptyList()

        val tileW = (width.toFloat() / (cols - (cols - 1) * overlap))
            .roundToInt().coerceIn(1, width)
        val tileH = (height.toFloat() / (rows - (rows - 1) * overlap))
            .roundToInt().coerceIn(1, height)

        val stepX = if (cols > 1) (width - tileW).toFloat() / (cols - 1) else 0f
        val stepY = if (rows > 1) (height - tileH).toFloat() / (rows - 1) else 0f

        val out = ArrayList<Rect>(cols * rows)
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                val left = (c * stepX).roundToInt().coerceIn(0, width - tileW)
                val top = (r * stepY).roundToInt().coerceIn(0, height - tileH)
                out += Rect(left, top, left + tileW, top + tileH)
            }
        }
        return out
    }

    val tileCount: Int get() = cols * rows

    companion object {
        /** 默认：3 列 × 2 行，相邻重叠 25%，共 6 片（每片约占 1/2.5 边长） */
        val DEFAULT = TileGrid(cols = 3, rows = 2)
    }
}
