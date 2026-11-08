package com.presenceguard.yolo.core

import android.graphics.RectF

/**
 * 检测结果的来源。
 *
 * v2 只有一路外观检测（YOLO），所以这里不再区分"骨架/轮廓"两种引擎来源，
 * 而是区分**证据强度**：
 *  - [DETECTOR] 单帧检测结果，还没被时间维度确认；
 *  - [CONFIRMED] 已经通过跨帧轨迹确认的目标（远处小目标必须走到这一步）。
 *
 * 这个区分在界面上是有意义的：橙色框表示"模型说是人，但只看到一帧"，
 * 绿色框表示"连续多帧都在同一位置看到人"。现场调试时一眼能看出误报是怎么来的。
 */
enum class DetectionSource { DETECTOR, CONFIRMED }

/** 打分依据，用于界面解释"为什么判定为有人"，也便于现场调参。 */
data class Evidence(
    /** YOLO 的原始置信度 */
    val detectorScore: Float = 0f,
    /** 目标框面积占画面比例 */
    val areaRatio: Float = 0f,
    /** 模型给出的类别 id（COCO 的 person = 0） */
    val classId: Int = 0,
    /** 该目标在轨迹上累计命中帧数 */
    val trackHits: Int = 0
)

/**
 * 一个人体候选目标。[box] 是旋转校正后图像的像素坐标。
 */
data class PersonCandidate(
    val box: RectF,
    val score: Float,
    val source: DetectionSource = DetectionSource.DETECTOR,
    val evidence: Evidence = Evidence()
) {
    val centerX: Float get() = box.centerX()
    val centerY: Float get() = box.centerY()

    fun area(): Float = if (box.isEmpty) 0f else box.width() * box.height()
}

/** 一帧的完整检测结果，直接交给渲染层使用。 */
data class FrameResult(
    val candidates: List<PersonCandidate>,
    /** 旋转校正后图像尺寸，用于把坐标映射到 View */
    val sourceWidth: Int,
    val sourceHeight: Int,
    /** 是否来自前置摄像头（叠加层需要水平镜像才能与预览对齐） */
    val mirrored: Boolean,
    val processingMs: Long,
    val fps: Float = 0f
)
