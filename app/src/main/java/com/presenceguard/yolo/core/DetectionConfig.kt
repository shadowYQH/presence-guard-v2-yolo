package com.presenceguard.yolo.core

/**
 * 灵敏度档位。三档只是同一套算法的不同阈值组合，不存在"另一套逻辑"，
 * 这样现场调参时行为可预期。
 *
 * v2 只有一路 YOLO 检测，所以阈值重心从"两个引擎互相印证"移到了"单帧分数 + 时间证据"。
 * 三档之间的差异主要体现在**敢不敢收低分框**和**要求连续几帧**这两件事上。
 */
enum class Sensitivity(val displayName: String, val hint: String) {
    HIGH("灵敏", "远处的小个子、只露出半个身子也会报，代价是偶尔误报"),
    MEDIUM("均衡", "室内正常距离的人都能稳定识别，误报很少（推荐）"),
    LOW("稳重", "只在人足够大、特征足够明确时才报，几乎不误报");

    fun toConfig(): DetectionConfig = when (this) {
        HIGH -> DetectionConfig(
            enterScore = 0.34f,
            exitScore = 0.24f,
            votesToEnter = 3,
            votesToExit = 10,
            minAreaRatio = 0.0012f,
            smallTargetAreaRatio = 0.020f,
            smallTargetMinScore = 0.26f,
            smallTargetMinHits = 2,
            smallTargetScoreFloor = 0.44f
        )
        MEDIUM -> DetectionConfig(
            enterScore = 0.46f,
            exitScore = 0.32f,
            votesToEnter = 4,
            votesToExit = 12,
            minAreaRatio = 0.0030f,
            smallTargetAreaRatio = 0.015f,
            smallTargetMinScore = 0.32f,
            smallTargetMinHits = 3,
            smallTargetScoreFloor = 0.52f
        )
        LOW -> DetectionConfig(
            enterScore = 0.62f,
            exitScore = 0.45f,
            votesToEnter = 5,
            votesToExit = 14,
            minAreaRatio = 0.010f,
            smallTargetAreaRatio = 0.012f,
            smallTargetMinScore = 0.45f,
            smallTargetMinHits = 5,
            smallTargetScoreFloor = 0.66f
        )
    }
}

/**
 * 检测参数。默认值即"均衡"档。
 *
 * 判定链路：单帧分数 → 几何合理性 → 面积门限 → 小目标轨迹持久化 → 滑动窗口投票 → 迟滞状态机。
 *
 * 和 v1 的区别：v1 靠两个引擎在同一位置互相印证来压误报，v2 只有一路检测，
 * 于是把"可信度"的第二个来源换成了**时间**——同一位置连续出现才算数。
 * 这也是为什么 [smallTargetMinHits] 和 [votesToEnter] 在 v2 里权重更高。
 */
data class DetectionConfig(
    /** 单帧分数达到该值即视为"本帧有人" */
    val enterScore: Float = 0.46f,
    /** 平滑分数低于该值才可能退出；比 [enterScore] 低，构成迟滞，避免边界抖动 */
    val exitScore: Float = 0.32f,
    /** 滑动窗口帧数（约 5fps 时 8 帧 ≈ 1.6 秒） */
    val windowSize: Int = 8,
    /** 窗口内至少这么多帧命中才确认有人 */
    val votesToEnter: Int = 4,
    /** 连续这么多帧无目标才解除 */
    val votesToExit: Int = 12,

    /** 目标框面积占画面比例下限，过滤远处噪点与墙角误检 */
    val minAreaRatio: Float = 0.0030f,
    /**
     * 单引擎目标框的面积上限。超过这个比例基本只有两种可能：
     * 贴着镜头拍的人，或者无纹理的纯色面被当成了人。交给投票环节处理，不在这里否决。
     */
    val maxAreaRatio: Float = 0.94f,

    // ---------------------------------------------------------------- 几何
    /**
     * 目标框的宽高比（高 / 宽）合法区间。
     *
     * 只拦"离谱"的形状：横得超过 6.7:1（像一根横杆）、竖得超过 12:1（像一根立柱）。
     * 人躺着、坐着、只露一条腿都在这个区间内，所以不会误伤。
     * v1 的教训是别在这里加"人必须是竖直的"这类朝向相关约束。
     */
    val minHeightWidthRatio: Float = 0.15f,
    val maxHeightWidthRatio: Float = 12f,

    // ---------------------------------------------------------------- 小目标
    /**
     * 面积占比低于该值视为"小目标"，改用跨帧轨迹判定。
     * 1080×1440 下 0.015 ≈ 23000 像素，约等于 10 米外人体的占屏比例。
     */
    val smallTargetAreaRatio: Float = 0.015f,
    /** 小目标的单帧原始分下限（比正常人低得多，靠持续命中兜底） */
    val smallTargetMinScore: Float = 0.32f,
    /** 小目标需要在同一位置连续命中这么多帧才采信 */
    val smallTargetMinHits: Int = 3,
    /** 轨迹确认后给予的分数下限，让它能正常推动状态机 */
    val smallTargetScoreFloor: Float = 0.52f,

    /** 两次报警之间的最短间隔 */
    val cooldownMs: Long = 3500L,
    /** 已在场的人数变多时是否再响一次 */
    val alertOnCountIncrease: Boolean = true
)
