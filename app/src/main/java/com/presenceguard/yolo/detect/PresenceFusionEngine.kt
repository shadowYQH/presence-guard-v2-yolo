package com.presenceguard.yolo.detect

import android.graphics.RectF
import com.presenceguard.yolo.core.DetectionConfig
import com.presenceguard.yolo.core.DetectionSource
import com.presenceguard.yolo.core.PersonCandidate
import kotlin.math.max
import kotlin.math.min

/**
 * 单引擎（YOLO）判定链 + 时序状态机。这是 v2 里决定"准不准"的部分。
 *
 * v1 靠两个引擎在同一位置互相印证来压误报；v2 只有一路检测，于是把可信度的
 * 第二个来源换成了**时间**。四层过滤，逐层消除不同类型的错误：
 *
 * 1) 形状门限 —— 只拦离谱的形状（像横杆、像立柱），不用"人必须竖直"这类
 *    朝向相关约束。v1 在这上面栽过：那条规则会把躺着、侧身、横屏拍摄的人全判死。
 *
 * 2) 面积门限 —— 太小的是远处噪点；太大且分数不高的多半是无纹理的纯色面。
 *
 * 3) 小目标轨迹持久化 —— 远处的人单帧分数天然偏低，靠"低门槛 + 同一位置连续命中"
 *    用时间证据换空间置信度。误检往往偶发、位置飘忽，真人会稳定出现在同一位置。
 *
 * 4) 滑动窗口投票 + 迟滞 —— 单帧命中不算数；退出比进入苛刻得多。
 */
class PresenceFusionEngine {

    enum class Event { NONE, APPEARED, COUNT_INCREASED, CLEARED }

    data class Outcome(
        /** 状态机判定：画面里是否有人 */
        val present: Boolean,
        /** 本帧识别到的人数（已去重） */
        val personCount: Int,
        /** 本帧最高分 */
        val frameScore: Float,
        /** 平滑后的分数，UI 上用它展示"把握程度" */
        val smoothedScore: Float,
        /** 窗口命中率 0~1 */
        val hitRate: Float,
        /** 窗口内命中帧数 */
        val votes: Int,
        val event: Event,
        /** 本次是否需要响提示音（已应用冷却） */
        val shouldAlert: Boolean,
        val candidates: List<PersonCandidate>
    )

    private val window = ArrayDeque<Boolean>()
    private var smoothed = 0f
    private var present = false
    private var lastCount = 0
    private var emptyStreak = 0
    private var lastAlertAt = 0L

    /** 小目标的短期轨迹记忆，用于跨帧确认 */
    private val tracks = ArrayList<Track>()

    fun reset() {
        window.clear()
        smoothed = 0f
        present = false
        lastCount = 0
        emptyStreak = 0
        tracks.clear()
    }

    fun update(
        detections: List<PersonCandidate>,
        sourceWidth: Int,
        sourceHeight: Int,
        now: Long,
        cfg: DetectionConfig
    ): Outcome {
        val frameArea = (sourceWidth * sourceHeight).toFloat().coerceAtLeast(1f)
        val kept = pipeline(detections, sourceWidth, sourceHeight, frameArea, now, cfg)

        val frameScore = kept.maxOfOrNull { it.score } ?: 0f
        val hit = frameScore >= cfg.enterScore

        window.addLast(hit)
        while (window.size > cfg.windowSize) window.removeFirst()

        val votes = window.count { it }
        val hitRate = votes.toFloat() / window.size
        emptyStreak = if (hit) 0 else emptyStreak + 1

        // EMA 平滑：单帧分数突然拉高不会立刻触发，必须持续有证据
        smoothed = if (smoothed <= 0f && frameScore <= 0f) 0f else smoothed * 0.62f + frameScore * 0.38f

        var event = Event.NONE
        if (!present) {
            val enoughVotes = votes >= cfg.votesToEnter
            val enoughScore = smoothed >= cfg.enterScore
            if (enoughVotes && enoughScore) {
                present = true
                lastCount = kept.size
                event = Event.APPEARED
            }
        } else {
            if (emptyStreak >= cfg.votesToExit && smoothed < cfg.exitScore) {
                present = false
                lastCount = 0
                event = Event.CLEARED
            } else if (cfg.alertOnCountIncrease && kept.size > lastCount && kept.isNotEmpty()) {
                lastCount = kept.size
                event = Event.COUNT_INCREASED
            }
        }

        var shouldAlert = false
        if (event == Event.APPEARED || event == Event.COUNT_INCREASED) {
            if (now - lastAlertAt >= cfg.cooldownMs) {
                shouldAlert = true
                lastAlertAt = now
            }
        }

        val displayCount = if (present) max(1, kept.size) else kept.size

        return Outcome(
            present = present,
            personCount = displayCount,
            frameScore = frameScore,
            smoothedScore = smoothed,
            hitRate = hitRate,
            votes = votes,
            event = event,
            shouldAlert = shouldAlert,
            candidates = kept
        )
    }

    // ---------------------------------------------------------------- 判定链

    private fun pipeline(
        input: List<PersonCandidate>,
        sourceWidth: Int,
        sourceHeight: Int,
        frameArea: Float,
        now: Long,
        cfg: DetectionConfig
    ): List<PersonCandidate> {
        // 1) 形状 + 面积门限
        val gated = ArrayList<PersonCandidate>(input.size)
        for (candidate in input) {
            val box = clampToFrame(candidate.box, sourceWidth, sourceHeight)
            if (box.width() <= 1f || box.height() <= 1f) continue

            val ratio = box.width() * box.height() / frameArea
            if (ratio < cfg.minAreaRatio || ratio > cfg.maxAreaRatio) continue

            val shape = box.height() / box.width()
            if (shape < cfg.minHeightWidthRatio || shape > cfg.maxHeightWidthRatio) continue

            gated += candidate.copy(box = box, evidence = candidate.evidence.copy(areaRatio = ratio))
        }
        if (gated.isEmpty()) {
            pruneTracks(now)
            return emptyList()
        }

        // 2) 去重：多个选区或瓦片可能对同一个人各报一次
        val deduped = nonMaximumSuppression(gated)

        // 3) 小目标轨迹确认
        return confirmTracks(deduped, now, cfg)
    }

    /**
     * 小目标轨迹持久化。
     *
     * 远距离目标即使被 640 输入的模型看到了，分数也常常只有 0.3~0.4。用"门槛 + 单帧分数"
     * 几乎无法与杂物区分，但两者在时间维度上差别很大：真人会连续出现在同一位置，
     * 而墙面图案、衣架这类误检往往偶发且位置飘忽。
     */
    private fun confirmTracks(
        candidates: List<PersonCandidate>,
        now: Long,
        cfg: DetectionConfig
    ): List<PersonCandidate> {
        pruneTracks(now)

        val passed = ArrayList<PersonCandidate>(candidates.size)
        for (candidate in candidates) {
            val ratio = candidate.evidence.areaRatio

            // 够大的目标、分数也够 → 单帧即可采信，不必等轨迹
            if (ratio >= cfg.smallTargetAreaRatio && candidate.score >= cfg.enterScore) {
                passed += candidate
                continue
            }

            val track = tracks.firstOrNull { iou(it.box, candidate.box) >= TRACK_IOU }
            if (track == null) {
                tracks += Track(RectF(candidate.box), 1, now)
                // 面积不小但分数偏低：先记账，下一帧再看。这里不直接丢，
                // 否则人在远处走近的过程中会有一段"看得见但不算数"的空窗。
                continue
            }

            track.hits++
            track.box.set(candidate.box)
            track.lastSeen = now

            if (track.hits >= cfg.smallTargetMinHits && candidate.score >= cfg.smallTargetMinScore) {
                passed += candidate.copy(
                    score = max(candidate.score, cfg.smallTargetScoreFloor),
                    source = DetectionSource.CONFIRMED,
                    evidence = candidate.evidence.copy(trackHits = track.hits)
                )
            }
        }
        return passed
    }

    private fun pruneTracks(now: Long) {
        tracks.removeAll { now - it.lastSeen > TRACK_TTL_MS }
    }

    /** 一条轨迹：位置 + 连续命中次数 + 最后出现时间 */
    private class Track(var box: RectF, var hits: Int, var lastSeen: Long)

    /** 检测框可能超出画面边界（模型回归不受约束），先裁齐再算面积占比 */
    private fun clampToFrame(box: RectF, width: Int, height: Int): RectF {
        val w = width.toFloat()
        val h = height.toFloat()
        return RectF(
            box.left.coerceIn(0f, w),
            box.top.coerceIn(0f, h),
            box.right.coerceIn(0f, w),
            box.bottom.coerceIn(0f, h)
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

    companion object {
        private const val NMS_IOU = 0.60f

        /** 轨迹匹配阈值：松一点，容忍远处目标的抖动 */
        private const val TRACK_IOU = 0.20f

        /** 轨迹存活时间。5fps 下约等于 5 帧，够攒够 smallTargetMinHits */
        private const val TRACK_TTL_MS = 1200L
    }
}
