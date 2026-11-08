#pragma once

#include <algorithm>
#include <vector>

#include "safety_engine/types.hpp"

namespace safety_engine {

enum class AlertState {
    Safe = 0,
    SuspectedViolation = 1,
    ActiveAlert = 2
};

enum class DropoutStreakPolicy {
    HoldStreak = 0,
    StrictReset = 1,
    DecayStreak = 2
};

struct TrackedWorker {
    int track_id{0};
    WorkerStatus status;
    int age_frames{0};
    int missed_frames{0};
    int violation_streak{0};
    int consecutive_observed_violations{0};
    int total_violation_frames{0};
    float vx{0.0f};
    float vy{0.0f};
    float vw{0.0f};
    float vh{0.0f};
    AlertState alert_state{AlertState::Safe};

    cv::Rect2f predict_next_bbox() const noexcept {
        const float steps = static_cast<float>(missed_frames + 1);
        const float pred_w = std::max(0.0f, status.person_bbox.width + vw * steps);
        const float pred_h = std::max(0.0f, status.person_bbox.height + vh * steps);
        return cv::Rect2f(status.person_bbox.x + vx * steps, status.person_bbox.y + vy * steps,
                          pred_w, pred_h);
    }
};

class WorkerTracker {
   public:
    explicit WorkerTracker(float iou_threshold = 0.30f, int alert_streak_threshold = 3,
                           int max_missed_frames = 5,
                           DropoutStreakPolicy dropout_policy = DropoutStreakPolicy::HoldStreak,
                           float high_conf_thresh = 0.50f);

    std::vector<TrackedWorker> update(const std::vector<WorkerStatus>& current_workers);

    const std::vector<TrackedWorker>& active_tracks() const noexcept { return tracks_; }

    void set_dropout_policy(DropoutStreakPolicy policy) noexcept { dropout_policy_ = policy; }

    DropoutStreakPolicy dropout_policy() const noexcept { return dropout_policy_; }

    void set_high_conf_threshold(float high_conf_thresh);

    float high_conf_threshold() const noexcept { return high_conf_thresh_; }

    void reset() noexcept;

   private:
    AlertState compute_alert_state(int violation_streak) const noexcept;

    float iou_threshold_;
    int alert_streak_threshold_;
    int max_missed_frames_;
    DropoutStreakPolicy dropout_policy_;
    float high_conf_thresh_;
    int next_track_id_{1};
    std::vector<TrackedWorker> tracks_;
};

}  // namespace safety_engine
