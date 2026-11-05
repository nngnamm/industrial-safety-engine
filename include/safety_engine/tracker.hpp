#pragma once

#include <vector>

#include "safety_engine/types.hpp"

namespace safety_engine {

enum class AlertState {
    Safe = 0,
    SuspectedViolation = 1,
    ActiveAlert = 2
};

struct TrackedWorker {
    int track_id{0};
    WorkerStatus status;
    int age_frames{0};
    int missed_frames{0};
    int violation_streak{0};
    AlertState alert_state{AlertState::Safe};
};

class WorkerTracker {
   public:
    explicit WorkerTracker(float iou_threshold = 0.30f, int alert_streak_threshold = 3,
                           int max_missed_frames = 5);

    std::vector<TrackedWorker> update(const std::vector<WorkerStatus>& current_workers);

    const std::vector<TrackedWorker>& active_tracks() const noexcept { return tracks_; }

    void reset() noexcept;

   private:
    AlertState compute_alert_state(int violation_streak) const noexcept;

    float iou_threshold_;
    int alert_streak_threshold_;
    int max_missed_frames_;
    int next_track_id_{1};
    std::vector<TrackedWorker> tracks_;
};

}  // namespace safety_engine
