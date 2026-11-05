#include "safety_engine/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "safety_engine/geometry.hpp"

namespace safety_engine {

namespace {

constexpr float kIoUTieEpsilon = 1e-6f;

struct CandidateMatch {
    std::size_t track_idx;
    std::size_t worker_idx;
    float iou;
};

}  // namespace

WorkerTracker::WorkerTracker(float iou_threshold, int alert_streak_threshold, int max_missed_frames)
    : iou_threshold_(iou_threshold),
      alert_streak_threshold_(alert_streak_threshold),
      max_missed_frames_(max_missed_frames) {
    if (iou_threshold_ <= 0.0f || iou_threshold_ > 1.0f) {
        throw std::invalid_argument("WorkerTracker: iou_threshold must be within (0.0, 1.0].");
    }
    if (alert_streak_threshold_ < 1) {
        throw std::invalid_argument("WorkerTracker: alert_streak_threshold must be >= 1.");
    }
    if (max_missed_frames_ < 0) {
        throw std::invalid_argument("WorkerTracker: max_missed_frames must be >= 0.");
    }
}

AlertState WorkerTracker::compute_alert_state(int violation_streak) const noexcept {
    if (violation_streak <= 0) {
        return AlertState::Safe;
    }
    if (violation_streak < alert_streak_threshold_) {
        return AlertState::SuspectedViolation;
    }
    return AlertState::ActiveAlert;
}

void WorkerTracker::reset() noexcept {
    tracks_.clear();
    next_track_id_ = 1;
}

std::vector<TrackedWorker> WorkerTracker::update(const std::vector<WorkerStatus>& current_workers) {
    const std::size_t num_tracks = tracks_.size();
    const std::size_t num_workers = current_workers.size();

    std::vector<CandidateMatch> candidates;
    candidates.reserve(num_tracks * num_workers);

    for (std::size_t t = 0; t < num_tracks; ++t) {
        for (std::size_t w = 0; w < num_workers; ++w) {
            const float iou =
                compute_iou(tracks_[t].status.person_bbox, current_workers[w].person_bbox);
            if (iou >= iou_threshold_) {
                candidates.push_back(CandidateMatch{t, w, iou});
            }
        }
    }

    // Sort greedily in descending order of IoU with deterministic index tie-breaking.
    std::sort(candidates.begin(), candidates.end(),
              [](const CandidateMatch& lhs, const CandidateMatch& rhs) {
                  if (std::abs(lhs.iou - rhs.iou) > kIoUTieEpsilon) {
                      return lhs.iou > rhs.iou;
                  }
                  if (lhs.track_idx != rhs.track_idx) {
                      return lhs.track_idx < rhs.track_idx;
                  }
                  return lhs.worker_idx < rhs.worker_idx;
              });

    std::vector<bool> track_matched(num_tracks, false);
    std::vector<bool> worker_matched(num_workers, false);

    // 1 & 2. Pair up matches greedily (one-to-one) and update matched tracks.
    for (const CandidateMatch& match : candidates) {
        if (track_matched[match.track_idx] || worker_matched[match.worker_idx]) {
            continue;
        }
        track_matched[match.track_idx] = true;
        worker_matched[match.worker_idx] = true;

        TrackedWorker& track = tracks_[match.track_idx];
        const WorkerStatus& worker = current_workers[match.worker_idx];

        track.status = worker;
        track.missed_frames = 0;
        ++track.age_frames;

        if (worker.is_violating) {
            ++track.violation_streak;
        } else {
            track.violation_streak = 0;
        }
        track.alert_state = compute_alert_state(track.violation_streak);
    }

    // 3. Update unmatched existing tracks and prune expired tracks.
    std::vector<TrackedWorker> surviving_tracks;
    surviving_tracks.reserve(num_tracks + num_workers);

    for (std::size_t t = 0; t < num_tracks; ++t) {
        if (track_matched[t]) {
            surviving_tracks.push_back(std::move(tracks_[t]));
        } else {
            ++tracks_[t].missed_frames;
            if (tracks_[t].missed_frames <= max_missed_frames_) {
                surviving_tracks.push_back(std::move(tracks_[t]));
            }
        }
    }

    // 4. Spawn new tracks for unmatched detections.
    for (std::size_t w = 0; w < num_workers; ++w) {
        if (worker_matched[w]) {
            continue;
        }
        const WorkerStatus& worker = current_workers[w];

        TrackedWorker new_track;
        new_track.track_id = next_track_id_++;
        new_track.status = worker;
        new_track.age_frames = 1;
        new_track.missed_frames = 0;
        new_track.violation_streak = worker.is_violating ? 1 : 0;
        new_track.alert_state = compute_alert_state(new_track.violation_streak);

        surviving_tracks.push_back(std::move(new_track));
    }

    // 5. Ensure deterministic ordering by track_id and return active non-expired tracks.
    std::sort(surviving_tracks.begin(), surviving_tracks.end(),
              [](const TrackedWorker& a, const TrackedWorker& b) { return a.track_id < b.track_id; });

    tracks_ = std::move(surviving_tracks);
    return tracks_;
}

}  // namespace safety_engine
