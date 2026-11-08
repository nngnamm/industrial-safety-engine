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
constexpr float kVelocitySmoothingAlpha = 0.70f;

struct CandidateMatch {
    std::size_t track_idx;
    std::size_t worker_idx;
    float iou;
};

void sort_and_assign_stage(const std::vector<TrackedWorker>& tracks,
                           const std::vector<WorkerStatus>& current_workers,
                           const std::vector<std::size_t>& worker_indices, float iou_threshold,
                           std::vector<bool>& track_matched, std::vector<bool>& worker_matched,
                           std::vector<std::size_t>& matched_worker_for_track) {
    const std::size_t num_tracks = tracks.size();
    if (num_tracks == 0U || worker_indices.empty()) {
        return;
    }

    std::vector<CandidateMatch> candidates;
    candidates.reserve(num_tracks * worker_indices.size());

    for (std::size_t t = 0; t < num_tracks; ++t) {
        if (track_matched[t]) {
            continue;
        }
        const cv::Rect2f pred_bbox = tracks[t].predict_next_bbox();
        for (const std::size_t w : worker_indices) {
            if (worker_matched[w]) {
                continue;
            }
            const float iou = compute_iou(pred_bbox, current_workers[w].person_bbox);
            if (iou >= iou_threshold) {
                candidates.push_back(CandidateMatch{t, w, iou});
            }
        }
    }

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

    for (const CandidateMatch& match : candidates) {
        if (track_matched[match.track_idx] || worker_matched[match.worker_idx]) {
            continue;
        }
        track_matched[match.track_idx] = true;
        worker_matched[match.worker_idx] = true;
        matched_worker_for_track[match.track_idx] = match.worker_idx;
    }
}

}  // namespace

WorkerTracker::WorkerTracker(float iou_threshold, int alert_streak_threshold, int max_missed_frames,
                             DropoutStreakPolicy dropout_policy, float high_conf_thresh)
    : iou_threshold_(iou_threshold),
      alert_streak_threshold_(alert_streak_threshold),
      max_missed_frames_(max_missed_frames),
      dropout_policy_(dropout_policy),
      high_conf_thresh_(high_conf_thresh) {
    if (iou_threshold_ <= 0.0f || iou_threshold_ > 1.0f) {
        throw std::invalid_argument("WorkerTracker: iou_threshold must be within (0.0, 1.0].");
    }
    if (alert_streak_threshold_ < 1) {
        throw std::invalid_argument("WorkerTracker: alert_streak_threshold must be >= 1.");
    }
    if (max_missed_frames_ < 0) {
        throw std::invalid_argument("WorkerTracker: max_missed_frames must be >= 0.");
    }
    if (high_conf_thresh_ < 0.0f || high_conf_thresh_ > 1.0f) {
        throw std::invalid_argument("WorkerTracker: high_conf_thresh must be within [0.0, 1.0].");
    }
}

void WorkerTracker::set_high_conf_threshold(float high_conf_thresh) {
    if (high_conf_thresh < 0.0f || high_conf_thresh > 1.0f) {
        throw std::invalid_argument("WorkerTracker: high_conf_thresh must be within [0.0, 1.0].");
    }
    high_conf_thresh_ = high_conf_thresh;
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

    std::vector<std::size_t> high_conf_indices;
    std::vector<std::size_t> low_conf_indices;
    high_conf_indices.reserve(num_workers);
    low_conf_indices.reserve(num_workers);

    for (std::size_t w = 0; w < num_workers; ++w) {
        if (current_workers[w].person_confidence >= high_conf_thresh_) {
            high_conf_indices.push_back(w);
        } else {
            low_conf_indices.push_back(w);
        }
    }

    std::vector<bool> track_matched(num_tracks, false);
    std::vector<bool> worker_matched(num_workers, false);
    std::vector<std::size_t> matched_worker_for_track(num_tracks, 0U);

    // ByteTrack Stage 1: Match active tracks against high-confidence detections.
    sort_and_assign_stage(tracks_, current_workers, high_conf_indices, iou_threshold_,
                          track_matched, worker_matched, matched_worker_for_track);

    // ByteTrack Stage 2: Match remaining unmatched active tracks against low-confidence detections.
    sort_and_assign_stage(tracks_, current_workers, low_conf_indices, iou_threshold_, track_matched,
                          worker_matched, matched_worker_for_track);

    // Update matched tracks with constant-velocity alpha-beta smoothing and violation counters.
    for (std::size_t t = 0; t < num_tracks; ++t) {
        if (!track_matched[t]) {
            continue;
        }

        TrackedWorker& track = tracks_[t];
        const WorkerStatus& worker = current_workers[matched_worker_for_track[t]];

        const float dt = static_cast<float>(track.missed_frames + 1);
        const float inst_vx = (worker.person_bbox.x - track.status.person_bbox.x) / dt;
        const float inst_vy = (worker.person_bbox.y - track.status.person_bbox.y) / dt;
        const float inst_vw = (worker.person_bbox.width - track.status.person_bbox.width) / dt;
        const float inst_vh = (worker.person_bbox.height - track.status.person_bbox.height) / dt;

        if (track.age_frames == 1) {
            track.vx = inst_vx;
            track.vy = inst_vy;
            track.vw = inst_vw;
            track.vh = inst_vh;
        } else {
            track.vx =
                kVelocitySmoothingAlpha * inst_vx + (1.0f - kVelocitySmoothingAlpha) * track.vx;
            track.vy =
                kVelocitySmoothingAlpha * inst_vy + (1.0f - kVelocitySmoothingAlpha) * track.vy;
            track.vw =
                kVelocitySmoothingAlpha * inst_vw + (1.0f - kVelocitySmoothingAlpha) * track.vw;
            track.vh =
                kVelocitySmoothingAlpha * inst_vh + (1.0f - kVelocitySmoothingAlpha) * track.vh;
        }

        track.status = worker;
        track.missed_frames = 0;
        ++track.age_frames;

        if (worker.is_violating) {
            ++track.violation_streak;
            ++track.consecutive_observed_violations;
            ++track.total_violation_frames;
        } else {
            track.violation_streak = 0;
            track.consecutive_observed_violations = 0;
        }
        track.alert_state = compute_alert_state(track.violation_streak);
    }

    // Update unmatched existing tracks according to DropoutStreakPolicy and prune expired tracks.
    std::vector<TrackedWorker> surviving_tracks;
    surviving_tracks.reserve(num_tracks + high_conf_indices.size());

    for (std::size_t t = 0; t < num_tracks; ++t) {
        if (track_matched[t]) {
            surviving_tracks.push_back(std::move(tracks_[t]));
        } else {
            TrackedWorker& track = tracks_[t];
            ++track.missed_frames;
            track.consecutive_observed_violations = 0;

            switch (dropout_policy_) {
                case DropoutStreakPolicy::HoldStreak:
                    break;
                case DropoutStreakPolicy::StrictReset:
                    track.violation_streak = 0;
                    track.alert_state = AlertState::Safe;
                    break;
                case DropoutStreakPolicy::DecayStreak:
                    track.violation_streak = std::max(0, track.violation_streak - 1);
                    track.alert_state = compute_alert_state(track.violation_streak);
                    break;
            }

            if (track.missed_frames <= max_missed_frames_) {
                surviving_tracks.push_back(std::move(track));
            }
        }
    }

    // Spawn new tracks ONLY for unmatched high-confidence detections.
    for (const std::size_t w : high_conf_indices) {
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
        new_track.consecutive_observed_violations = worker.is_violating ? 1 : 0;
        new_track.total_violation_frames = worker.is_violating ? 1 : 0;
        new_track.alert_state = compute_alert_state(new_track.violation_streak);

        surviving_tracks.push_back(std::move(new_track));
    }

    // Ensure deterministic ordering by track_id and return active non-expired tracks.
    std::sort(surviving_tracks.begin(), surviving_tracks.end(),
              [](const TrackedWorker& a, const TrackedWorker& b) { return a.track_id < b.track_id; });

    tracks_ = std::move(surviving_tracks);
    return tracks_;
}

}  // namespace safety_engine
