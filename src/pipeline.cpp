#include "safety_engine/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "safety_engine/frame_queue.hpp"

namespace safety_engine {

namespace {

double interpolate_percentile(const std::vector<double>& sorted_samples, double quantile) noexcept {
    if (sorted_samples.empty()) {
        return 0.0;
    }
    if (sorted_samples.size() == 1U) {
        return sorted_samples.front();
    }

    const double clamped_q = std::clamp(quantile, 0.0, 1.0);
    const double pos = clamped_q * static_cast<double>(sorted_samples.size() - 1U);
    const auto lo_idx = static_cast<std::size_t>(std::floor(pos));
    const auto hi_idx = static_cast<std::size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo_idx);

    return sorted_samples[lo_idx] + frac * (sorted_samples[hi_idx] - sorted_samples[lo_idx]);
}

cv::Scalar alert_state_color(AlertState state) noexcept {
    switch (state) {
        case AlertState::Safe:
            return cv::Scalar(0, 200, 0);  // Green (BGR)
        case AlertState::SuspectedViolation:
            return cv::Scalar(0, 180, 255);  // Amber / Orange (BGR)
        case AlertState::ActiveAlert:
            return cv::Scalar(0, 0, 255);  // Red (BGR)
    }
    return cv::Scalar(200, 200, 200);
}

const char* alert_state_label(AlertState state) noexcept {
    switch (state) {
        case AlertState::Safe:
            return "SAFE";
        case AlertState::SuspectedViolation:
            return "SUSPECTED";
        case AlertState::ActiveAlert:
            return "ALERT";
    }
    return "UNKNOWN";
}

}  // namespace

StageLatencyStats compute_latency_stats(std::vector<double> samples_ms) {
    StageLatencyStats stats;
    if (samples_ms.empty()) {
        return stats;
    }

    std::sort(samples_ms.begin(), samples_ms.end());
    const double sum = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0);

    stats.mean_ms = sum / static_cast<double>(samples_ms.size());
    stats.p50_ms = interpolate_percentile(samples_ms, 0.50);
    stats.p95_ms = interpolate_percentile(samples_ms, 0.95);
    stats.p99_ms = interpolate_percentile(samples_ms, 0.99);
    return stats;
}

SafetyPipeline::SafetyPipeline(Detector& detector, const SafetyAnalyzer& analyzer,
                               WorkerTracker& tracker, std::vector<HazardZone> hazard_zones)
    : detector_(detector),
      analyzer_(analyzer),
      tracker_(tracker),
      hazard_zones_(std::move(hazard_zones)) {}

std::vector<TrackedWorker> SafetyPipeline::process_frame(const cv::Mat& frame) {
    const std::vector<Detection> detections = detector_.detect(frame);
    const std::vector<WorkerStatus> statuses = analyzer_.analyze_frame(detections, hazard_zones_);
    return tracker_.update(statuses);
}

void SafetyPipeline::render_hud(cv::Mat& frame, const std::vector<HazardZone>& zones,
                                const std::vector<TrackedWorker>& tracked_workers) {
    if (frame.empty()) {
        return;
    }
    if (frame.type() != CV_8UC3) {
        throw std::invalid_argument("SafetyPipeline::render_hud: expected 3-channel CV_8UC3 frame.");
    }

    // 1. Render semi-transparent hazard zone overlays and borders
    std::vector<std::vector<cv::Point>> zone_polygons;
    std::vector<const HazardZone*> valid_zones;
    zone_polygons.reserve(zones.size());
    valid_zones.reserve(zones.size());

    for (const HazardZone& zone : zones) {
        if (zone.polygon.size() < 3U) {
            continue;
        }
        std::vector<cv::Point> pts;
        pts.reserve(zone.polygon.size());
        for (const cv::Point2f& pt : zone.polygon) {
            pts.emplace_back(static_cast<int>(std::lround(pt.x)),
                             static_cast<int>(std::lround(pt.y)));
        }
        zone_polygons.push_back(std::move(pts));
        valid_zones.push_back(&zone);
    }

    if (!zone_polygons.empty()) {
        cv::Mat overlay = frame.clone();
        cv::fillPoly(overlay, zone_polygons, cv::Scalar(0, 50, 210));
        cv::addWeighted(overlay, 0.25, frame, 0.75, 0.0, frame);

        for (std::size_t i = 0; i < zone_polygons.size(); ++i) {
            cv::polylines(frame, zone_polygons[i], /*isClosed=*/true, cv::Scalar(0, 140, 255), 2,
                          cv::LINE_AA);
            if (!valid_zones[i]->name.empty()) {
                const cv::Point anchor = zone_polygons[i].front();
                cv::putText(frame, "ZONE: " + valid_zones[i]->name,
                            cv::Point(anchor.x + 6, std::max(20, anchor.y + 22)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 200, 255), 2, cv::LINE_AA);
            }
        }
    }

    // 2. Render tracked workers color-coded by AlertState
    for (const TrackedWorker& worker : tracked_workers) {
        if (worker.missed_frames > 0) {
            continue;
        }

        const cv::Scalar color = alert_state_color(worker.alert_state);
        const cv::Rect box(static_cast<int>(std::lround(worker.status.person_bbox.x)),
                           static_cast<int>(std::lround(worker.status.person_bbox.y)),
                           static_cast<int>(std::lround(worker.status.person_bbox.width)),
                           static_cast<int>(std::lround(worker.status.person_bbox.height)));

        cv::rectangle(frame, box, color, 2, cv::LINE_AA);

        const cv::Point feet_pt(static_cast<int>(std::lround(worker.status.feet_point.x)),
                                static_cast<int>(std::lround(worker.status.feet_point.y)));
        cv::circle(frame, feet_pt, 4, color, cv::FILLED, cv::LINE_AA);

        const std::string tag = "ID:" + std::to_string(worker.track_id) + " " +
                                alert_state_label(worker.alert_state) +
                                " [streak:" + std::to_string(worker.violation_streak) + "]";

        int baseline = 0;
        const cv::Size text_size =
            cv::getTextSize(tag, cv::FONT_HERSHEY_SIMPLEX, 0.55, 2, &baseline);
        const int label_top = std::max(box.y - text_size.height - 8, 0);
        const cv::Rect label_bg(box.x, label_top, text_size.width + 10,
                                text_size.height + baseline + 6);

        cv::rectangle(frame, label_bg, color, cv::FILLED);
        cv::putText(frame, tag, cv::Point(box.x + 5, label_top + text_size.height + 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(15, 15, 15), 2, cv::LINE_AA);
    }
}

BenchmarkReport SafetyPipeline::run_multithreaded_benchmark(int num_frames, int width, int height,
                                                            std::size_t queue_capacity) {
    if (num_frames <= 0) {
        throw std::invalid_argument(
            "SafetyPipeline::run_multithreaded_benchmark: num_frames must be > 0.");
    }
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument(
            "SafetyPipeline::run_multithreaded_benchmark: frame dimensions must be > 0.");
    }
    if (queue_capacity == 0U) {
        throw std::invalid_argument(
            "SafetyPipeline::run_multithreaded_benchmark: queue_capacity must be > 0.");
    }

    tracker_.reset();

    BoundedQueue<cv::Mat> frame_queue(queue_capacity);

    const auto expected_frames = static_cast<std::size_t>(num_frames);
    std::vector<double> detect_ms;
    std::vector<double> geometry_ms;
    std::vector<double> tracker_ms;
    std::vector<double> total_ms;

    detect_ms.reserve(expected_frames);
    geometry_ms.reserve(expected_frames);
    tracker_ms.reserve(expected_frames);
    total_ms.reserve(expected_frames);

    using Clock = std::chrono::steady_clock;
    using Millis = std::chrono::duration<double, std::milli>;

    const Clock::time_point wall_start = Clock::now();

    std::thread producer([&frame_queue, num_frames, width, height]() {
        for (int i = 0; i < num_frames; ++i) {
            // Allocate an independent buffer per frame so producer and consumer never alias memory.
            cv::Mat synthetic_frame(height, width, CV_8UC3,
                                    cv::Scalar(static_cast<double>((i * 7) % 255), 96.0, 128.0));
            if (!frame_queue.push(std::move(synthetic_frame))) {
                break;
            }
        }
        frame_queue.close();
    });

    while (true) {
        std::optional<cv::Mat> frame_opt = frame_queue.pop();
        if (!frame_opt.has_value()) {
            break;
        }

        const Clock::time_point t0 = Clock::now();
        const std::vector<Detection> detections = detector_.detect(*frame_opt);
        const Clock::time_point t1 = Clock::now();
        const std::vector<WorkerStatus> statuses =
            analyzer_.analyze_frame(detections, hazard_zones_);
        const Clock::time_point t2 = Clock::now();
        const std::vector<TrackedWorker> tracked = tracker_.update(statuses);
        (void)tracked;
        const Clock::time_point t3 = Clock::now();

        detect_ms.push_back(Millis(t1 - t0).count());
        geometry_ms.push_back(Millis(t2 - t1).count());
        tracker_ms.push_back(Millis(t3 - t2).count());
        total_ms.push_back(Millis(t3 - t0).count());
    }

    producer.join();

    const Clock::time_point wall_end = Clock::now();
    const double wall_seconds = std::chrono::duration<double>(wall_end - wall_start).count();

    BenchmarkReport report;
    report.total_frames = total_ms.size();
    report.detect_stats = compute_latency_stats(std::move(detect_ms));
    report.geometry_stats = compute_latency_stats(std::move(geometry_ms));
    report.tracker_stats = compute_latency_stats(std::move(tracker_ms));
    report.total_frame_stats = compute_latency_stats(std::move(total_ms));
    report.throughput_fps =
        (wall_seconds > 0.0) ? (static_cast<double>(report.total_frames) / wall_seconds) : 0.0;

    return report;
}

}  // namespace safety_engine
