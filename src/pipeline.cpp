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
            return cv::Scalar(40, 210, 40);  // Green (BGR)
        case AlertState::SuspectedViolation:
            return cv::Scalar(0, 190, 255);  // Amber / Orange-Yellow (BGR)
        case AlertState::ActiveAlert:
            return cv::Scalar(25, 25, 240);  // Red (BGR)
    }
    return cv::Scalar(200, 200, 200);
}

std::string format_worker_label(const TrackedWorker& worker) {
    const std::string prefix = "Worker #" + std::to_string(worker.track_id) + ": ";
    switch (worker.alert_state) {
        case AlertState::Safe:
            return prefix + "SAFE";
        case AlertState::SuspectedViolation:
            return prefix + "SUSPECTED (Streak: " + std::to_string(worker.violation_streak) +
                   "/3)";
        case AlertState::ActiveAlert: {
            std::string missing_tag = "[ZONE VIOLATION]";
            if (!worker.status.has_helmet && !worker.status.has_vest) {
                missing_tag = "[NO PPE]";
            } else if (!worker.status.has_helmet) {
                missing_tag = "[NO HELMET]";
            } else if (!worker.status.has_vest) {
                missing_tag = "[NO VEST]";
            }
            return prefix + "ACTIVE ALERT " + missing_tag;
        }
    }
    return prefix + "UNKNOWN";
}

bool zone_has_state(const HazardZone& zone, const std::vector<TrackedWorker>& tracked_workers,
                    AlertState target_state) {
    for (const TrackedWorker& worker : tracked_workers) {
        if (worker.missed_frames > 0 || worker.alert_state != target_state) {
            continue;
        }
        for (const std::string& vz : worker.status.violated_zones) {
            if (vz == zone.name) {
                return true;
            }
        }
    }
    return false;
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
                                const std::vector<TrackedWorker>& tracked_workers,
                                const std::string& telemetry_text) {
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

    for (std::size_t i = 0; i < zone_polygons.size(); ++i) {
        const bool is_active_alert =
            zone_has_state(*valid_zones[i], tracked_workers, AlertState::ActiveAlert);
        const bool is_suspected =
            !is_active_alert &&
            zone_has_state(*valid_zones[i], tracked_workers, AlertState::SuspectedViolation);

        cv::Scalar fill_color(0, 95, 175);
        cv::Scalar border_color(0, 185, 245);
        double alpha = 0.16;
        int border_thickness = 2;
        std::string status_suffix;

        if (is_active_alert) {
            fill_color = cv::Scalar(0, 0, 230);
            border_color = cv::Scalar(20, 20, 255);
            alpha = 0.32;
            border_thickness = 3;
            status_suffix = " [ACTIVE ALERT]";
        } else if (is_suspected) {
            fill_color = cv::Scalar(0, 140, 255);
            border_color = cv::Scalar(0, 200, 255);
            alpha = 0.22;
            border_thickness = 2;
            status_suffix = " [SUSPECTED]";
        }

        cv::Mat overlay = frame.clone();
        const std::vector<std::vector<cv::Point>> single_poly = {zone_polygons[i]};
        cv::fillPoly(overlay, single_poly, fill_color);
        cv::addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame);

        cv::polylines(frame, zone_polygons[i], /*isClosed=*/true, border_color, border_thickness,
                      cv::LINE_AA);
        if (!valid_zones[i]->name.empty()) {
            const cv::Point anchor = zone_polygons[i].front();
            const std::string zone_title = "ZONE: " + valid_zones[i]->name + status_suffix;
            const cv::Scalar title_color =
                is_active_alert ? cv::Scalar(70, 70, 255) : cv::Scalar(0, 215, 255);
            cv::putText(frame, zone_title, cv::Point(anchor.x + 8, std::max(24, anchor.y + 26)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(12, 12, 12), 4, cv::LINE_AA);
            cv::putText(frame, zone_title, cv::Point(anchor.x + 8, std::max(24, anchor.y + 26)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.62, title_color, 2, cv::LINE_AA);
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

        if (worker.status.helmet_bbox.has_value()) {
            const cv::Rect hbox(static_cast<int>(std::lround(worker.status.helmet_bbox->x)),
                                static_cast<int>(std::lround(worker.status.helmet_bbox->y)),
                                static_cast<int>(std::lround(worker.status.helmet_bbox->width)),
                                static_cast<int>(std::lround(worker.status.helmet_bbox->height)));
            cv::rectangle(frame, hbox, cv::Scalar(190, 245, 60), 1, cv::LINE_AA);
        }
        if (worker.status.vest_bbox.has_value()) {
            const cv::Rect vbox(static_cast<int>(std::lround(worker.status.vest_bbox->x)),
                                static_cast<int>(std::lround(worker.status.vest_bbox->y)),
                                static_cast<int>(std::lround(worker.status.vest_bbox->width)),
                                static_cast<int>(std::lround(worker.status.vest_bbox->height)));
            cv::rectangle(frame, vbox, cv::Scalar(50, 225, 225), 1, cv::LINE_AA);
        }

        const int box_thickness = (worker.alert_state == AlertState::ActiveAlert) ? 3 : 2;
        cv::rectangle(frame, box, color, box_thickness, cv::LINE_AA);

        const cv::Point feet_pt(static_cast<int>(std::lround(worker.status.feet_point.x)),
                                static_cast<int>(std::lround(worker.status.feet_point.y)));
        cv::circle(frame, feet_pt, 5, color, cv::FILLED, cv::LINE_AA);
        cv::circle(frame, feet_pt, 9, color, 1, cv::LINE_AA);

        const std::string tag = format_worker_label(worker);

        int baseline = 0;
        const cv::Size text_size =
            cv::getTextSize(tag, cv::FONT_HERSHEY_SIMPLEX, 0.54, 2, &baseline);
        const int label_top = std::max(box.y - text_size.height - 10, 0);
        const int label_x = std::clamp(box.x, 4, std::max(4, frame.cols - text_size.width - 16));
        const cv::Rect label_bg(label_x, label_top, text_size.width + 12,
                                text_size.height + baseline + 8);

        cv::rectangle(frame, label_bg, color, cv::FILLED);
        const cv::Scalar text_fg = (worker.alert_state == AlertState::ActiveAlert)
                                       ? cv::Scalar(250, 250, 250)
                                       : cv::Scalar(15, 15, 15);
        cv::putText(frame, tag, cv::Point(label_x + 6, label_top + text_size.height + 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.54, text_fg, 2, cv::LINE_AA);
    }

    // 3. Render top-left live stage telemetry bar
    if (!telemetry_text.empty()) {
        int baseline = 0;
        const cv::Size telem_size =
            cv::getTextSize(telemetry_text, cv::FONT_HERSHEY_SIMPLEX, 0.56, 2, &baseline);
        const cv::Rect telem_rect(14, 12, std::min(frame.cols - 28, telem_size.width + 24),
                                  telem_size.height + baseline + 16);
        if (telem_rect.width > 0 && telem_rect.height > 0 &&
            telem_rect.x + telem_rect.width <= frame.cols &&
            telem_rect.y + telem_rect.height <= frame.rows) {
            cv::Mat roi = frame(telem_rect);
            cv::Mat dark_bg(roi.size(), roi.type(), cv::Scalar(18, 22, 28));
            cv::addWeighted(dark_bg, 0.82, roi, 0.18, 0.0, roi);
            cv::rectangle(frame, telem_rect, cv::Scalar(90, 180, 220), 1, cv::LINE_AA);
            cv::putText(frame, telemetry_text,
                        cv::Point(telem_rect.x + 12, telem_rect.y + telem_size.height + 7),
                        cv::FONT_HERSHEY_SIMPLEX, 0.56, cv::Scalar(230, 245, 255), 2, cv::LINE_AA);
        }
    }
}

BenchmarkReport SafetyPipeline::run_multithreaded_benchmark(int num_frames, int width, int height,
                                                            std::size_t queue_capacity,
                                                            int warmup_frames) {
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
    if (warmup_frames < 0) {
        throw std::invalid_argument(
            "SafetyPipeline::run_multithreaded_benchmark: warmup_frames must be >= 0.");
    }

    tracker_.reset();

    // Execute warm-up frames through the full pipeline (Detector + Geometry + Tracker)
    // prior to starting latency timers so JIT/arena initialization does not skew P50/P95/P99.
    if (warmup_frames > 0) {
        cv::Mat warmup_frame(height, width, CV_8UC3, cv::Scalar(64.0, 96.0, 128.0));
        for (int w = 0; w < warmup_frames; ++w) {
            const std::vector<Detection> detections = detector_.detect(warmup_frame);
            const std::vector<WorkerStatus> statuses =
                analyzer_.analyze_frame(detections, hazard_zones_);
            const std::vector<TrackedWorker> tracked = tracker_.update(statuses);
            (void)tracked;
        }
        tracker_.reset();
    }

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
    report.warmup_frames = static_cast<std::size_t>(warmup_frames);
    report.detect_stats = compute_latency_stats(std::move(detect_ms));
    report.geometry_stats = compute_latency_stats(std::move(geometry_ms));
    report.tracker_stats = compute_latency_stats(std::move(tracker_ms));
    report.total_frame_stats = compute_latency_stats(std::move(total_ms));
    report.throughput_fps =
        (wall_seconds > 0.0) ? (static_cast<double>(report.total_frames) / wall_seconds) : 0.0;

    return report;
}

}  // namespace safety_engine
