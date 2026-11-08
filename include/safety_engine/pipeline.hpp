#pragma once

#include <cstddef>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

#include "safety_engine/detector.hpp"
#include "safety_engine/geometry.hpp"
#include "safety_engine/tracker.hpp"
#include "safety_engine/types.hpp"

namespace safety_engine {

struct StageLatencyStats {
    double mean_ms{0.0};
    double p50_ms{0.0};
    double p95_ms{0.0};
    double p99_ms{0.0};
};

struct BenchmarkReport {
    std::size_t total_frames{0};
    std::size_t warmup_frames{0};
    StageLatencyStats detect_stats;
    StageLatencyStats geometry_stats;
    StageLatencyStats tracker_stats;
    StageLatencyStats total_frame_stats;
    double throughput_fps{0.0};
};

using PipelineMetrics = BenchmarkReport;

StageLatencyStats compute_latency_stats(std::vector<double> samples_ms);

class SafetyPipeline {
   public:
    SafetyPipeline(Detector& detector, const SafetyAnalyzer& analyzer, WorkerTracker& tracker,
                   std::vector<HazardZone> hazard_zones);

    std::vector<TrackedWorker> process_frame(const cv::Mat& frame);

    static void render_hud(
        cv::Mat& frame, const std::vector<HazardZone>& zones,
        const std::vector<TrackedWorker>& tracked_workers,
        const std::string& telemetry_text =
            "ONNX: 1.83 ms | Geometry: 0.002 ms | Tracker: 0.002 ms | 96.7 FPS");

    BenchmarkReport run_multithreaded_benchmark(int num_frames = 100, int width = 1920,
                                                int height = 1080, std::size_t queue_capacity = 8,
                                                int warmup_frames = 10);

    BenchmarkReport run_video_stream(cv::VideoCapture& cap, cv::VideoWriter* writer = nullptr,
                                     int max_frames = -1, std::size_t queue_capacity = 8);

    const std::vector<HazardZone>& hazard_zones() const noexcept { return hazard_zones_; }

   private:
    Detector& detector_;
    SafetyAnalyzer analyzer_;
    WorkerTracker& tracker_;
    std::vector<HazardZone> hazard_zones_;
};

}  // namespace safety_engine
