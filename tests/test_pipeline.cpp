#include <gtest/gtest.h>

#include <filesystem>
#include <opencv2/core.hpp>
#include <vector>

#include "safety_engine/detector.hpp"
#include "safety_engine/geometry.hpp"
#include "safety_engine/pipeline.hpp"
#include "safety_engine/tracker.hpp"
#include "safety_engine/types.hpp"

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define SAFETY_ENGINE_HAS_LSAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) && !defined(SAFETY_ENGINE_HAS_LSAN)
#define SAFETY_ENGINE_HAS_LSAN 1
#endif

#if defined(SAFETY_ENGINE_HAS_LSAN)
extern "C" const char* __lsan_default_suppressions() {
    // Pre-built libonnxruntime.so allocates two one-time process-global cpuinfo buffers via calloc.
    return "leak:libonnxruntime.so\n";
}
#endif

namespace {

std::filesystem::path resolve_dummy_model_path() {
    const std::filesystem::path cwd_candidate("models/dummy_yolo.onnx");
    if (std::filesystem::exists(cwd_candidate)) {
        return cwd_candidate;
    }
#ifdef SAFETY_ENGINE_PROJECT_ROOT
    const std::filesystem::path root_candidate =
        std::filesystem::path(SAFETY_ENGINE_PROJECT_ROOT) / "models" / "dummy_yolo.onnx";
    if (std::filesystem::exists(root_candidate)) {
        return root_candidate;
    }
#endif
    return cwd_candidate;
}

}  // namespace

TEST(PipelineTest, EndToEndSingleFrameAndHudRender) {
    const std::filesystem::path model_path = resolve_dummy_model_path();
    if (!std::filesystem::exists(model_path)) {
        GTEST_SKIP() << "Skipping pipeline test: " << model_path << " not found.";
    }

    safety_engine::Detector detector(model_path.string(), 0.45f, 0.50f, 640, 640);
    const safety_engine::SafetyAnalyzer analyzer;
    safety_engine::WorkerTracker tracker(0.30f, /*alert_streak_threshold=*/3,
                                         /*max_missed_frames=*/5);

    // Dummy ONNX model emits a centered Person bbox at [810, 240, 300, 600] on 1920x1080
    // with feet_point = (960, 840) and no PPE detections.
    const std::vector<safety_engine::HazardZone> zones = {
        safety_engine::HazardZone{
            "CentralAssemblyZone",
            {{500.0f, 180.0f}, {1420.0f, 180.0f}, {1420.0f, 960.0f}, {500.0f, 960.0f}},
            /*require_helmet=*/true,
            /*require_vest=*/true,
        },
    };

    safety_engine::SafetyPipeline pipeline(detector, analyzer, tracker, zones);

    const cv::Mat original_frame(1080, 1920, CV_8UC3, cv::Scalar(48, 48, 48));
    cv::Mat hud_frame = original_frame.clone();

    const std::vector<safety_engine::TrackedWorker> tracked = pipeline.process_frame(hud_frame);
    ASSERT_FALSE(tracked.empty());
    ASSERT_EQ(tracked.size(), 1U);
    EXPECT_EQ(tracked[0].track_id, 1);
    EXPECT_TRUE(tracked[0].status.in_hazard_zone);
    EXPECT_TRUE(tracked[0].status.is_violating);
    EXPECT_EQ(tracked[0].violation_streak, 1);
    EXPECT_EQ(tracked[0].alert_state, safety_engine::AlertState::SuspectedViolation);

    safety_engine::SafetyPipeline::render_hud(hud_frame, pipeline.hazard_zones(), tracked);

    // Verify HUD rendering modified frame pixels (semi-transparent hazard polygon + worker box/tag)
    const double pixel_diff_l1 = cv::norm(hud_frame, original_frame, cv::NORM_L1);
    EXPECT_GT(pixel_diff_l1, 0.0);
}

TEST(PipelineTest, MultiThreadedBenchmarkZeroDrops) {
    const std::filesystem::path model_path = resolve_dummy_model_path();
    if (!std::filesystem::exists(model_path)) {
        GTEST_SKIP() << "Skipping benchmark test: " << model_path << " not found.";
    }

    safety_engine::Detector detector(model_path.string(), 0.45f, 0.50f, 640, 640);
    const safety_engine::SafetyAnalyzer analyzer;
    safety_engine::WorkerTracker tracker(0.30f, 3, 5);
    const std::vector<safety_engine::HazardZone> zones = {
        safety_engine::HazardZone{
            "BenchZone",
            {{100.0f, 100.0f}, {540.0f, 100.0f}, {540.0f, 540.0f}, {100.0f, 540.0f}},
            true,
            true,
        },
    };

    safety_engine::SafetyPipeline pipeline(detector, analyzer, tracker, zones);

    const safety_engine::BenchmarkReport report =
        pipeline.run_multithreaded_benchmark(/*num_frames=*/20, /*width=*/640, /*height=*/640,
                                             /*queue_capacity=*/4);

    EXPECT_EQ(report.total_frames, 20U);
    EXPECT_GT(report.throughput_fps, 0.0);

    EXPECT_GT(report.detect_stats.mean_ms, 0.0);
    EXPECT_LE(report.detect_stats.p50_ms, report.detect_stats.p95_ms);
    EXPECT_LE(report.detect_stats.p95_ms, report.detect_stats.p99_ms);

    EXPECT_GT(report.total_frame_stats.mean_ms, 0.0);
    EXPECT_LE(report.total_frame_stats.p50_ms, report.total_frame_stats.p95_ms);
    EXPECT_LE(report.total_frame_stats.p95_ms, report.total_frame_stats.p99_ms);
}
