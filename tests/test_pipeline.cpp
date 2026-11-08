#include <gtest/gtest.h>

#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
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

TEST(PipelineTest, EndToEndVideoFileInference) {
    const std::filesystem::path model_path = resolve_dummy_model_path();
    if (!std::filesystem::exists(model_path)) {
        GTEST_SKIP() << "Skipping video file inference test: " << model_path << " not found.";
    }

    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    const std::filesystem::path in_video_path = temp_dir / "safety_engine_test_input.avi";
    const std::filesystem::path out_video_path = temp_dir / "safety_engine_test_output.avi";

    constexpr int kWidth = 640;
    constexpr int kHeight = 480;
    constexpr int kNumFrames = 8;
    constexpr double kFps = 25.0;

    // 1. Generate temporary synthetic input video via cv::VideoWriter
    {
        cv::VideoWriter input_writer(in_video_path.string(),
                                     cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), kFps,
                                     cv::Size(kWidth, kHeight));
        if (!input_writer.isOpened()) {
            input_writer.open(in_video_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                              kFps, cv::Size(kWidth, kHeight));
        }
        ASSERT_TRUE(input_writer.isOpened());

        for (int i = 0; i < kNumFrames; ++i) {
            cv::Mat frame(kHeight, kWidth, CV_8UC3,
                          cv::Scalar(40 + i * 5, 70 + i * 3, 100 + i * 4));
            input_writer.write(frame);
        }
        input_writer.release();
    }

    ASSERT_TRUE(std::filesystem::exists(in_video_path));
    EXPECT_GT(std::filesystem::file_size(in_video_path), 0U);

    // 2. Run SafetyPipeline::run_video_stream end-to-end producing an annotated output video
    safety_engine::Detector detector(model_path.string(), 0.45f, 0.50f, 640, 640);
    const safety_engine::SafetyAnalyzer analyzer;
    safety_engine::WorkerTracker tracker(0.30f, 3, 5);
    const std::vector<safety_engine::HazardZone> zones = {
        safety_engine::HazardZone{
            "VideoTestZone",
            {{80.0f, 60.0f}, {560.0f, 60.0f}, {560.0f, 440.0f}, {80.0f, 440.0f}},
            true,
            true,
        },
    };
    safety_engine::SafetyPipeline pipeline(detector, analyzer, tracker, zones);

    cv::VideoCapture cap(in_video_path.string());
    ASSERT_TRUE(cap.isOpened());

    cv::VideoWriter output_writer(out_video_path.string(),
                                  cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), kFps,
                                  cv::Size(kWidth, kHeight));
    if (!output_writer.isOpened()) {
        output_writer.open(out_video_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           kFps, cv::Size(kWidth, kHeight));
    }
    ASSERT_TRUE(output_writer.isOpened());

    const safety_engine::BenchmarkReport report =
        pipeline.run_video_stream(cap, &output_writer, /*max_frames=*/-1, /*queue_capacity=*/4);
    output_writer.release();
    cap.release();

    // 3. Verify processed frame count, telemetry stats, and output video validity
    EXPECT_EQ(report.total_frames, static_cast<std::size_t>(kNumFrames));
    EXPECT_GT(report.throughput_fps, 0.0);
    EXPECT_GT(report.detect_stats.mean_ms, 0.0);
    EXPECT_GT(report.total_frame_stats.mean_ms, 0.0);

    ASSERT_TRUE(std::filesystem::exists(out_video_path));
    EXPECT_GT(std::filesystem::file_size(out_video_path), 0U);

    cv::VideoCapture verify_cap(out_video_path.string());
    ASSERT_TRUE(verify_cap.isOpened());
    int decoded_out_frames = 0;
    cv::Mat out_frame;
    while (verify_cap.read(out_frame) && !out_frame.empty()) {
        EXPECT_EQ(out_frame.cols, kWidth);
        EXPECT_EQ(out_frame.rows, kHeight);
        ++decoded_out_frames;
    }
    verify_cap.release();
    EXPECT_EQ(decoded_out_frames, kNumFrames);

    std::error_code ec;
    std::filesystem::remove(in_video_path, ec);
    std::filesystem::remove(out_video_path, ec);
}
