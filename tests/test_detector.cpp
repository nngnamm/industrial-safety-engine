#include <gtest/gtest.h>

#include <filesystem>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "safety_engine/detector.hpp"
#include "safety_engine/types.hpp"

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

TEST(DetectorTest, ComputeLetterboxInfo_1920x1080_to_640x640) {
    const safety_engine::LetterboxInfo info =
        safety_engine::Detector::compute_letterbox_info(1920, 1080, 640, 640);

    EXPECT_FLOAT_EQ(info.scale, 640.0f / 1920.0f);
    EXPECT_EQ(info.pad_w, 0);
    EXPECT_EQ(info.pad_h, 140);
}

TEST(DetectorTest, UnletterboxBbox_RoundTrip) {
    const safety_engine::LetterboxInfo info =
        safety_engine::Detector::compute_letterbox_info(1920, 1080, 640, 640);

    // Centered box in 640x640 letterbox space: cx=320, cy=320, w=100, h=200
    // Expected in 1920x1080: x=(320-50)/scale=810, y=(320-100-140)/scale=240, w=300, h=600
    const cv::Rect2f rect =
        safety_engine::Detector::unletterbox_bbox(320.0f, 320.0f, 100.0f, 200.0f, info, 1920, 1080);

    EXPECT_NEAR(rect.x, 810.0f, 1e-3f);
    EXPECT_NEAR(rect.y, 240.0f, 1e-3f);
    EXPECT_NEAR(rect.width, 300.0f, 1e-3f);
    EXPECT_NEAR(rect.height, 600.0f, 1e-3f);

    // Verify boundary clamping to [0, 1920] x [0, 1080]
    const cv::Rect2f clamped =
        safety_engine::Detector::unletterbox_bbox(10.0f, 100.0f, 80.0f, 80.0f, info, 1920, 1080);
    EXPECT_GE(clamped.x, 0.0f);
    EXPECT_GE(clamped.y, 0.0f);
    EXPECT_LE(clamped.x + clamped.width, 1920.0f);
    EXPECT_LE(clamped.y + clamped.height, 1080.0f);
}

TEST(DetectorTest, DetectorEndToEnd_DummyModel) {
    const std::filesystem::path model_path = resolve_dummy_model_path();
    if (!std::filesystem::exists(model_path)) {
        GTEST_SKIP() << "Skipping end-to-end inference test: " << model_path << " not found.";
    }

    safety_engine::Detector detector(model_path.string(), 0.45f, 0.50f, 640, 640);

    const cv::Mat dummy_frame(1080, 1920, CV_8UC3, cv::Scalar(64, 128, 192));
    const std::vector<safety_engine::Detection> detections = detector.detect(dummy_frame);

    ASSERT_EQ(detections.size(), 1U);
    EXPECT_EQ(detections[0].class_id, static_cast<int>(safety_engine::ClassId::Person));
    EXPECT_NEAR(detections[0].confidence, 0.95f, 1e-5f);
    EXPECT_NEAR(detections[0].bbox.x, 810.0f, 1e-2f);
    EXPECT_NEAR(detections[0].bbox.y, 240.0f, 1e-2f);
    EXPECT_NEAR(detections[0].bbox.width, 300.0f, 1e-2f);
    EXPECT_NEAR(detections[0].bbox.height, 600.0f, 1e-2f);

    // Verify empty frame validation
    const cv::Mat empty_frame;
    EXPECT_THROW(detector.detect(empty_frame), std::invalid_argument);
}
