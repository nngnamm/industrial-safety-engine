#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "safety_engine/detector.hpp"
#include "safety_engine/types.hpp"

namespace {

std::filesystem::path resolve_project_path(const std::filesystem::path& relative_path) {
    if (std::filesystem::exists(relative_path)) {
        return relative_path;
    }
#ifdef SAFETY_ENGINE_PROJECT_ROOT
    const std::filesystem::path root_candidate =
        std::filesystem::path(SAFETY_ENGINE_PROJECT_ROOT) / relative_path;
    if (std::filesystem::exists(root_candidate)) {
        return root_candidate;
    }
#endif
    return relative_path;
}

std::filesystem::path resolve_dummy_model_path() {
    return resolve_project_path("models/dummy_yolo.onnx");
}

std::filesystem::path resolve_real_ppe_model_path() {
    return resolve_project_path("models/ppe_yolov8n.onnx");
}

void append_varint(std::string& out, uint64_t val) {
    while (val > 0x7FU) {
        out.push_back(static_cast<char>((val & 0x7FU) | 0x80U));
        val >>= 7U;
    }
    out.push_back(static_cast<char>(val & 0x7FU));
}

std::string field_varint(uint32_t field_num, uint64_t val) {
    std::string out;
    append_varint(out, (static_cast<uint64_t>(field_num) << 3U) | 0U);
    append_varint(out, val);
    return out;
}

std::string field_bytes(uint32_t field_num, const std::string& payload) {
    std::string out;
    append_varint(out, (static_cast<uint64_t>(field_num) << 3U) | 2U);
    append_varint(out, payload.size());
    out.append(payload);
    return out;
}

std::string make_value_info_proto(const std::string& name, uint64_t elem_type,
                                  const std::vector<uint64_t>& dims) {
    std::string dim_msgs;
    for (uint64_t d : dims) {
        dim_msgs.append(field_bytes(1, field_varint(1, d)));
    }
    const std::string tensor_type = field_varint(1, elem_type) + field_bytes(2, dim_msgs);
    const std::string type_proto = field_bytes(1, tensor_type);
    return field_bytes(1, name) + field_bytes(2, type_proto);
}

std::string make_tensor_proto(const std::string& name, uint64_t elem_type,
                              const std::vector<uint64_t>& dims, const std::string& raw_data) {
    std::string msg;
    for (uint64_t d : dims) {
        msg.append(field_varint(1, d));
    }
    msg.append(field_varint(2, elem_type));
    msg.append(field_bytes(8, name));
    msg.append(field_bytes(9, raw_data));
    return msg;
}

void write_minimal_yolo_onnx(const std::filesystem::path& path, uint64_t out_channels,
                             uint64_t out_anchors = 16) {
    constexpr uint64_t kFloatType = 1;  // TensorProto.FLOAT
    const std::string images_vi = make_value_info_proto("images", kFloatType, {1, 3, 640, 640});
    const std::string output_vi =
        make_value_info_proto("output0", kFloatType, {1, out_channels, out_anchors});

    const std::string zero_bytes(sizeof(float), '\0');
    const std::string pred_bytes(static_cast<std::size_t>(out_channels * out_anchors) * sizeof(float),
                                 '\0');

    const std::string zero_init = make_tensor_proto("zero_scalar", kFloatType, {}, zero_bytes);
    const std::string pred_init =
        make_tensor_proto("dummy_pred", kFloatType, {1, out_channels, out_anchors}, pred_bytes);

    const std::string keepdims_attr =
        field_bytes(1, "keepdims") + field_varint(3, 0) + field_varint(20, 2);
    const std::string n_reduce = field_bytes(1, "images") + field_bytes(2, "img_mean") +
                                 field_bytes(3, "reduce_images") + field_bytes(4, "ReduceMean") +
                                 field_bytes(5, keepdims_attr);
    const std::string n_mul = field_bytes(1, "img_mean") + field_bytes(1, "zero_scalar") +
                              field_bytes(2, "zero_dep") + field_bytes(3, "zero_mul") +
                              field_bytes(4, "Mul");
    const std::string n_add = field_bytes(1, "dummy_pred") + field_bytes(1, "zero_dep") +
                              field_bytes(2, "output0") + field_bytes(3, "emit_output") +
                              field_bytes(4, "Add");

    const std::string graph_msg = field_bytes(1, n_reduce) + field_bytes(1, n_mul) +
                                  field_bytes(1, n_add) + field_bytes(2, "InvalidContractGraph") +
                                  field_bytes(5, zero_init) + field_bytes(5, pred_init) +
                                  field_bytes(11, images_vi) + field_bytes(12, output_vi);
    const std::string opset_msg = field_bytes(1, "") + field_varint(2, 13);
    const std::string model_msg = field_varint(1, 7) + field_bytes(2, "test_generator") +
                                  field_bytes(7, graph_msg) + field_bytes(8, opset_msg);

    std::ofstream ofs(path, std::ios::binary);
    ofs.write(model_msg.data(), static_cast<std::streamsize>(model_msg.size()));
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

TEST(DetectorTest, RejectsInvalidModelContract) {
    const std::filesystem::path bad_model_path =
        std::filesystem::temp_directory_path() / "safety_engine_invalid_6ch_yolo.onnx";
    write_minimal_yolo_onnx(bad_model_path, /*out_channels=*/6, /*out_anchors=*/16);

    // Fewer than 7 output channels (4 box + 3 PPE classes) must throw std::runtime_error
    EXPECT_THROW(safety_engine::Detector(bad_model_path.string(), 0.45f, 0.50f, 640, 640),
                 std::runtime_error);

    std::error_code ec;
    std::filesystem::remove(bad_model_path, ec);

    // Mismatched static input resolution must also throw std::runtime_error
    const std::filesystem::path valid_model_path = resolve_dummy_model_path();
    if (std::filesystem::exists(valid_model_path)) {
        EXPECT_THROW(safety_engine::Detector(valid_model_path.string(), 0.45f, 0.50f, 320, 320),
                     std::runtime_error);
    }
}

TEST(DetectorTest, RealTrainedPPEModelContractAndInference) {
    const std::filesystem::path model_path = resolve_real_ppe_model_path();
    if (!std::filesystem::exists(model_path)) {
        GTEST_SKIP() << "Skipping real trained PPE model test: " << model_path << " not found.";
    }

    // 1. Verify exact [1, 3, 640, 640] -> [1, 7, 8400] float32 ONNX tensor contract
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "RealTrainedPPEContractTest");
        Ort::SessionOptions opts;
        Ort::Session session(env, model_path.c_str(), opts);

        ASSERT_EQ(session.GetInputCount(), 1U);
        ASSERT_EQ(session.GetOutputCount(), 1U);

        const Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
        ASSERT_EQ(input_type_info.GetONNXType(), ONNX_TYPE_TENSOR);
        const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
        EXPECT_EQ(input_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        const std::vector<int64_t> expected_in_shape = {1, 3, 640, 640};
        EXPECT_EQ(input_info.GetShape(), expected_in_shape);

        const Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
        ASSERT_EQ(output_type_info.GetONNXType(), ONNX_TYPE_TENSOR);
        const auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
        EXPECT_EQ(output_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        const std::vector<int64_t> expected_out_shape = {1, 7, 8400};
        EXPECT_EQ(output_info.GetShape(), expected_out_shape);
    }

    // 2. Verify end-to-end Detector construction and inference execution
    safety_engine::Detector detector(model_path.string(), 0.25f, 0.45f, 640, 640);

    const cv::Mat synthetic_frame(720, 1280, CV_8UC3, cv::Scalar(114, 114, 114));
    const std::vector<safety_engine::Detection> synth_dets = detector.detect(synthetic_frame);
    for (const auto& det : synth_dets) {
        EXPECT_GE(det.class_id, 0);
        EXPECT_LT(det.class_id, static_cast<int>(safety_engine::Detector::kExpectedNumClasses));
        EXPECT_GE(det.confidence, 0.25f);
        EXPECT_LE(det.confidence, 1.0f);
    }

    const std::filesystem::path video_path = resolve_project_path("assets/ppe_test_stream.mp4");
    if (std::filesystem::exists(video_path)) {
        cv::VideoCapture cap(video_path.string());
        cv::Mat real_frame;
        if (cap.isOpened() && cap.read(real_frame) && !real_frame.empty()) {
            const std::vector<safety_engine::Detection> real_dets = detector.detect(real_frame);
            EXPECT_FALSE(real_dets.empty());
            for (const auto& det : real_dets) {
                EXPECT_GE(det.class_id, 0);
                EXPECT_LT(det.class_id,
                          static_cast<int>(safety_engine::Detector::kExpectedNumClasses));
                EXPECT_GE(det.confidence, 0.25f);
                EXPECT_LE(det.confidence, 1.0f);
                EXPECT_GE(det.bbox.x, 0.0f);
                EXPECT_GE(det.bbox.y, 0.0f);
                EXPECT_LE(det.bbox.x + det.bbox.width, static_cast<float>(real_frame.cols) + 1e-2f);
                EXPECT_LE(det.bbox.y + det.bbox.height,
                          static_cast<float>(real_frame.rows) + 1e-2f);
            }
        }
    }
}
