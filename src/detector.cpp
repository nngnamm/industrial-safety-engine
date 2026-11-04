#include "safety_engine/detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace safety_engine {

Detector::Detector(const std::string& model_path, float conf_threshold, float iou_threshold,
                   int input_width, int input_height)
    : env_(ORT_LOGGING_LEVEL_WARNING, "SafetyEngineDetector"),
      session_options_(),
      session_(nullptr),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      input_width_(input_width),
      input_height_(input_height),
      conf_threshold_(conf_threshold),
      iou_threshold_(iou_threshold),
      input_shape_{1, 3, static_cast<int64_t>(input_height), static_cast<int64_t>(input_width)} {
    if (model_path.empty()) {
        throw std::invalid_argument("Detector: model_path must not be empty.");
    }
    if (input_width_ <= 0 || input_height_ <= 0) {
        throw std::invalid_argument("Detector: input dimensions must be strictly positive.");
    }

    // Pre-allocate tensor and image workspaces once
    const std::size_t tensor_size =
        static_cast<std::size_t>(3) * static_cast<std::size_t>(input_height_) *
        static_cast<std::size_t>(input_width_);
    input_tensor_values_.assign(tensor_size, 0.0f);
    letterbox_buffer_.create(input_height_, input_width_, CV_8UC3);
    float_buffer_.create(input_height_, input_width_, CV_32FC3);

    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const std::filesystem::path fs_path(model_path);
    session_ = Ort::Session(env_, fs_path.c_str(), session_options_);

    Ort::AllocatorWithDefaultOptions allocator;

    const std::size_t num_inputs = session_.GetInputCount();
    if (num_inputs == 0) {
        throw std::runtime_error("Detector: ONNX model has zero inputs.");
    }
    input_names_str_.reserve(num_inputs);
    input_names_.reserve(num_inputs);
    for (std::size_t i = 0; i < num_inputs; ++i) {
        Ort::AllocatedStringPtr name_ptr = session_.GetInputNameAllocated(i, allocator);
        input_names_str_.emplace_back(name_ptr.get());
        input_names_.push_back(input_names_str_.back().c_str());
    }

    const std::size_t num_outputs = session_.GetOutputCount();
    if (num_outputs == 0) {
        throw std::runtime_error("Detector: ONNX model has zero outputs.");
    }
    output_names_str_.reserve(num_outputs);
    output_names_.reserve(num_outputs);
    for (std::size_t i = 0; i < num_outputs; ++i) {
        Ort::AllocatedStringPtr name_ptr = session_.GetOutputNameAllocated(i, allocator);
        output_names_str_.emplace_back(name_ptr.get());
        output_names_.push_back(output_names_str_.back().c_str());
    }
}

LetterboxInfo Detector::compute_letterbox_info(int src_w, int src_h, int dst_w, int dst_h) {
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        throw std::invalid_argument(
            "Detector::compute_letterbox_info: source and destination dimensions must be > 0.");
    }

    const float scale = std::min(static_cast<float>(dst_w) / static_cast<float>(src_w),
                                 static_cast<float>(dst_h) / static_cast<float>(src_h));
    const int new_unpad_w = static_cast<int>(std::round(static_cast<float>(src_w) * scale));
    const int new_unpad_h = static_cast<int>(std::round(static_cast<float>(src_h) * scale));

    const int pad_w = (dst_w - new_unpad_w) / 2;
    const int pad_h = (dst_h - new_unpad_h) / 2;

    return LetterboxInfo{scale, pad_w, pad_h};
}

cv::Rect2f Detector::unletterbox_bbox(float cx, float cy, float w, float h,
                                      const LetterboxInfo& info, int orig_w, int orig_h) {
    const float half_w = 0.5f * w;
    const float half_h = 0.5f * h;

    float x1 = (cx - half_w - static_cast<float>(info.pad_w)) / info.scale;
    float y1 = (cy - half_h - static_cast<float>(info.pad_h)) / info.scale;
    float x2 = (cx + half_w - static_cast<float>(info.pad_w)) / info.scale;
    float y2 = (cy + half_h - static_cast<float>(info.pad_h)) / info.scale;

    const float max_w = static_cast<float>(orig_w);
    const float max_h = static_cast<float>(orig_h);

    x1 = std::clamp(x1, 0.0f, max_w);
    y1 = std::clamp(y1, 0.0f, max_h);
    x2 = std::clamp(x2, 0.0f, max_w);
    y2 = std::clamp(y2, 0.0f, max_h);

    return cv::Rect2f(x1, y1, std::max(0.0f, x2 - x1), std::max(0.0f, y2 - y1));
}

LetterboxInfo Detector::preprocess(const cv::Mat& frame) {
    if (frame.type() != CV_8UC3) {
        throw std::invalid_argument("Detector::preprocess: expected 3-channel BGR CV_8UC3 frame.");
    }

    const LetterboxInfo info =
        compute_letterbox_info(frame.cols, frame.rows, input_width_, input_height_);

    const int new_unpad_w =
        static_cast<int>(std::round(static_cast<float>(frame.cols) * info.scale));
    const int new_unpad_h =
        static_cast<int>(std::round(static_cast<float>(frame.rows) * info.scale));

    const int top = info.pad_h;
    const int bottom = input_height_ - new_unpad_h - top;
    const int left = info.pad_w;
    const int right = input_width_ - new_unpad_w - left;

    const cv::Mat* unpadded_src = &frame;
    if (new_unpad_w != frame.cols || new_unpad_h != frame.rows) {
        cv::resize(frame, resized_buffer_, cv::Size(new_unpad_w, new_unpad_h), 0.0, 0.0,
                   cv::INTER_LINEAR);
        unpadded_src = &resized_buffer_;
    }

    cv::copyMakeBorder(*unpadded_src, letterbox_buffer_, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // Convert BGR uint8 -> RGB uint8 in-place, then scale to normalized float32 [0.0, 1.0]
    cv::cvtColor(letterbox_buffer_, letterbox_buffer_, cv::COLOR_BGR2RGB);
    letterbox_buffer_.convertTo(float_buffer_, CV_32FC3, 1.0 / 255.0);

    // Split HWC RGB float_buffer_ directly into planar NCHW input_tensor_values_ (zero extra alloc)
    const std::size_t channel_area =
        static_cast<std::size_t>(input_height_) * static_cast<std::size_t>(input_width_);
    float* tensor_ptr = input_tensor_values_.data();
    std::array<cv::Mat, 3> chw_channels = {
        cv::Mat(input_height_, input_width_, CV_32F, tensor_ptr),
        cv::Mat(input_height_, input_width_, CV_32F, tensor_ptr + channel_area),
        cv::Mat(input_height_, input_width_, CV_32F, tensor_ptr + 2 * channel_area),
    };
    cv::split(float_buffer_, chw_channels.data());

    return info;
}

std::vector<Detection> Detector::postprocess(Ort::Value& output_tensor,
                                             const LetterboxInfo& letterbox_info, int orig_w,
                                             int orig_h) const {
    const Ort::TensorTypeAndShapeInfo shape_info = output_tensor.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = shape_info.GetShape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] < 5 || shape[2] <= 0) {
        throw std::runtime_error(
            "Detector::postprocess: unexpected output tensor shape (expected [1, 4+C, N]).");
    }

    const int rows = static_cast<int>(shape[1]);  // 4 + num_classes
    const int cols = static_cast<int>(shape[2]);  // num_anchors (e.g., 8400)
    const int num_classes = rows - 4;

    float* raw_ptr = output_tensor.GetTensorMutableData<float>();
    const cv::Mat raw_output(rows, cols, CV_32F, raw_ptr);
    const cv::Mat transposed = raw_output.t();  // [num_anchors, 4 + num_classes]

    std::vector<cv::Rect2f> candidate_boxes;
    std::vector<cv::Rect2d> nms_boxes;
    std::vector<float> candidate_scores;
    std::vector<int> candidate_class_ids;

    // Offset boxes per class during NMS so overlapping multi-class detections (e.g., Person + Vest)
    // do not suppress one another.
    const double class_offset_stride = static_cast<double>(std::max(orig_w, orig_h) + 4096);

    for (int i = 0; i < transposed.rows; ++i) {
        const float* row_ptr = transposed.ptr<float>(i);

        int best_class_id = 0;
        float max_score = row_ptr[4];
        for (int c = 1; c < num_classes; ++c) {
            const float score = row_ptr[4 + c];
            if (score > max_score) {
                max_score = score;
                best_class_id = c;
            }
        }

        if (max_score < conf_threshold_) {
            continue;
        }

        const cv::Rect2f box = unletterbox_bbox(row_ptr[0], row_ptr[1], row_ptr[2], row_ptr[3],
                                                letterbox_info, orig_w, orig_h);
        if (box.width <= 0.0f || box.height <= 0.0f) {
            continue;
        }

        const double offset = static_cast<double>(best_class_id) * class_offset_stride;
        candidate_boxes.push_back(box);
        nms_boxes.emplace_back(static_cast<double>(box.x) + offset,
                               static_cast<double>(box.y) + offset, static_cast<double>(box.width),
                               static_cast<double>(box.height));
        candidate_scores.push_back(max_score);
        candidate_class_ids.push_back(best_class_id);
    }

    std::vector<int> keep_indices;
    if (!nms_boxes.empty()) {
        cv::dnn::NMSBoxes(nms_boxes, candidate_scores, conf_threshold_, iou_threshold_,
                          keep_indices);
    }

    std::vector<Detection> detections;
    detections.reserve(keep_indices.size());
    for (const int idx : keep_indices) {
        detections.push_back(
            Detection{candidate_boxes[idx], candidate_scores[idx], candidate_class_ids[idx]});
    }

    return detections;
}

std::vector<Detection> Detector::detect(const cv::Mat& frame) {
    if (frame.empty()) {
        throw std::invalid_argument("Detector::detect: input frame must not be empty.");
    }

    const LetterboxInfo letterbox_info = preprocess(frame);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_tensor_values_.data(), input_tensor_values_.size(),
        input_shape_.data(), input_shape_.size());

    std::vector<Ort::Value> output_tensors =
        session_.Run(Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor, 1,
                     output_names_.data(), output_names_.size());

    return postprocess(output_tensors.front(), letterbox_info, frame.cols, frame.rows);
}

}  // namespace safety_engine
