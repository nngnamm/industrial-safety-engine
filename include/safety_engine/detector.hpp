#pragma once

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdint>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "safety_engine/types.hpp"

namespace safety_engine {

class Detector {
   public:
    explicit Detector(const std::string& model_path, float conf_threshold = 0.45f,
                      float iou_threshold = 0.50f, int input_width = 640,
                      int input_height = 640);
    ~Detector() = default;

    // Non-copyable and non-movable to preserve const char* views into owned name strings
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;
    Detector(Detector&&) = delete;
    Detector& operator=(Detector&&) = delete;

    std::vector<Detection> detect(const cv::Mat& frame);

    static LetterboxInfo compute_letterbox_info(int src_w, int src_h, int dst_w, int dst_h);

    static cv::Rect2f unletterbox_bbox(float cx, float cy, float w, float h,
                                       const LetterboxInfo& info, int orig_w, int orig_h);

   private:
    LetterboxInfo preprocess(const cv::Mat& frame);
    std::vector<Detection> postprocess(Ort::Value& output_tensor,
                                       const LetterboxInfo& letterbox_info, int orig_w,
                                       int orig_h) const;

    // Member declaration order strictly preserves LIFO destruction safety:
    // session_ is destroyed before session_options_ and env_;
    // const char* name views are destroyed before their owning std::string vectors.
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};
    std::vector<std::string> input_names_str_;
    std::vector<const char*> input_names_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> output_names_;
    int input_width_;
    int input_height_;
    float conf_threshold_;
    float iou_threshold_;
    std::array<int64_t, 4> input_shape_;

    // Pre-allocated buffers (allocated once in constructor for zero-allocation steady state)
    std::vector<float> input_tensor_values_;
    cv::Mat resized_buffer_;
    cv::Mat letterbox_buffer_;
    cv::Mat float_buffer_;
};

}  // namespace safety_engine
