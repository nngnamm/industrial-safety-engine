#pragma once

#include <opencv2/core/types.hpp>

namespace safety_engine {

enum class ClassId : int {
    Person = 0,
    Helmet = 1,
    Vest = 2
};

struct Detection {
    cv::Rect2f bbox;
    float confidence;
    int class_id;
};

struct LetterboxInfo {
    float scale;
    int pad_w;
    int pad_h;
};

}  // namespace safety_engine
