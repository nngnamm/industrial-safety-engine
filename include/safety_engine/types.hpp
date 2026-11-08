#pragma once

#include <opencv2/core/types.hpp>
#include <optional>
#include <string>
#include <vector>

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

struct HazardZone {
    std::string name;
    std::vector<cv::Point2f> polygon;
    bool require_helmet{true};
    bool require_vest{true};
};

struct WorkerStatus {
    cv::Rect2f person_bbox;
    float person_confidence{0.0f};
    cv::Point2f feet_point;
    std::optional<cv::Point2f> ground_point;
    bool ground_projection_valid{true};
    std::optional<cv::Rect2f> helmet_bbox;
    std::optional<cv::Rect2f> vest_bbox;
    bool has_helmet{false};
    bool has_vest{false};
    bool in_hazard_zone{false};
    bool is_violating{false};
    std::vector<std::string> violated_zones;
};

}  // namespace safety_engine
