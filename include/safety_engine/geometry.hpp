#pragma once

#include <opencv2/core/types.hpp>
#include <vector>

#include "safety_engine/types.hpp"

namespace safety_engine {

float compute_iou(const cv::Rect2f& a, const cv::Rect2f& b) noexcept;

float compute_containment_ratio(const cv::Rect2f& person_box, const cv::Rect2f& ppe_box) noexcept;

cv::Point2f compute_feet_point(const cv::Rect2f& person_box) noexcept;

bool is_point_in_polygon(const cv::Point2f& pt, const std::vector<cv::Point2f>& polygon);

class SafetyAnalyzer {
   public:
    explicit SafetyAnalyzer(float ppe_containment_thresh = 0.60f, float helmet_max_rel_y = 0.35f,
                            float vest_min_rel_y = 0.15f, float vest_max_rel_y = 0.75f);

    std::vector<WorkerStatus> analyze_frame(const std::vector<Detection>& detections,
                                            const std::vector<HazardZone>& hazard_zones) const;

   private:
    float ppe_containment_thresh_;
    float helmet_max_rel_y_;
    float vest_min_rel_y_;
    float vest_max_rel_y_;
};

}  // namespace safety_engine
