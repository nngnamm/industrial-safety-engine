#pragma once

#include <array>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <optional>
#include <vector>

#include "safety_engine/types.hpp"

namespace safety_engine {

float compute_iou(const cv::Rect2f& a, const cv::Rect2f& b) noexcept;

float compute_containment_ratio(const cv::Rect2f& person_box, const cv::Rect2f& ppe_box) noexcept;

cv::Point2f compute_feet_point(const cv::Rect2f& person_box) noexcept;

bool is_point_in_polygon(const cv::Point2f& pt, const std::vector<cv::Point2f>& polygon);

cv::Matx33f compute_ground_homography(const std::array<cv::Point2f, 4>& image_points,
                                      const std::array<cv::Point2f, 4>& ground_points);

cv::Point2f project_point_homography(const cv::Point2f& pt, const cv::Matx33f& H) noexcept;

class SafetyAnalyzer {
   public:
    explicit SafetyAnalyzer(float ppe_containment_thresh = 0.60f, float helmet_max_rel_y = 0.35f,
                            float vest_min_rel_y = 0.15f, float vest_max_rel_y = 0.75f);

    void set_homography(const cv::Matx33f& H) noexcept;

    void clear_homography() noexcept;

    bool has_homography() const noexcept { return homography_.has_value(); }

    const std::optional<cv::Matx33f>& homography() const noexcept { return homography_; }

    std::vector<WorkerStatus> analyze_frame(const std::vector<Detection>& detections,
                                            const std::vector<HazardZone>& hazard_zones) const;

   private:
    float ppe_containment_thresh_;
    float helmet_max_rel_y_;
    float vest_min_rel_y_;
    float vest_max_rel_y_;
    std::optional<cv::Matx33f> homography_;
};

}  // namespace safety_engine
