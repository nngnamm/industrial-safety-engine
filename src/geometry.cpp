#include "safety_engine/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <tuple>

namespace safety_engine {

namespace {

constexpr float kAreaEpsilon = 1e-6f;
constexpr double kMinHomographyDet = 1e-9;

struct PPEAssignmentCandidate {
    std::size_t person_idx;
    std::size_t ppe_idx;
    float score;
    float ppe_confidence;
};

void assign_ppe_one_to_one(const std::vector<const Detection*>& persons,
                           const std::vector<const Detection*>& ppe_items, float min_rel_y,
                           float max_rel_y, float containment_thresh, bool is_helmet,
                           std::vector<WorkerStatus>& results) {
    if (persons.empty() || ppe_items.empty()) {
        return;
    }

    std::vector<PPEAssignmentCandidate> candidates;
    candidates.reserve(persons.size() * ppe_items.size());

    const float band_rel_h = std::max(max_rel_y - min_rel_y, kAreaEpsilon);
    const float target_rel_y = 0.5f * (min_rel_y + max_rel_y);

    for (std::size_t p_idx = 0; p_idx < persons.size(); ++p_idx) {
        const cv::Rect2f& person_box = persons[p_idx]->bbox;
        if (person_box.width <= 0.0f || person_box.height <= 0.0f) {
            continue;
        }

        const float min_y = person_box.y + min_rel_y * person_box.height;
        const float max_y = person_box.y + max_rel_y * person_box.height;
        const float person_cx = person_box.x + 0.5f * person_box.width;
        const float target_cy = person_box.y + target_rel_y * person_box.height;
        const float half_w = std::max(0.5f * person_box.width, kAreaEpsilon);
        const float half_band_h = std::max(0.5f * band_rel_h * person_box.height, kAreaEpsilon);

        for (std::size_t item_idx = 0; item_idx < ppe_items.size(); ++item_idx) {
            const Detection* cand = ppe_items[item_idx];
            const cv::Rect2f& ppe_box = cand->bbox;
            if (ppe_box.width <= 0.0f || ppe_box.height <= 0.0f) {
                continue;
            }

            const float cy = ppe_box.y + 0.5f * ppe_box.height;
            if (cy < min_y || cy > max_y) {
                continue;
            }

            const float containment = compute_containment_ratio(person_box, ppe_box);
            if (containment < containment_thresh) {
                continue;
            }

            const float cx = ppe_box.x + 0.5f * ppe_box.width;
            const float norm_dx = (cx - person_cx) / half_w;
            const float norm_dy = (cy - target_cy) / half_band_h;
            const float center_affinity =
                1.0f / (1.0f + norm_dx * norm_dx + 0.25f * norm_dy * norm_dy);

            // Primary rank by IoPA containment; break ties by anatomical center proximity & score.
            const float score = containment + 0.05f * center_affinity;
            candidates.push_back(
                PPEAssignmentCandidate{p_idx, item_idx, score, cand->confidence});
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const PPEAssignmentCandidate& lhs, const PPEAssignmentCandidate& rhs) {
                  if (lhs.score != rhs.score) {
                      return lhs.score > rhs.score;
                  }
                  return std::tie(lhs.person_idx, lhs.ppe_idx) <
                         std::tie(rhs.person_idx, rhs.ppe_idx);
              });

    std::vector<bool> person_claimed(persons.size(), false);
    std::vector<bool> ppe_claimed(ppe_items.size(), false);

    for (const PPEAssignmentCandidate& cand : candidates) {
        if (person_claimed[cand.person_idx] || ppe_claimed[cand.ppe_idx]) {
            continue;
        }
        person_claimed[cand.person_idx] = true;
        ppe_claimed[cand.ppe_idx] = true;

        WorkerStatus& status = results[cand.person_idx];
        const cv::Rect2f& matched_box = ppe_items[cand.ppe_idx]->bbox;
        if (is_helmet) {
            status.helmet_bbox = matched_box;
            status.has_helmet = true;
        } else {
            status.vest_bbox = matched_box;
            status.has_vest = true;
        }
    }
}

void validate_homography_matrix(const cv::Matx33f& H, const char* caller) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            if (!std::isfinite(H(r, c))) {
                throw std::invalid_argument(std::string(caller) +
                                            ": homography matrix contains non-finite entries.");
            }
        }
    }
    const double det = cv::determinant(H);
    if (!std::isfinite(det) || std::abs(det) <= kMinHomographyDet) {
        throw std::invalid_argument(std::string(caller) +
                                    ": homography matrix is singular or degenerate.");
    }
}

}  // namespace

float compute_iou(const cv::Rect2f& a, const cv::Rect2f& b) noexcept {
    if (a.width <= 0.0f || a.height <= 0.0f || b.width <= 0.0f || b.height <= 0.0f) {
        return 0.0f;
    }

    const float inter_area = (a & b).area();
    if (inter_area <= 0.0f) {
        return 0.0f;
    }

    const float union_area = a.area() + b.area() - inter_area;
    if (union_area <= kAreaEpsilon) {
        return 0.0f;
    }

    return inter_area / union_area;
}

float compute_containment_ratio(const cv::Rect2f& person_box, const cv::Rect2f& ppe_box) noexcept {
    if (person_box.width <= 0.0f || person_box.height <= 0.0f || ppe_box.width <= 0.0f ||
        ppe_box.height <= 0.0f) {
        return 0.0f;
    }

    const float ppe_area = ppe_box.area();
    if (ppe_area <= kAreaEpsilon) {
        return 0.0f;
    }

    const float inter_area = (person_box & ppe_box).area();
    return inter_area / ppe_area;
}

cv::Point2f compute_feet_point(const cv::Rect2f& person_box) noexcept {
    return cv::Point2f(person_box.x + person_box.width * 0.5f, person_box.y + person_box.height);
}

bool is_point_in_polygon(const cv::Point2f& pt, const std::vector<cv::Point2f>& polygon) {
    if (polygon.size() < 3U) {
        return false;
    }
    return cv::pointPolygonTest(polygon, pt, false) >= 0.0;
}

cv::Matx33f compute_ground_homography(const std::array<cv::Point2f, 4>& image_points,
                                      const std::array<cv::Point2f, 4>& ground_points) {
    const cv::Mat h64 = cv::getPerspectiveTransform(image_points.data(), ground_points.data());
    if (h64.rows != 3 || h64.cols != 3) {
        throw std::invalid_argument(
            "compute_ground_homography: failed to solve 3x3 perspective transform.");
    }

    const double w0 = h64.at<double>(2, 0);
    const double w1 = h64.at<double>(2, 1);
    const double w2 = h64.at<double>(2, 2);
    const double row2_norm = std::sqrt(w0 * w0 + w1 * w1 + w2 * w2);
    if (!std::isfinite(row2_norm) || row2_norm <= kMinHomographyDet) {
        throw std::invalid_argument(
            "compute_ground_homography: degenerate point configuration produced invalid "
            "homography.");
    }

    cv::Matx33f H;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            const double val = h64.at<double>(r, c) / row2_norm;
            if (!std::isfinite(val)) {
                throw std::invalid_argument(
                    "compute_ground_homography: degenerate point configuration produced non-finite "
                    "homography.");
            }
            H(r, c) = static_cast<float>(val);
        }
    }
    validate_homography_matrix(H, "compute_ground_homography");
    return H;
}

std::optional<cv::Point2f> project_point_homography(const cv::Point2f& pt,
                                                    const cv::Matx33f& H) noexcept {
    const float x = H(0, 0) * pt.x + H(0, 1) * pt.y + H(0, 2);
    const float y = H(1, 0) * pt.x + H(1, 1) * pt.y + H(1, 2);
    const float w = H(2, 0) * pt.x + H(2, 1) * pt.y + H(2, 2);
    if (!std::isfinite(w) || std::abs(w) <= 1e-6f) {
        return std::nullopt;
    }
    const float proj_x = x / w;
    const float proj_y = y / w;
    if (std::isnan(proj_x) || std::isinf(proj_x) || std::isnan(proj_y) || std::isinf(proj_y)) {
        return std::nullopt;
    }
    return cv::Point2f(proj_x, proj_y);
}

SafetyAnalyzer::SafetyAnalyzer(float ppe_containment_thresh, float helmet_max_rel_y,
                               float vest_min_rel_y, float vest_max_rel_y)
    : ppe_containment_thresh_(ppe_containment_thresh),
      helmet_max_rel_y_(helmet_max_rel_y),
      vest_min_rel_y_(vest_min_rel_y),
      vest_max_rel_y_(vest_max_rel_y) {
    if (ppe_containment_thresh_ < 0.0f || ppe_containment_thresh_ > 1.0f) {
        throw std::invalid_argument(
            "SafetyAnalyzer: ppe_containment_thresh must be within [0.0, 1.0].");
    }
    if (helmet_max_rel_y_ <= 0.0f || helmet_max_rel_y_ > 1.0f) {
        throw std::invalid_argument("SafetyAnalyzer: helmet_max_rel_y must be within (0.0, 1.0].");
    }
    if (vest_min_rel_y_ < 0.0f || vest_max_rel_y_ > 1.0f || vest_min_rel_y_ >= vest_max_rel_y_) {
        throw std::invalid_argument(
            "SafetyAnalyzer: vest relative Y bounds must satisfy 0.0 <= min < max <= 1.0.");
    }
}

void SafetyAnalyzer::set_homography(const cv::Matx33f& H) {
    validate_homography_matrix(H, "SafetyAnalyzer::set_homography");
    homography_ = H;
}

void SafetyAnalyzer::clear_homography() noexcept {
    homography_.reset();
}

std::vector<WorkerStatus> SafetyAnalyzer::analyze_frame(
    const std::vector<Detection>& detections, const std::vector<HazardZone>& hazard_zones) const {
    std::vector<const Detection*> persons;
    std::vector<const Detection*> helmets;
    std::vector<const Detection*> vests;

    persons.reserve(detections.size());
    helmets.reserve(detections.size());
    vests.reserve(detections.size());

    for (const Detection& det : detections) {
        switch (static_cast<ClassId>(det.class_id)) {
            case ClassId::Person:
                persons.push_back(&det);
                break;
            case ClassId::Helmet:
                helmets.push_back(&det);
                break;
            case ClassId::Vest:
                vests.push_back(&det);
                break;
            default:
                break;
        }
    }

    std::vector<WorkerStatus> results;
    results.reserve(persons.size());

    for (const Detection* person_det : persons) {
        WorkerStatus status;
        status.person_bbox = person_det->bbox;
        status.person_confidence = person_det->confidence;
        status.feet_point = compute_feet_point(status.person_bbox);
        if (homography_.has_value()) {
            const std::optional<cv::Point2f> projected =
                project_point_homography(status.feet_point, *homography_);
            if (projected.has_value()) {
                status.ground_projection_valid = true;
                status.ground_point = *projected;
            } else {
                status.ground_projection_valid = false;
                status.ground_point = std::nullopt;
            }
        } else {
            status.ground_projection_valid = true;
            status.ground_point = status.feet_point;
        }
        results.push_back(std::move(status));
    }

    // Greedy 1-to-1 containment assignment for helmets and vests
    assign_ppe_one_to_one(persons, helmets, 0.0f, helmet_max_rel_y_, ppe_containment_thresh_,
                          /*is_helmet=*/true, results);
    assign_ppe_one_to_one(persons, vests, vest_min_rel_y_, vest_max_rel_y_, ppe_containment_thresh_,
                          /*is_helmet=*/false, results);

    // Evaluate hazard zone compliance in ground-plane coordinates (defaults to feet_point)
    for (WorkerStatus& status : results) {
        if (!status.ground_projection_valid || !status.ground_point.has_value()) {
            continue;
        }
        for (const HazardZone& zone : hazard_zones) {
            if (is_point_in_polygon(*status.ground_point, zone.polygon)) {
                status.in_hazard_zone = true;
                const bool missing_helmet = zone.require_helmet && !status.has_helmet;
                const bool missing_vest = zone.require_vest && !status.has_vest;
                if (missing_helmet || missing_vest) {
                    status.is_violating = true;
                    status.violated_zones.push_back(zone.name);
                }
            }
        }
    }

    return results;
}

}  // namespace safety_engine
