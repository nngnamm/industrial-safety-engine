#include "safety_engine/geometry.hpp"

#include <cmath>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace safety_engine {

namespace {

constexpr float kAreaEpsilon = 1e-6f;
constexpr float kScoreTieEpsilon = 1e-5f;

const Detection* find_best_ppe_match(const cv::Rect2f& person_box,
                                     const std::vector<const Detection*>& candidates, float min_y,
                                     float max_y, float containment_thresh) noexcept {
    const Detection* best_match = nullptr;
    float best_containment = -1.0f;
    float best_confidence = -1.0f;

    for (const Detection* cand : candidates) {
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

        const bool strictly_better = containment > (best_containment + kScoreTieEpsilon);
        const bool tied_with_higher_conf =
            (std::abs(containment - best_containment) <= kScoreTieEpsilon) &&
            (cand->confidence > best_confidence);

        if (strictly_better || tied_with_higher_conf) {
            best_match = cand;
            best_containment = containment;
            best_confidence = cand->confidence;
        }
    }

    return best_match;
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

        if (status.person_bbox.width > 0.0f && status.person_bbox.height > 0.0f) {
            const float py = status.person_bbox.y;
            const float ph = status.person_bbox.height;

            const Detection* matched_helmet = find_best_ppe_match(
                status.person_bbox, helmets, py, py + helmet_max_rel_y_ * ph,
                ppe_containment_thresh_);
            if (matched_helmet != nullptr) {
                status.helmet_bbox = matched_helmet->bbox;
                status.has_helmet = true;
            }

            const Detection* matched_vest = find_best_ppe_match(
                status.person_bbox, vests, py + vest_min_rel_y_ * ph, py + vest_max_rel_y_ * ph,
                ppe_containment_thresh_);
            if (matched_vest != nullptr) {
                status.vest_bbox = matched_vest->bbox;
                status.has_vest = true;
            }
        }

        for (const HazardZone& zone : hazard_zones) {
            if (is_point_in_polygon(status.feet_point, zone.polygon)) {
                status.in_hazard_zone = true;
                const bool missing_helmet = zone.require_helmet && !status.has_helmet;
                const bool missing_vest = zone.require_vest && !status.has_vest;
                if (missing_helmet || missing_vest) {
                    status.is_violating = true;
                    status.violated_zones.push_back(zone.name);
                }
            }
        }

        results.push_back(std::move(status));
    }

    return results;
}

}  // namespace safety_engine
