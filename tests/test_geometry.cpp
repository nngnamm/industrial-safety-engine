#include <gtest/gtest.h>

#include <limits>
#include <opencv2/core/types.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "safety_engine/geometry.hpp"
#include "safety_engine/types.hpp"

namespace {

using safety_engine::ClassId;
using safety_engine::Detection;
using safety_engine::HazardZone;
using safety_engine::SafetyAnalyzer;
using safety_engine::WorkerStatus;

}  // namespace

TEST(GeometryTest, IoUAndContainmentMath) {
    const cv::Rect2f box_a(0.0f, 0.0f, 100.0f, 100.0f);

    // 1. Disjoint boxes
    const cv::Rect2f disjoint(200.0f, 200.0f, 50.0f, 50.0f);
    EXPECT_FLOAT_EQ(safety_engine::compute_iou(box_a, disjoint), 0.0f);
    EXPECT_FLOAT_EQ(safety_engine::compute_containment_ratio(box_a, disjoint), 0.0f);

    // 2. Identical boxes
    EXPECT_FLOAT_EQ(safety_engine::compute_iou(box_a, box_a), 1.0f);
    EXPECT_FLOAT_EQ(safety_engine::compute_containment_ratio(box_a, box_a), 1.0f);

    // 3. Partially overlapping boxes: intersection = 50x50 = 2500, union = 10000 + 10000 - 2500 = 17500
    const cv::Rect2f partial(50.0f, 50.0f, 100.0f, 100.0f);
    EXPECT_NEAR(safety_engine::compute_iou(box_a, partial), 2500.0f / 17500.0f, 1e-5f);
    EXPECT_NEAR(safety_engine::compute_containment_ratio(box_a, partial), 2500.0f / 10000.0f, 1e-5f);

    // 4. Fully contained PPE box inside person box
    const cv::Rect2f inner_ppe(25.0f, 10.0f, 30.0f, 30.0f);
    EXPECT_NEAR(safety_engine::compute_iou(box_a, inner_ppe), 900.0f / 10000.0f, 1e-5f);
    EXPECT_FLOAT_EQ(safety_engine::compute_containment_ratio(box_a, inner_ppe), 1.0f);

    // 5. Zero-area / degenerate edge cases
    const cv::Rect2f zero_box(10.0f, 10.0f, 0.0f, 0.0f);
    const cv::Rect2f negative_box(10.0f, 10.0f, -20.0f, 30.0f);
    EXPECT_FLOAT_EQ(safety_engine::compute_iou(box_a, zero_box), 0.0f);
    EXPECT_FLOAT_EQ(safety_engine::compute_iou(zero_box, zero_box), 0.0f);
    EXPECT_FLOAT_EQ(safety_engine::compute_iou(box_a, negative_box), 0.0f);
    EXPECT_FLOAT_EQ(safety_engine::compute_containment_ratio(box_a, zero_box), 0.0f);
    EXPECT_FLOAT_EQ(safety_engine::compute_containment_ratio(zero_box, inner_ppe), 0.0f);

    // 6. Feet point & degenerate polygon edge case
    const cv::Point2f feet = safety_engine::compute_feet_point(cv::Rect2f(100.0f, 100.0f, 100.0f, 300.0f));
    EXPECT_FLOAT_EQ(feet.x, 150.0f);
    EXPECT_FLOAT_EQ(feet.y, 400.0f);
    EXPECT_FALSE(safety_engine::is_point_in_polygon(feet, {{0.0f, 0.0f}, {500.0f, 0.0f}}));
}

TEST(GeometryTest, AnatomicalPPEBinding) {
    const SafetyAnalyzer analyzer;
    const Detection person{cv::Rect2f(100.0f, 100.0f, 100.0f, 300.0f), 0.95f,
                           static_cast<int>(ClassId::Person)};

    // Case 1: Helmet properly positioned on the worker's head (cy = 130 in [100, 205])
    {
        const Detection head_helmet{cv::Rect2f(130.0f, 110.0f, 40.0f, 40.0f), 0.90f,
                                    static_cast<int>(ClassId::Helmet)};
        const Detection torso_vest{cv::Rect2f(115.0f, 170.0f, 70.0f, 110.0f), 0.88f,
                                   static_cast<int>(ClassId::Vest)};

        const std::vector<WorkerStatus> statuses =
            analyzer.analyze_frame({person, head_helmet, torso_vest}, {});
        ASSERT_EQ(statuses.size(), 1U);
        EXPECT_TRUE(statuses[0].has_helmet);
        ASSERT_TRUE(statuses[0].helmet_bbox.has_value());
        EXPECT_EQ(*statuses[0].helmet_bbox, head_helmet.bbox);
        EXPECT_TRUE(statuses[0].has_vest);
        ASSERT_TRUE(statuses[0].vest_bbox.has_value());
        EXPECT_EQ(*statuses[0].vest_bbox, torso_vest.bbox);
    }

    // Case 2: Helmet dropped on the floor at the worker's feet (cy = 370 > 205) must be rejected
    {
        const Detection dropped_helmet{cv::Rect2f(130.0f, 350.0f, 40.0f, 40.0f), 0.92f,
                                       static_cast<int>(ClassId::Helmet)};

        const std::vector<WorkerStatus> statuses =
            analyzer.analyze_frame({person, dropped_helmet}, {});
        ASSERT_EQ(statuses.size(), 1U);
        EXPECT_FALSE(statuses[0].has_helmet);
        EXPECT_FALSE(statuses[0].helmet_bbox.has_value());
    }
}

TEST(GeometryTest, HazardZoneViolationDetection) {
    const SafetyAnalyzer analyzer;
    const HazardZone hazard_zone{
        "HighVoltageBay",
        {{0.0f, 0.0f}, {500.0f, 0.0f}, {500.0f, 500.0f}, {0.0f, 500.0f}},
        true,
        true,
    };

    // 1. Worker inside hazard zone WITH vest ONLY (missing helmet) -> violation
    {
        const Detection inside_person{cv::Rect2f(100.0f, 100.0f, 100.0f, 300.0f), 0.95f,
                                      static_cast<int>(ClassId::Person)};
        const Detection vest{cv::Rect2f(115.0f, 170.0f, 70.0f, 110.0f), 0.90f,
                             static_cast<int>(ClassId::Vest)};

        const std::vector<WorkerStatus> statuses =
            analyzer.analyze_frame({inside_person, vest}, {hazard_zone});
        ASSERT_EQ(statuses.size(), 1U);
        EXPECT_TRUE(statuses[0].in_hazard_zone);
        EXPECT_FALSE(statuses[0].has_helmet);
        EXPECT_TRUE(statuses[0].has_vest);
        EXPECT_TRUE(statuses[0].is_violating);
        ASSERT_EQ(statuses[0].violated_zones.size(), 1U);
        EXPECT_EQ(statuses[0].violated_zones[0], "HighVoltageBay");
    }

    // 2. Worker inside hazard zone WITH both helmet and vest -> compliant (no violation)
    {
        const Detection inside_person{cv::Rect2f(100.0f, 100.0f, 100.0f, 300.0f), 0.95f,
                                      static_cast<int>(ClassId::Person)};
        const Detection helmet{cv::Rect2f(130.0f, 110.0f, 40.0f, 40.0f), 0.91f,
                               static_cast<int>(ClassId::Helmet)};
        const Detection vest{cv::Rect2f(115.0f, 170.0f, 70.0f, 110.0f), 0.90f,
                             static_cast<int>(ClassId::Vest)};

        const std::vector<WorkerStatus> statuses =
            analyzer.analyze_frame({inside_person, helmet, vest}, {hazard_zone});
        ASSERT_EQ(statuses.size(), 1U);
        EXPECT_TRUE(statuses[0].in_hazard_zone);
        EXPECT_TRUE(statuses[0].has_helmet);
        EXPECT_TRUE(statuses[0].has_vest);
        EXPECT_FALSE(statuses[0].is_violating);
        EXPECT_TRUE(statuses[0].violated_zones.empty());
    }

    // 3. Worker outside hazard zone at (600, 600, 80, 200) without PPE -> not in zone, no violation
    {
        const Detection outside_person{cv::Rect2f(600.0f, 600.0f, 80.0f, 200.0f), 0.93f,
                                       static_cast<int>(ClassId::Person)};

        const std::vector<WorkerStatus> statuses =
            analyzer.analyze_frame({outside_person}, {hazard_zone});
        ASSERT_EQ(statuses.size(), 1U);
        EXPECT_FALSE(statuses[0].in_hazard_zone);
        EXPECT_FALSE(statuses[0].has_helmet);
        EXPECT_FALSE(statuses[0].has_vest);
        EXPECT_FALSE(statuses[0].is_violating);
        EXPECT_TRUE(statuses[0].violated_zones.empty());
    }
}

TEST(GeometryTest, OneToOnePPEAssignmentPreventsDoubleClaim) {
    const SafetyAnalyzer analyzer;

    // Two overlapping Person boxes standing close together:
    // Person 0 (left/background): [100, 100, 120, 300], head center X = 160
    // Person 1 (right/foreground, closer to helmet): [130, 100, 120, 300], head center X = 190
    const Detection person_far{cv::Rect2f(100.0f, 100.0f, 120.0f, 300.0f), 0.94f,
                               static_cast<int>(ClassId::Person)};
    const Detection person_closer{cv::Rect2f(130.0f, 100.0f, 120.0f, 300.0f), 0.95f,
                                  static_cast<int>(ClassId::Person)};

    // Single Helmet at [172, 115, 36, 36] (center X = 190, inside BOTH person boxes [100..220]
    // and [130..250] with 100% containment, but centered directly on person_closer's head).
    const Detection shared_helmet{cv::Rect2f(172.0f, 115.0f, 36.0f, 36.0f), 0.92f,
                                  static_cast<int>(ClassId::Helmet)};

    // Single Vest at [165, 175, 50, 95] (center X = 190, also centered on person_closer).
    const Detection shared_vest{cv::Rect2f(165.0f, 175.0f, 50.0f, 95.0f), 0.91f,
                                static_cast<int>(ClassId::Vest)};

    // Place person_far FIRST in the input list to prove 1-to-1 assignment is not a naive
    // first-come-first-served loop and never allows both persons to claim the same PPE.
    const std::vector<WorkerStatus> statuses =
        analyzer.analyze_frame({person_far, person_closer, shared_helmet, shared_vest}, {});
    ASSERT_EQ(statuses.size(), 2U);

    // Only the closer person (index 1) gets the helmet and vest; person 0 gets neither.
    EXPECT_FALSE(statuses[0].has_helmet);
    EXPECT_FALSE(statuses[0].helmet_bbox.has_value());
    EXPECT_FALSE(statuses[0].has_vest);
    EXPECT_FALSE(statuses[0].vest_bbox.has_value());

    EXPECT_TRUE(statuses[1].has_helmet);
    ASSERT_TRUE(statuses[1].helmet_bbox.has_value());
    EXPECT_EQ(*statuses[1].helmet_bbox, shared_helmet.bbox);
    EXPECT_TRUE(statuses[1].has_vest);
    ASSERT_TRUE(statuses[1].vest_bbox.has_value());
    EXPECT_EQ(*statuses[1].vest_bbox, shared_vest.bbox);
}

TEST(GeometryTest, HomographyGroundPlaneProjection) {
    // Perspective trapezoidal floor quad in image space mapping to a 10m x 10m metric square
    const std::array<cv::Point2f, 4> image_quad = {{
        {400.0f, 200.0f},  // top-left (far)
        {600.0f, 200.0f},  // top-right (far)
        {800.0f, 600.0f},  // bottom-right (near)
        {200.0f, 600.0f},  // bottom-left (near)
    }};
    const std::array<cv::Point2f, 4> ground_square = {{
        {0.0f, 0.0f},
        {10.0f, 0.0f},
        {10.0f, 10.0f},
        {0.0f, 10.0f},
    }};

    const cv::Matx33f H = safety_engine::compute_ground_homography(image_quad, ground_square);

    // Verify all 4 corners project accurately to the metric square in ground coordinates
    for (std::size_t i = 0; i < 4U; ++i) {
        const std::optional<cv::Point2f> projected =
            safety_engine::project_point_homography(image_quad[i], H);
        ASSERT_TRUE(projected.has_value());
        EXPECT_NEAR(projected->x, ground_square[i].x, 1e-3f);
        EXPECT_NEAR(projected->y, ground_square[i].y, 1e-3f);
    }

    // Define a calibrated ground-plane HazardZone covering the metric square [0, 10] x [0, 10]
    const HazardZone metric_zone{
        "CalibratedPressZone",
        {{0.0f, 0.0f}, {10.0f, 0.0f}, {10.0f, 10.0f}, {0.0f, 10.0f}},
        /*require_helmet=*/true,
        /*require_vest=*/false,
    };

    SafetyAnalyzer analyzer;
    analyzer.set_homography(H);
    EXPECT_TRUE(analyzer.has_homography());

    // Worker whose feet_point is at (500, 400) in image pixels -> maps into the metric square
    const Detection worker_inside{cv::Rect2f(460.0f, 200.0f, 80.0f, 200.0f), 0.95f,
                                  static_cast<int>(ClassId::Person)};
    // Worker whose feet_point is at (100, 400) in image pixels -> maps outside the metric square
    const Detection worker_outside{cv::Rect2f(60.0f, 200.0f, 80.0f, 200.0f), 0.95f,
                                   static_cast<int>(ClassId::Person)};

    const std::vector<WorkerStatus> calibrated_statuses =
        analyzer.analyze_frame({worker_inside, worker_outside}, {metric_zone});
    ASSERT_EQ(calibrated_statuses.size(), 2U);

    EXPECT_FLOAT_EQ(calibrated_statuses[0].feet_point.x, 500.0f);
    EXPECT_FLOAT_EQ(calibrated_statuses[0].feet_point.y, 400.0f);
    EXPECT_TRUE(calibrated_statuses[0].ground_projection_valid);
    ASSERT_TRUE(calibrated_statuses[0].ground_point.has_value());
    EXPECT_NEAR(calibrated_statuses[0].ground_point->x, 5.0f, 1e-3f);
    EXPECT_GT(calibrated_statuses[0].ground_point->y, 0.0f);
    EXPECT_LT(calibrated_statuses[0].ground_point->y, 10.0f);
    EXPECT_TRUE(calibrated_statuses[0].in_hazard_zone);
    EXPECT_TRUE(calibrated_statuses[0].is_violating);

    EXPECT_TRUE(calibrated_statuses[1].ground_projection_valid);
    ASSERT_TRUE(calibrated_statuses[1].ground_point.has_value());
    EXPECT_LT(calibrated_statuses[1].ground_point->x, 0.0f);
    EXPECT_FALSE(calibrated_statuses[1].in_hazard_zone);
    EXPECT_FALSE(calibrated_statuses[1].is_violating);

    // Clearing homography restores uncalibrated pixel-space behavior (ground_point == feet_point)
    analyzer.clear_homography();
    EXPECT_FALSE(analyzer.has_homography());
    const std::vector<WorkerStatus> uncalibrated_statuses =
        analyzer.analyze_frame({worker_inside}, {metric_zone});
    ASSERT_EQ(uncalibrated_statuses.size(), 1U);
    EXPECT_TRUE(uncalibrated_statuses[0].ground_projection_valid);
    ASSERT_TRUE(uncalibrated_statuses[0].ground_point.has_value());
    EXPECT_EQ(*uncalibrated_statuses[0].ground_point, uncalibrated_statuses[0].feet_point);
}

TEST(GeometryTest, DegenerateHomographyAndHorizonPointHandling) {
    SafetyAnalyzer analyzer;

    // 1. Singular zero matrix must be rejected by set_homography
    const cv::Matx33f zero_H = cv::Matx33f::zeros();
    EXPECT_THROW(analyzer.set_homography(zero_H), std::invalid_argument);

    // 2. Non-finite matrix must be rejected by set_homography
    cv::Matx33f nan_H = cv::Matx33f::eye();
    nan_H(0, 0) = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(analyzer.set_homography(nan_H), std::invalid_argument);

    // 3. Invertible homography with a horizon line at y = 100 (w = 0.01 * y - 1.0 == 0 at y = 100)
    const cv::Matx33f horizon_H(1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.01f, -1.0f);
    EXPECT_NO_THROW(analyzer.set_homography(horizon_H));

    // Point on the horizon (w -> 0) must return std::nullopt, never raw pixel coordinates
    const cv::Point2f horizon_pt(50.0f, 100.0f);
    const std::optional<cv::Point2f> projected =
        safety_engine::project_point_homography(horizon_pt, horizon_H);
    EXPECT_FALSE(projected.has_value());

    // Define a ground-plane HazardZone that deliberately covers raw pixel coordinates (50, 100).
    // Because homography projection fails at y = 100, analyze_frame must mark
    // ground_projection_valid = false and skip hazard-zone evaluation rather than matching (50, 100).
    const HazardZone ground_zone{
        "MetricGroundZone",
        {{0.0f, 0.0f}, {200.0f, 0.0f}, {200.0f, 200.0f}, {0.0f, 200.0f}},
        /*require_helmet=*/true,
        /*require_vest=*/true,
    };
    const Detection horizon_worker{cv::Rect2f(10.0f, 0.0f, 80.0f, 100.0f), 0.95f,
                                   static_cast<int>(ClassId::Person)};

    const std::vector<WorkerStatus> statuses =
        analyzer.analyze_frame({horizon_worker}, {ground_zone});
    ASSERT_EQ(statuses.size(), 1U);
    EXPECT_FLOAT_EQ(statuses[0].feet_point.x, 50.0f);
    EXPECT_FLOAT_EQ(statuses[0].feet_point.y, 100.0f);
    EXPECT_FALSE(statuses[0].ground_projection_valid);
    EXPECT_FALSE(statuses[0].ground_point.has_value());
    EXPECT_FALSE(statuses[0].in_hazard_zone);
    EXPECT_FALSE(statuses[0].is_violating);
}
