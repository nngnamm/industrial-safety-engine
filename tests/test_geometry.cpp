#include <gtest/gtest.h>

#include <opencv2/core/types.hpp>
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
