#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <string>

#include "safety_engine/version.hpp"

TEST(SmokeTest, EngineVersionNotEmpty) {
    const std::string version_info = safety_engine::get_engine_info();
    EXPECT_FALSE(version_info.empty());

    const cv::Mat frame = cv::Mat::zeros(640, 640, CV_8UC3);
    EXPECT_FALSE(frame.empty());
    EXPECT_EQ(frame.rows, 640);
    EXPECT_EQ(frame.cols, 640);
    EXPECT_EQ(frame.channels(), 3);
    EXPECT_EQ(frame.type(), CV_8UC3);
}
