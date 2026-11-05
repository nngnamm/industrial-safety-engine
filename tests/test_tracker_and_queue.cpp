#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <opencv2/core/types.hpp>
#include <optional>
#include <thread>
#include <vector>

#include "safety_engine/frame_queue.hpp"
#include "safety_engine/tracker.hpp"
#include "safety_engine/types.hpp"

namespace {

using safety_engine::AlertState;
using safety_engine::BoundedQueue;
using safety_engine::TrackedWorker;
using safety_engine::WorkerStatus;
using safety_engine::WorkerTracker;

WorkerStatus make_worker(const cv::Rect2f& bbox, bool is_violating, bool has_helmet = false,
                         bool has_vest = true) {
    WorkerStatus w;
    w.person_bbox = bbox;
    w.person_confidence = 0.92f;
    w.feet_point = cv::Point2f(bbox.x + 0.5f * bbox.width, bbox.y + bbox.height);
    w.has_helmet = has_helmet;
    w.has_vest = has_vest;
    w.in_hazard_zone = true;
    w.is_violating = is_violating;
    if (is_violating) {
        w.violated_zones.push_back("PressZoneA");
    }
    return w;
}

}  // namespace

TEST(TrackerTest, DebouncingStateMachine) {
    WorkerTracker tracker(0.3f, /*alert_streak_threshold=*/3, /*max_missed_frames=*/2);

    const cv::Rect2f base_bbox(100.0f, 100.0f, 80.0f, 220.0f);
    const WorkerStatus violating_worker = make_worker(base_bbox, /*is_violating=*/true);

    // Frame 1: SuspectedViolation, streak = 1
    const std::vector<TrackedWorker> f1 = tracker.update({violating_worker});
    ASSERT_EQ(f1.size(), 1U);
    const int initial_track_id = f1[0].track_id;
    EXPECT_EQ(f1[0].age_frames, 1);
    EXPECT_EQ(f1[0].missed_frames, 0);
    EXPECT_EQ(f1[0].violation_streak, 1);
    EXPECT_EQ(f1[0].alert_state, AlertState::SuspectedViolation);

    // Frame 2: SuspectedViolation, streak = 2 (slight natural movement)
    const WorkerStatus f2_worker =
        make_worker(cv::Rect2f(104.0f, 101.0f, 80.0f, 220.0f), /*is_violating=*/true);
    const std::vector<TrackedWorker> f2 = tracker.update({f2_worker});
    ASSERT_EQ(f2.size(), 1U);
    EXPECT_EQ(f2[0].track_id, initial_track_id);
    EXPECT_EQ(f2[0].age_frames, 2);
    EXPECT_EQ(f2[0].missed_frames, 0);
    EXPECT_EQ(f2[0].violation_streak, 2);
    EXPECT_EQ(f2[0].alert_state, AlertState::SuspectedViolation);

    // Frame 3: ActiveAlert, streak = 3
    const WorkerStatus f3_worker =
        make_worker(cv::Rect2f(108.0f, 102.0f, 80.0f, 220.0f), /*is_violating=*/true);
    const std::vector<TrackedWorker> f3 = tracker.update({f3_worker});
    ASSERT_EQ(f3.size(), 1U);
    EXPECT_EQ(f3[0].track_id, initial_track_id);
    EXPECT_EQ(f3[0].age_frames, 3);
    EXPECT_EQ(f3[0].missed_frames, 0);
    EXPECT_EQ(f3[0].violation_streak, 3);
    EXPECT_EQ(f3[0].alert_state, AlertState::ActiveAlert);

    // Frame 4: Compliant worker -> immediate reset to Safe and streak = 0 with same track_id
    const WorkerStatus compliant_worker = make_worker(
        cv::Rect2f(110.0f, 102.0f, 80.0f, 220.0f), /*is_violating=*/false, /*has_helmet=*/true);
    const std::vector<TrackedWorker> f4 = tracker.update({compliant_worker});
    ASSERT_EQ(f4.size(), 1U);
    EXPECT_EQ(f4[0].track_id, initial_track_id);
    EXPECT_EQ(f4[0].age_frames, 4);
    EXPECT_EQ(f4[0].missed_frames, 0);
    EXPECT_EQ(f4[0].violation_streak, 0);
    EXPECT_EQ(f4[0].alert_state, AlertState::Safe);
    EXPECT_TRUE(f4[0].status.has_helmet);
    EXPECT_FALSE(f4[0].status.is_violating);
}

TEST(TrackerTest, OcclusionPersistence) {
    WorkerTracker tracker(0.3f, /*alert_streak_threshold=*/3, /*max_missed_frames=*/2);

    const WorkerStatus violating_worker =
        make_worker(cv::Rect2f(200.0f, 150.0f, 90.0f, 240.0f), /*is_violating=*/true);

    // Frames 1..3: Reach ActiveAlert on Frame 3
    const std::vector<TrackedWorker> f1 = tracker.update({violating_worker});
    ASSERT_EQ(f1.size(), 1U);
    const int expected_id = f1[0].track_id;

    tracker.update({violating_worker});
    const std::vector<TrackedWorker> f3 = tracker.update({violating_worker});
    ASSERT_EQ(f3.size(), 1U);
    EXPECT_EQ(f3[0].track_id, expected_id);
    EXPECT_EQ(f3[0].violation_streak, 3);
    EXPECT_EQ(f3[0].alert_state, AlertState::ActiveAlert);

    // Frame 4: Detector dropout (empty detections) -> track survives with missed_frames == 1
    const std::vector<TrackedWorker> f4 = tracker.update({});
    ASSERT_EQ(f4.size(), 1U);
    EXPECT_EQ(f4[0].track_id, expected_id);
    EXPECT_EQ(f4[0].missed_frames, 1);
    EXPECT_EQ(f4[0].violation_streak, 3);
    EXPECT_EQ(f4[0].alert_state, AlertState::ActiveAlert);

    // Frame 5: Worker reappears -> retains exact same track_id and continues streak = 4
    const std::vector<TrackedWorker> f5 = tracker.update({violating_worker});
    ASSERT_EQ(f5.size(), 1U);
    EXPECT_EQ(f5[0].track_id, expected_id);
    EXPECT_EQ(f5[0].missed_frames, 0);
    EXPECT_EQ(f5[0].age_frames, 4);
    EXPECT_EQ(f5[0].violation_streak, 4);
    EXPECT_EQ(f5[0].alert_state, AlertState::ActiveAlert);

    // Frames 6..8: Exceed max_missed_frames (2) -> pruned on 3rd consecutive miss
    EXPECT_EQ(tracker.update({}).size(), 1U);  // missed_frames = 1
    EXPECT_EQ(tracker.update({}).size(), 1U);  // missed_frames = 2
    EXPECT_TRUE(tracker.update({}).empty());   // missed_frames = 3 > 2 -> removed
}

TEST(TrackerTest, MultiWorkerGreedyOneToOneMatching) {
    WorkerTracker tracker(0.3f, /*alert_streak_threshold=*/2, /*max_missed_frames=*/2);

    const WorkerStatus worker_left =
        make_worker(cv::Rect2f(50.0f, 100.0f, 80.0f, 200.0f), /*is_violating=*/true);
    const WorkerStatus worker_right =
        make_worker(cv::Rect2f(400.0f, 100.0f, 80.0f, 200.0f), /*is_violating=*/false, true);

    const std::vector<TrackedWorker> f1 = tracker.update({worker_left, worker_right});
    ASSERT_EQ(f1.size(), 2U);
    const int left_id = f1[0].track_id;
    const int right_id = f1[1].track_id;
    EXPECT_LT(left_id, right_id);

    // Swap input order in Frame 2 and add a third newcomer in the middle
    const WorkerStatus worker_left_f2 =
        make_worker(cv::Rect2f(55.0f, 102.0f, 80.0f, 200.0f), /*is_violating=*/true);
    const WorkerStatus worker_right_f2 =
        make_worker(cv::Rect2f(395.0f, 98.0f, 80.0f, 200.0f), /*is_violating=*/false, true);
    const WorkerStatus newcomer =
        make_worker(cv::Rect2f(220.0f, 100.0f, 80.0f, 200.0f), /*is_violating=*/true);

    const std::vector<TrackedWorker> f2 =
        tracker.update({worker_right_f2, newcomer, worker_left_f2});
    ASSERT_EQ(f2.size(), 3U);

    // Returned tracks are deterministically sorted by track_id
    EXPECT_EQ(f2[0].track_id, left_id);
    EXPECT_EQ(f2[0].violation_streak, 2);
    EXPECT_EQ(f2[0].alert_state, AlertState::ActiveAlert);

    EXPECT_EQ(f2[1].track_id, right_id);
    EXPECT_EQ(f2[1].violation_streak, 0);
    EXPECT_EQ(f2[1].alert_state, AlertState::Safe);

    EXPECT_GT(f2[2].track_id, right_id);
    EXPECT_EQ(f2[2].age_frames, 1);
    EXPECT_EQ(f2[2].violation_streak, 1);
    EXPECT_EQ(f2[2].alert_state, AlertState::SuspectedViolation);
}

TEST(BoundedQueueTest, MultiThreadedProducerConsumer) {
    constexpr int kTotalItems = 100;
    BoundedQueue<int> queue(5);

    std::vector<int> consumed;
    consumed.reserve(kTotalItems);

    std::thread producer([&queue]() {
        for (int i = 0; i < kTotalItems; ++i) {
            EXPECT_TRUE(queue.push(i));
        }
        queue.close();
    });

    std::thread consumer([&queue, &consumed]() {
        while (true) {
            std::optional<int> item = queue.pop();
            if (!item.has_value()) {
                break;
            }
            consumed.push_back(*item);
        }
    });

    producer.join();
    consumer.join();

    EXPECT_TRUE(queue.is_closed());
    EXPECT_EQ(queue.size(), 0U);
    ASSERT_EQ(consumed.size(), static_cast<std::size_t>(kTotalItems));
    for (int i = 0; i < kTotalItems; ++i) {
        EXPECT_EQ(consumed[static_cast<std::size_t>(i)], i);
    }
}

TEST(BoundedQueueTest, DropOldestRealTimeMode) {
    BoundedQueue<int> queue(3);

    for (int i = 0; i < 5; ++i) {
        queue.push_drop_oldest(i);
    }

    EXPECT_EQ(queue.size(), 3U);

    const std::optional<int> first = queue.pop();
    const std::optional<int> second = queue.pop();
    const std::optional<int> third = queue.pop();

    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    ASSERT_TRUE(third.has_value());
    EXPECT_EQ(*first, 2);
    EXPECT_EQ(*second, 3);
    EXPECT_EQ(*third, 4);
    EXPECT_EQ(queue.size(), 0U);

    queue.close();
    EXPECT_FALSE(queue.pop().has_value());
}

TEST(BoundedQueueTest, CloseUnblocksBlockedProducerAndSupportsMoveOnlyTypes) {
    BoundedQueue<std::unique_ptr<int>> queue(1);
    ASSERT_TRUE(queue.push(std::make_unique<int>(42)));

    std::atomic<bool> producer_entered{false};
    std::atomic<bool> second_push_result{true};

    std::thread blocked_producer([&]() {
        producer_entered.store(true);
        second_push_result.store(queue.push(std::make_unique<int>(99)));
    });

    while (!producer_entered.load()) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    queue.close();
    blocked_producer.join();

    EXPECT_FALSE(second_push_result.load());

    std::optional<std::unique_ptr<int>> remaining = queue.pop();
    ASSERT_TRUE(remaining.has_value());
    ASSERT_NE(*remaining, nullptr);
    EXPECT_EQ(**remaining, 42);
    EXPECT_FALSE(queue.pop().has_value());
}
