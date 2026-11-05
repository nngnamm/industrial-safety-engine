#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <utility>

namespace safety_engine {

template <typename T>
class BoundedQueue {
   public:
    explicit BoundedQueue(std::size_t max_capacity) : max_capacity_(max_capacity) {
        if (max_capacity_ == 0U) {
            throw std::invalid_argument("BoundedQueue: max_capacity must be greater than 0.");
        }
    }

    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;

    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_full_.wait(lock, [this]() { return queue_.size() < max_capacity_ || closed_; });
        if (closed_) {
            return false;
        }
        queue_.push(std::move(item));
        lock.unlock();
        cv_not_empty_.notify_one();
        return true;
    }

    void push_drop_oldest(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        if (queue_.size() >= max_capacity_) {
            queue_.pop();
        }
        queue_.push(std::move(item));
        lock.unlock();
        cv_not_empty_.notify_one();
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_empty_.wait(lock, [this]() { return !queue_.empty() || closed_; });
        if (queue_.empty() && closed_) {
            return std::nullopt;
        }
        T item = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        cv_not_full_.notify_one();
        return item;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool is_closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

    std::size_t capacity() const noexcept { return max_capacity_; }

   private:
    std::size_t max_capacity_;
    std::queue<T> queue_;
    bool closed_{false};
    mutable std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
};

}  // namespace safety_engine
