#ifndef YOLO26_CPM_H
#define YOLO26_CPM_H

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace cpm {

// 将多个调用者提交的单张图片自动合并成 N 张批量推理。
template <typename Result, typename Input, typename Model> class Instance {
public:
  using Future = std::shared_future<Result>;

  Instance() = default;
  ~Instance() { stop(); }

  Instance(const Instance &) = delete;
  Instance &operator=(const Instance &) = delete;

  template <typename LoadMethod>
  bool start(LoadMethod load_method, std::size_t max_batch_size) {
    if (max_batch_size == 0) {
      throw std::invalid_argument("CPM max batch size must be greater than 0");
    }
    stop();
    max_batch_size_ = max_batch_size;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      starting_ = true;
    }

    std::promise<bool> ready;
    std::future<bool> status = ready.get_future();
    worker_ = std::thread([this, load_method = std::move(load_method),
                           ready = std::move(ready)]() mutable {
      worker(std::move(load_method), std::move(ready));
    });
    return status.get();
  }

  Future commit(const Input &input) {
    Item item{input, std::make_shared<std::promise<Result>>()};
    Future future = item.promise->get_future().share();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!running_) {
        throw std::runtime_error("CPM is not running");
      }
      queue_.push_back(std::move(item));
    }
    condition_.notify_one();
    return future;
  }

  std::vector<Future> commits(const std::vector<Input> &inputs) {
    std::vector<Future> futures;
    futures.reserve(inputs.size());
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!running_) {
        throw std::runtime_error("CPM is not running");
      }
      for (const Input &input : inputs) {
        Item item{input, std::make_shared<std::promise<Result>>()};
        futures.push_back(item.promise->get_future().share());
        queue_.push_back(std::move(item));
      }
    }
    condition_.notify_one();
    return futures;
  }

  void stop() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      running_ = false;
      starting_ = false;
    }
    condition_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
    try {
      fail_pending(std::make_exception_ptr(
          std::runtime_error("CPM stopped before inference completed")));
    } catch (...) {
      fail_pending(std::current_exception());
    }
  }

private:
  struct Item {
    Input input;
    std::shared_ptr<std::promise<Result>> promise;
  };

  template <typename LoadMethod>
  void worker(LoadMethod load_method, std::promise<bool> ready) noexcept {
    std::shared_ptr<Model> model;
    try {
      model = load_method();
      if (!model) {
        throw std::runtime_error("cannot create YOLO26 detector");
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!starting_) {
          throw std::runtime_error("CPM was stopped while starting");
        }
        starting_ = false;
        running_ = true;
      }
      ready.set_value(true);
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        starting_ = false;
      }
      ready.set_exception(std::current_exception());
      return;
    }

    while (true) {
      std::vector<Item> items;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !running_ || !queue_.empty(); });
        if (!running_ && queue_.empty()) {
          break;
        }
        // 给同一时刻到达的请求一个很短的合批窗口。
        if (running_ && queue_.size() < max_batch_size_) {
          condition_.wait_for(lock, batch_wait_, [this] {
            return !running_ || queue_.size() >= max_batch_size_;
          });
        }
        const std::size_t count = std::min(max_batch_size_, queue_.size());
        items.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
          items.push_back(std::move(queue_.front()));
          queue_.pop_front();
        }
      }

      try {
        std::vector<Input> inputs;
        inputs.reserve(items.size());
        for (const Item &item : items) {
          inputs.push_back(item.input);
        }
        std::vector<Result> results = model->predict(inputs);
        if (results.size() != items.size()) {
          throw std::runtime_error("YOLO26 returned an invalid batch size");
        }
        for (std::size_t i = 0; i < items.size(); ++i) {
          items[i].promise->set_value(std::move(results[i]));
        }
      } catch (...) {
        const std::exception_ptr error = std::current_exception();
        for (Item &item : items) {
          item.promise->set_exception(error);
        }
      }
    }
  }

  void fail_pending(const std::exception_ptr &error) noexcept {
    std::deque<Item> pending;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending.swap(queue_);
    }
    while (!pending.empty()) {
      try {
        pending.front().promise->set_exception(error);
      } catch (...) {
      }
      pending.pop_front();
    }
  }

  std::condition_variable condition_;
  std::mutex mutex_;
  std::deque<Item> queue_;
  std::thread worker_;
  bool running_ = false;
  bool starting_ = false;
  std::size_t max_batch_size_ = 1;
  const std::chrono::milliseconds batch_wait_{2};
};

} // namespace cpm

#endif // YOLO26_CPM_H
