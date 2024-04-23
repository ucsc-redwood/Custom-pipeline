#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <thread>

class ThreadPool {
public:
    ThreadPool(size_t numThreads) : stop(false), thread_count_(numThreads), tasks_remaining_(0) {
	std::cout << "Number of processor cores available: " << std::thread::hardware_concurrency() << std::endl;
        std::cout << "ThreadPool created with " << numThreads << " threads." << std::endl;
        start();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        cond_.notify_all();
        for (std::thread &worker : workers) {
            worker.join();
        }
        std::cout << "ThreadPool destroyed, all threads joined." << std::endl;
    }

    template<class F, class... Args>
    void enqueue(F&& f, Args&&... args) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace([=]() { f(args...); });
            tasks_remaining_++;
            std::cout << "Task enqueued, total tasks remaining: " << tasks_remaining_ << std::endl;
        }
        cond_.notify_one();
    }

    void wait_complete() {
        std::unique_lock<std::mutex> lock(queueMutex);
        all_done_.wait(lock, [this]() { return tasks_remaining_ == 0; });
        std::cout << "All tasks completed." << std::endl;
    }

private:
    void start() {
        for (size_t i = 0; i < thread_count_; ++i) {
            workers.emplace_back([this, i] {
                std::cout << "Thread " << i << " started." << std::endl;
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        cond_.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) {
                            std::cout << "Thread stopping as no tasks left and stop flag is true." << std::endl;
                            return;
                        }
                        task = std::move(tasks.front());
                        tasks.pop();
                        std::cout << "Thread picked a task, remaining tasks: " << tasks.size() << std::endl;
                    }
                    auto start_time = std::chrono::high_resolution_clock::now();
                    task();
                    auto end_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> task_duration = end_time - start_time;
                    std::cout << "Task executed by thread, duration: " << task_duration.count() << " ms" << std::endl;
                    {
                        std::lock_guard<std::mutex> lock(queueMutex);
                        tasks_remaining_--;
                        if (tasks_remaining_ == 0) all_done_.notify_one();
                    }
                }
            });
        }
    }

    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable cond_;
    std::condition_variable all_done_;
    bool stop;
    size_t thread_count_;
    int tasks_remaining_;
};

