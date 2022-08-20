
/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>

/**
 * Single task multiple data
 */
class STMD {
    typedef std::uint_fast32_t u32;
    typedef std::uint_fast64_t u64;
private:

    typedef enum {PREPARE = 1, BUSY = 2, IDLE = 2} status_t;

    // std::mutex mutex;
    // std::condition_variable cv;

    u32 nthreads;
    std::unique_ptr<std::thread[]> threads;
    std::queue<std::function<void(u32)>> tasks = {};
    volatile int total_tasks = 0;

    // start, end, per
    std::queue<std::tuple<u32, u32, u32>> configs = {};
    u32 ntasks = 0;
    volatile bool running = true;
    std::atomic<int> fin_cnt = 0;
    std::atomic<int> start_cnt = 0;
    volatile status_t status = BUSY;
public:

    STMD(const u32 &_nthreads = std::thread::hardware_concurrency())
        : nthreads(_nthreads), threads(new std::thread[_nthreads]) {
        int nhardware_threads = std::thread::hardware_concurrency();
        int idx_per_hardware_threads = nhardware_threads / nthreads;
        for (u32 i = 0;i < nthreads;i++) {
            threads[i] = std::thread(&STMD::worker, this, i);
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);
            int rc = pthread_setaffinity_np(threads[i].native_handle(),
                                            sizeof(cpu_set_t), &cpuset);
            if (rc != 0) {
                std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
            }
        }
    }
    ~STMD() {
        running = false;
        status = PREPARE;
        for (u32 i = 0; i < nthreads; i++)
        {
            threads[i].join();
        }
    }


    void submit_task(const std::function<void(u32)> &task, u32 start, u32 end, u32 per) {
        tasks.push(task);
        configs.push(std::make_tuple(start, end, per));
        fin_cnt = 0;
        start_cnt = 0;
        status = PREPARE;
        total_tasks += 1;
    }

    void fin_task() {
        // wait for all to finish
        while (start_cnt < nthreads);
        status = BUSY;
        while (fin_cnt < nthreads);
        status = IDLE;
        tasks.pop();
        configs.pop();
        total_tasks -= 1;
    }

    void parallelize_loop(const u32 &first_index, const u32 &index_after_last, const std::function<void(u32)> &loop, u32 min_block = 0) {
        u32 idx_per_thread = ((index_after_last - first_index) + (nthreads - 1)) / nthreads;
        if (min_block) {
            idx_per_thread = (idx_per_thread < min_block)? min_block : idx_per_thread;
        }
        submit_task(loop, first_index, index_after_last, idx_per_thread);
        fin_task();
    }

    void worker(int tidx) {
        while (running)
        {
            while (status != PREPARE) {}
            // printf("start task\n");
            if (!total_tasks) continue;
            auto task = tasks.front();
            auto config = configs.front();
            auto start = std::get<0>(config);
            auto end = std::get<1>(config);
            auto per = std::get<2>(config);
            start_cnt += 1;
            for (u32 i = start + per * tidx; 
                i < end && i < start + per * (tidx + 1); i++ ) {
                task(i);
            }
            while (status == PREPARE) {}
            fin_cnt += 1;
        }
    }
};