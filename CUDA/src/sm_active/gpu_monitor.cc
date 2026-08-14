#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include <nvml.h>
#include <pthread.h>

#include "gpu_monitor.h"

namespace gpu {
namespace monitor {

class GpuMonitorImpl {
public:
  GpuMonitorImpl() = default;

  ~GpuMonitorImpl() {
    exit_signal_.notify_one();
    running_ = false;
    if (monitor_thread_) {
      monitor_thread_->join();
      delete monitor_thread_;
      monitor_thread_ = nullptr;
    }

    nvmlGpmSampleFree(sample1_);
    nvmlGpmSampleFree(sample2_);
    nvmlShutdown();
  }

  // start background monitor thread
  void start(int dev_idx) {
    dev_idx_ = dev_idx;

    // Initialize NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
      std::cerr << "NVML Init Failed: " << nvmlErrorString(result) << "\n";
      std::exit(EXIT_FAILURE);
    }

    // Get Device Handle
    nvmlDeviceGetHandleByIndex(dev_idx, &device_);

    // Allocate tracking sample buffers
    nvmlGpmSampleAlloc(&sample1_);
    nvmlGpmSampleAlloc(&sample2_);

    running_ = true;
    monitor_thread_ = new std::thread(&GpuMonitorImpl::monitor, this);
  }

private:
  int dev_idx_ = 0;
  nvmlDevice_t device_;
  nvmlGpmSample_t sample1_;
  nvmlGpmSample_t sample2_;

  // necessary elements for monitoring thread
  volatile bool running_;
  std::mutex exit_mutex_;
  std::condition_variable exit_signal_;
  std::thread *monitor_thread_ = nullptr;

  void query_sm_active() {
    // Capture the first hardware snapshot
    nvmlGpmSampleGet(device_, sample1_);

    // Wait for a valid calculation period (minimum 100ms required)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Capture the second hardware snapshot
    nvmlGpmSampleGet(device_, sample2_);

    // Configure and invoke the metrics request
    nvmlGpmMetricsGet_t metricsGet;
    metricsGet.version = NVML_GPM_METRICS_GET_VERSION;
    metricsGet.numMetrics = 2;
    metricsGet.sample1 = sample1_;
    metricsGet.sample2 = sample2_;

    // Percentage of SMs that were busy. 0.0 - 100.0
    metricsGet.metrics[0].metricId = NVML_GPM_METRIC_SM_UTIL;

    // Percentage of warps that were active vs theoretical maximum. 0.0 - 100.0
    metricsGet.metrics[1].metricId = NVML_GPM_METRIC_SM_OCCUPANCY;

    auto result = nvmlGpmMetricsGet(&metricsGet);
    if (result == NVML_SUCCESS) {
      // Output the percentage value (0.0 to 100.0)
      std::cout << "SM Utilization: " << metricsGet.metrics[0].value
                << "%\n";
      std::cout << "SM Active Warps: " << metricsGet.metrics[1].value
                << "%\n";
      std::cout << "\n";
    } else {
      std::cerr << "Metrics calculation failed: " << nvmlErrorString(result)
                << "\n";
    }
  }

  // monitor loop
  void monitor() {
    const int SAMPLING_INTERVAL = 1; // in Second
    const int QUERY_INTERVAL = 3;   // in Second
    pthread_setname_np(pthread_self(), "GPU Monitor");
    std::string start_str = std::string("\nStart GPU Monitor for device ") + std::to_string(dev_idx_) + "\n";
    std::cout << start_str;

    using namespace std::chrono;
    // align to each 30 seconds
    time_t next_query = system_clock::to_time_t(system_clock::now()) /
                        QUERY_INTERVAL * QUERY_INTERVAL;
    while (true) {
      std::unique_lock<std::mutex> exit_lock(exit_mutex_);
      if (!running_ ||
          exit_signal_.wait_for(exit_lock,
                                std::chrono::seconds(SAMPLING_INTERVAL)) !=
              std::cv_status::timeout) {
        std::cout << "Exit GPU Monitor\n";
        return;
      }
      time_t now = system_clock::to_time_t(system_clock::now());

      // update query value every 30 seconds
      if (now > next_query) {
        query_sm_active();
        next_query += QUERY_INTERVAL;
      }
    }
  }
};

GpuMonitor::GpuMonitor() { impl = std::make_unique<GpuMonitorImpl>(); }

GpuMonitor::~GpuMonitor() {}

void GpuMonitor::start(int dev_idx) { impl->start(dev_idx); }

} // namespace monitor
} // namespace gpu
