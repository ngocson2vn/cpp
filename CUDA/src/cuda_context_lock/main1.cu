#include <random>
#include <thread>
#include <vector>
#include <functional>
#include <atomic>

#include <immintrin.h>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(apiCall)                                              \
  do {                                                                         \
    cudaError_t error = apiCall;                                               \
    if (error != cudaSuccess) {                                                \
      auto errorName = cudaGetErrorName(error);                                \
      auto errorString = cudaGetErrorString(error);                            \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, errorName,         \
              errorString);                                                    \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

#define CHECK_KERNEL_LAUNCH()                                                  \
  do {                                                                         \
    auto lastError = cudaGetLastError();                                       \
    if (lastError != cudaSuccess) {                                            \
      auto errorName = cudaGetErrorName(lastError);                            \
      auto errorString = cudaGetErrorString(lastError);                        \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, errorName,         \
              errorString);                                                    \
    }                                                                          \
  } while (0);

extern "C" {

__global__ void gelu(const float *__restrict__ input,
                     float *__restrict__ output, int length) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < length) {
    float elem = input[tid];
    float res = 0.5f * elem * (1.0f + erff(elem * 0.7071067811f));
    output[tid] = res;
  }
}
}

class Worker {
 public:
  Worker() : id_(++Worker::counter_) {

  };

  void schedule(std::function<void()> job) {
    jobs_.push_back(job);
  }

  void start() {
    worker_ = std::thread(&Worker::run, this);
  }

  void stop() {
    while (!jobs_.empty()) {
      __pause();
    }

    // Worker should complete all jobs
    keep_running_ = false;
    worker_.join();
  }

 private:
  static std::atomic<int64_t> counter_;
  int64_t id_;
  bool keep_running_ = true;
  std::thread worker_;
  std::vector<std::function<void()>> jobs_;

  void run() {
    std::size_t done_jobs = 0;
    while(keep_running_) {
      if (!jobs_.empty()) {
        for (auto& job : jobs_) {
          job();
        }

        done_jobs += jobs_.size();
        jobs_.clear();
      } else {
        __pause();
      }
    }

    printf("Worker %lu completed %zu jobs\n", id_, done_jobs);
  }
};

std::atomic<int64_t> Worker::counter_(-1);

int main(int argc, char **argv) {
  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  int length = 1024;
  std::vector<float> x(length, 0.0f);
  for (int i = 0; i < length; i++) {
    x[i] = dist(gen);
  }

  float *dev_inp_ptr = nullptr;
  cudaMalloc(&dev_inp_ptr, length * sizeof(float));
  cudaMemcpy(dev_inp_ptr, x.data(), length * sizeof(float),
             cudaMemcpyHostToDevice);

  auto threads = 512;
  auto blocks = (length + threads - 1) / threads;

  constexpr uint32_t kNumLaunches = 4000;
  std::vector<float *> dev_out_ptrs(kNumLaunches, nullptr);
  float *dev_out_ptr = nullptr;
  for (uint32_t i = 0; i < kNumLaunches; i++) {
    CHECK_CUDA_ERROR(cudaMalloc(&dev_out_ptr, length * sizeof(float)));
    dev_out_ptrs[i] = dev_out_ptr;
  }

  constexpr uint32_t kNumWorkers = 1;
  constexpr uint32_t kWorkerJobs = kNumLaunches / kNumWorkers;
  std::vector<Worker> workers(kNumWorkers);
  uint32_t k = 0;
  for (uint32_t idx = 0; idx < kNumLaunches; idx++) {
    k = idx / kWorkerJobs;
    workers[k].schedule([&, idx]() {
      gelu<<<blocks, threads>>>(dev_inp_ptr, dev_out_ptrs[idx], length);
      CHECK_KERNEL_LAUNCH();
      // printf("Launched gelu %d\n", idx);
    });
  }

  for (auto &worker : workers) {
    worker.start();
  }

  for (auto &worker : workers) {
    worker.stop();
  }

  cudaDeviceSynchronize();

  printf("DONE\n");
}
