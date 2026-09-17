#include <atomic>
#include <cassert>
#include <functional>
#include <random>
#include <thread>
#include <vector>

#include <cuda.h>
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

#define CHECK_KERNEL_LAUNCH(...)                                               \
  do {                                                                         \
    __VA_ARGS__;                                                               \
    auto lastError = cudaGetLastError();                                       \
    if (lastError != cudaSuccess) {                                            \
      auto errorName = cudaGetErrorName(lastError);                            \
      auto errorString = cudaGetErrorString(lastError);                        \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, errorName,         \
              errorString);                                                    \
    }                                                                          \
  } while (0);

static std::vector<CUcontext> &getContexts() {
  static std::vector<CUcontext> contexts;
  return contexts;
}

static CUcontext getContext(uint32_t idx) { return getContexts()[idx]; }

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

class ScopedContext {
public:
  ScopedContext(CUcontext ctx) : ctx_(ctx) { cuCtxPushCurrent(ctx); }

  ~ScopedContext() {
    CUcontext pctx;
    cuCtxPopCurrent(&pctx);
    assert(pctx == ctx_ && "ctx mismatch!");
  }

private:
  CUcontext ctx_;
};

class Worker {
public:
  Worker()
      : id_(++Worker::counter_){

        };

  void schedule(std::function<void()> job) { jobs_.push_back(job); }

  void start() { worker_ = std::thread(&Worker::run, this); }

  void stop() {
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
    auto ctx = getContext(id_);
    ScopedContext sc(ctx);
    printf("Worker %lu uses ctx %p\n", id_, ctx);

    std::size_t done_jobs = 0;
    while (keep_running_) {
      for (auto &job : jobs_) {
        job();
      }

      done_jobs += jobs_.size();
      jobs_.clear();
    }

    printf("Worker %lu completed %zu jobs\n", id_, done_jobs);
  }
};

std::atomic<int64_t> Worker::counter_(-1);

int main(int argc, char **argv) {
  //===============================================================================
  // Prologue
  //===============================================================================
  // Initialize CUDA
  cuInit(0);
  CUdevice device;
  cuDeviceGet(&device, 0);

  // Retain primary context
  CUcontext primary_ctx;
  cuDevicePrimaryCtxRetain(&primary_ctx, device);

  // PUSH the Primary Context onto the main thread's stack.
  // This makes it the active context for all subsequent Runtime API calls.
  cuCtxPushCurrent(primary_ctx);

  // Create a pool of contexts for worker threads
  constexpr uint32_t kNumWorkers = 4;
  auto &contexts = getContexts();
  for (int i = 0; i < kNumWorkers; i++) {
    CUcontext ctx;

    // cuCtxCreate creates the context AND implicitly pushes it to the
    // top of this thread's stack, burying the primary context temporarily.
    cuCtxCreate(&ctx, nullptr, 0, device);

    // IMMEDIATELY POP the new custom context off the stack.
    // This restores the primary context as the active context for the main
    // thread.
    CUcontext pctx;
    cuCtxPopCurrent(&pctx);

    contexts.push_back(ctx);
  }

  //===============================================================================
  // Body
  //===============================================================================
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

  constexpr uint32_t kWorkerJobs = kNumLaunches / kNumWorkers;
  std::vector<Worker> workers(kNumWorkers);
  uint32_t k = 0;
  for (uint32_t idx = 0; idx < kNumLaunches; idx++) {
    k = idx / kWorkerJobs;
    workers[k].schedule([&, idx]() {
      CHECK_KERNEL_LAUNCH(
          gelu<<<blocks, threads>>>(dev_inp_ptr, dev_out_ptrs[idx], length));
    });
  }

  for (auto &worker : workers) {
    worker.start();
  }

  for (auto &worker : workers) {
    worker.stop();
  }

  cudaDeviceSynchronize();

  //===============================================================================
  // Epilogue
  //===============================================================================
  // Free allocated buffers
  CHECK_CUDA_ERROR(cudaFree(dev_inp_ptr));
  for (uint32_t i = 0; i < kNumLaunches; i++) {
    CHECK_CUDA_ERROR(cudaFree(dev_out_ptrs[i]));
  }

  // Destroy non-primary contexts
  for (int i = 0; i < kNumWorkers; i++) {
    cuCtxDestroy(contexts[i]);
  }

  // Release primary context
  cuCtxPopCurrent(&primary_ctx);
  cuDevicePrimaryCtxRelease(device);

  printf("DONE\n");
  return 0;
}
