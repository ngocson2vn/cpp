#include <cstdio>
#include <cassert>
#include <vector>

#include <cstdlib>
#include <cuda_runtime.h>

#include "timer.h"

#define KERNEL_VERSION 1
#define TOSTR(x) #x
#define STRINGIFY(x) TOSTR(x)

#define WARP_SIZE 32
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

extern "C" {

__device__ void set_done(uint32_t tid, bool* ptr) {
  asm volatile(
    "{\n"
    "\t\t.reg .pred %p;\n"
    "\t\tsetp.eq.u32 %p, %0, 0;\n"
    "\t\t@%p st.global.release.gpu.b8 [%1], 1;\n"
    "\t}"
    :
    : "r"(tid), "l"(ptr)
  );
}

__global__ void add_vectors(const float *__restrict__ a,
                            const float *__restrict__ b,
                            float *__restrict__ c,
                            bool *__restrict__ done_stats,
                            int blockSize, int totalElems) {
  auto blockIndex = blockIdx.x;

  while (blockIndex > 0 && !done_stats[blockIndex - 1]) {
    __nanosleep(1000);
  }

  auto numThreads = blockDim.x;
  auto numElems = (blockSize + numThreads - 1) / numThreads;
  auto tid = threadIdx.x;

  //================================================
  // Normal case
  //================================================
  // numThreads = 32
  //  blockSize = 128
  //   numElems = 4
  // 
  //  tid=0  |  tid=1  |  tid=2
  // 0 1 2 3 | 4 5 6 7 | 8 9 10 11

  //================================================
  // Abnormal case
  //================================================
  // numThreads = 64
  //  blockSize = 32
  //   numElems = 1
  // 
  //  tid=0  |  tid=1  |  tid=2  |  tid=3
  //    0    |    1    |    2    |    3

  auto blockLowerBound = blockIndex * blockSize;
  auto blockUpperBound = blockLowerBound + blockSize;

  auto lowerBound = blockLowerBound + tid * numElems;
  auto upperBound = min(lowerBound + numElems, blockUpperBound);

  // Ensure that `upperBound` is not greater than `totalElems`
  upperBound = min(upperBound, totalElems);

  #pragma unroll 1
  for (int idx = lowerBound; idx < upperBound; idx++) {
    c[idx] = a[idx] + b[idx];
  }

  __syncthreads();
  set_done(tid, &done_stats[blockIndex]);
}

}

int main(int argc, char **argv) {
  using DataType = float;
  constexpr int kNumWarps = 1;
  constexpr int kNumThreads = kNumWarps * WARP_SIZE;

  int totalElems = 1024000;
  if (argc > 1) {
    totalElems = std::atoi(argv[1]);
  }

  const int kBlockX = totalElems / 2;
  const int threadElems = (kBlockX + kNumThreads - 1) / kNumThreads;
  printf("threadElems = %d\n", threadElems);
  assert(threadElems % 4 == 0 && "threadElems is not divisible by 4");

  const std::size_t kNumBytes = totalElems * sizeof(DataType);

  printf("KERNEL_VERSION = %d\n", KERNEL_VERSION);

  auto a = std::vector<DataType>(totalElems, 0);
  auto b = std::vector<DataType>(totalElems, 0);
  auto gpu_res = std::vector<DataType>(totalElems, 0);

  for (int i = 0; i < totalElems; i++) {
    a[i] = DataType(i);
    b[i] = DataType(i);
  }

  DataType *dev_a_ptr = nullptr;
  DataType *dev_b_ptr = nullptr;
  DataType *dev_c_ptr = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&dev_a_ptr, kNumBytes));
  CHECK_CUDA_ERROR(cudaMalloc(&dev_b_ptr, kNumBytes));
  CHECK_CUDA_ERROR(cudaMalloc(&dev_c_ptr, kNumBytes));

  bool *dev_done_stats = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&dev_done_stats, 2));

  // Copy inputs from CPU to GPU
  CHECK_CUDA_ERROR(
      cudaMemcpy(dev_a_ptr, a.data(), kNumBytes, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(dev_b_ptr, b.data(), kNumBytes, cudaMemcpyHostToDevice));

  // Clear dev_done_stats
  bool init_stats[] = {false, false};
  CHECK_CUDA_ERROR(
      cudaMemcpy(dev_done_stats, init_stats, 2, cudaMemcpyHostToDevice));

  // Launch 2 blocks
  dim3 gridSize(2, 1, 1);
  dim3 blockSize(kNumThreads, 1, 1);


  Timer timer;
  add_vectors<<<gridSize, blockSize>>>(dev_a_ptr, dev_b_ptr, 
                                       dev_c_ptr, dev_done_stats,
                                       kBlockX, totalElems);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  printf("Kernel time: %lu us\n", timer.elapsed_time_us());

  CHECK_CUDA_ERROR(
      cudaMemcpy(gpu_res.data(), dev_c_ptr, kNumBytes, cudaMemcpyDeviceToHost));

  cudaFree(dev_a_ptr);
  cudaFree(dev_b_ptr);
  cudaFree(dev_c_ptr);
  cudaFree(dev_done_stats);

  auto cpu_res = std::vector<DataType>(totalElems, 0);
  for (int i = 0; i < totalElems; i++) {
    cpu_res[i] = a[i] + b[i];
  }

  bool ok = true;
  for (int i = 0; i < totalElems; i++) {
    if (gpu_res[i] != cpu_res[i]) {
      ok = false;
      break;
    }
  }

  printf("%s\n", (ok ? "PASSED" : "FAILED"));
}
