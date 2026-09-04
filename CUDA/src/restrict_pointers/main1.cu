#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "timer.h"

#define KERNEL_VERSION 1
#define TOSTR(x) #x
#define STRINGIFY(x) TOSTR(x)

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


__global__ void add_vectors(const float *a, const float *b, float *c,
                            int totalElems, int maxRound) {
  auto numThreads = blockDim.x;
  auto numElems = (totalElems + numThreads - 1) / numThreads;
  auto tid = threadIdx.x;

  // numElems = 4
  //  tid=0  |  tid=1  |  tid=2
  // 0 1 2 3 | 4 5 6 7 | 8 9 10 11

  // Ensure that `maxIdx` is not greater than `totalElems`
  auto maxIdx = min((tid + 1) * numElems, totalElems);

  for (int i = 0; i < maxRound; i++) {
    for (int idx = tid * numElems; idx < maxIdx; idx++) {
      c[idx] = a[idx] + b[idx];
    }
  }
}

int main(int argc, char **argv) {
  using DataType = float;
  constexpr std::size_t kTotalElems = 128;
  constexpr std::size_t kMaxRound = 1000000;
  constexpr std::size_t kNumBytes = kTotalElems * sizeof(DataType);

  printf("KERNEL_VERSION = %d\n\n", KERNEL_VERSION);

  auto a = std::vector<DataType>(kTotalElems, 0);
  auto b = std::vector<DataType>(kTotalElems, 0);
  auto gpu_res = std::vector<DataType>(kTotalElems, 0);

  for (int i = 0; i < kTotalElems; i++) {
    a[i] = DataType(i);
    b[i] = DataType(i);
  }

  DataType *dev_a_ptr = nullptr;
  DataType *dev_b_ptr = nullptr;
  DataType *dev_c_ptr = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&dev_a_ptr, kNumBytes));
  CHECK_CUDA_ERROR(cudaMalloc(&dev_b_ptr, kNumBytes));
  CHECK_CUDA_ERROR(cudaMalloc(&dev_c_ptr, kNumBytes));

  // Copy inputs from CPU to GPU
  CHECK_CUDA_ERROR(
      cudaMemcpy(dev_a_ptr, a.data(), kNumBytes, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(dev_b_ptr, b.data(), kNumBytes, cudaMemcpyHostToDevice));

  dim3 gridSize(1, 1, 1);
  dim3 blockSize(32, 1, 1);

  Timer timer;
  add_vectors<<<gridSize, blockSize>>>(dev_a_ptr, dev_b_ptr, dev_c_ptr,
                                       kTotalElems, kMaxRound);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  printf("Kernel time: %lu ns\n\n", timer.elapsed_time());

  CHECK_CUDA_ERROR(
      cudaMemcpy(gpu_res.data(), dev_c_ptr, kNumBytes, cudaMemcpyDeviceToHost));

  cudaFree(dev_a_ptr);
  cudaFree(dev_b_ptr);
  cudaFree(dev_c_ptr);

  auto cpu_res = std::vector<DataType>(kTotalElems, 0);
  for (int i = 0; i < kTotalElems; i++) {
    cpu_res[i] = a[i] + b[i];
  }

  bool ok = true;
  for (int i = 0; i < kTotalElems; i++) {
    if (gpu_res[i] != cpu_res[i]) {
      ok = false;
      break;
    }
  }

  printf("%s\n", (ok ? "PASSED" : "FAILED"));
}
