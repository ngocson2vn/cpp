#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "timer.h"

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

#if (KERNEL_VERSION == 1)
__global__ void add_vectors(const float *a, const float *b, float *c,
                            int totalElems)

#elif (KERNEL_VERSION == 2)
__global__ void add_vectors(const float *__restrict__ a,
                            const float *__restrict__ b, float *__restrict__ c,
                            int totalElems)

#else
static_assert(false,
              "KERNEL_VERSION=" STRINGIFY(KERNEL_VERSION) " is unsupported!");
#endif
{
  auto numThreads = blockDim.x;
  auto numElems = (totalElems + numThreads - 1) / numThreads;
  auto tid = threadIdx.x;

  // numElems = 4
  //  tid=0  |  tid=1  |  tid=2
  // 0 1 2 3 | 4 5 6 7 | 8 9 10 11
  for (int idx = tid * numElems; idx < min((tid + 1) * numElems, totalElems);
       idx++) {
    c[idx] = a[idx] + b[idx];
  }
}

int main(int argc, char **argv) {
  using DataType = float;
  constexpr std::size_t kTotalElems = 128;
  constexpr std::size_t kNumBytes = kTotalElems * sizeof(DataType);

  printf("KERNEL_VERSION = %d\n\n", KERNEL_VERSION);

  auto a = std::vector<DataType>(kTotalElems, 0);
  auto b = std::vector<DataType>(kTotalElems, 0);
  auto c = std::vector<DataType>(kTotalElems, 0);

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
                                       kTotalElems);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  printf("Kernel time: %lu ns\n\n", timer.elapsed_time());

  CHECK_CUDA_ERROR(
      cudaMemcpy(c.data(), dev_c_ptr, kNumBytes, cudaMemcpyDeviceToHost));

  cudaFree(dev_a_ptr);
  cudaFree(dev_b_ptr);
  cudaFree(dev_c_ptr);

  printf("a: ");
  for (int i = 0; i < kTotalElems; i++) {
    printf("%7.1f", a[i]);
  }
  printf("\n\n");

  printf("b: ");
  for (int i = 0; i < kTotalElems; i++) {
    printf("%7.1f", b[i]);
  }
  printf("\n\n");

  printf("c: ");
  for (int i = 0; i < kTotalElems; i++) {
    printf("%7.1f", c[i]);
  }
  printf("\n");
}
