#include <cstdio>
#include <cassert>
#include <vector>

#include <cstdlib>
#include <cuda_runtime.h>

#include "timer.h"

#define KERNEL_VERSION 3
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


__device__ void ld_v4_f32(float& v0, float& v1, float& v2, float& v3, const float* ptr) {
  asm volatile(
    "ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
    : "=f"(v0), "=f"(v1), "=f"(v2), "=f"(v3)
    : "l"(ptr)
  );
}

__device__ void st_v4_f32(float* ptr, float v0, float v1, float v2, float v3) {
  asm volatile(
    "st.global.v4.f32 [%0], {%1, %2, %3, %4};"
    :
    : "l"(ptr), "f"(v0), "f"(v1), "f"(v2), "f"(v3)
  );
}

template <int TILE_SIZE>
__global__ void add_vectors(const float *__restrict__ a,
                            const float *__restrict__ b,
                            float *__restrict__ c, const int N) {
  const int TOTAL_TILES =(N + TILE_SIZE - 1) / TILE_SIZE;
  const int TOTAL_THREADS = gridDim.x * blockDim.x;
  int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;

  float a0, a1, a2, a3;
  float b0, b1, b2, b3;
  #pragma unroll 1
  while (tile_idx < TOTAL_TILES) {
    int start_idx = tile_idx * TILE_SIZE;
    int idx = start_idx;

    #pragma unroll 1
    for (; (idx + 4 - start_idx <= TILE_SIZE) && (idx + 4 < N); idx+=4) {
      ld_v4_f32(a0, a1, a2, a3, &a[idx]);
      ld_v4_f32(b0, b1, b2, b3, &b[idx]);
      st_v4_f32(&c[idx], a0 + b0, a1 + b1, a2 + b2, a3 + b3);
    }

    #pragma unroll 1
    for (; (idx - start_idx <= TILE_SIZE) && (idx < N); idx++) {
      c[idx] = a[idx] + b[idx];
    }

    tile_idx += TOTAL_THREADS;
  }
}


int main(int argc, char **argv) {
  using DataType = float;

  int N = 1024 * 1024 * 128 + 1;
  if (argc > 1) {
    N = std::atoi(argv[1]);
  }

  const std::size_t kNumBytes = N * sizeof(DataType);

  printf("KERNEL_VERSION = %d\n", KERNEL_VERSION);

  auto a = std::vector<DataType>(N, 0);
  auto b = std::vector<DataType>(N, 0);
  auto gpu_res = std::vector<DataType>(N, 0);

  for (int i = 0; i < N; i++) {
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


  // Each thread processes TILE_SIZE elements
  constexpr int TILE_SIZE = 4;
  static_assert(TILE_SIZE % 4 == 0 && "TILE_SIZE must be equal to 4");

  const int totalTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
  printf("N = %d\n", N);
  printf("totalTiles = %d\n", totalTiles);

  // Compute optimal blocks and threads
  int blocks = 1;
  int threads = 256;
  auto kernel_fn = add_vectors<TILE_SIZE>;
  cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, kernel_fn, 0, 0);
  printf("[PRE] blocks = %d threads = %d\n", blocks, threads);

  if (totalTiles < threads) {
    blocks = 1;
    threads = totalTiles;
  } else if (totalTiles < blocks * threads) {
    blocks = (totalTiles + threads - 1) / threads;
  } else {
    // Using the heuristic blocks and threads above
  }
  printf("[OPT] blocks = %d threads = %d\n", blocks, threads);

  // Timer timer;
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0, 0);
  kernel_fn<<<blocks, threads, 0, 0>>>(dev_a_ptr, dev_b_ptr, dev_c_ptr, N);
  cudaEventRecord(e1, 0);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  float elapsed_time_ms;
  cudaEventElapsedTime(&elapsed_time_ms, e0, e1);
  printf("Kernel time: %f ms\n", elapsed_time_ms);

  CHECK_CUDA_ERROR(
      cudaMemcpy(gpu_res.data(), dev_c_ptr, kNumBytes, cudaMemcpyDeviceToHost));

  cudaFree(dev_a_ptr);
  cudaFree(dev_b_ptr);
  cudaFree(dev_c_ptr);

  auto cpu_res = std::vector<DataType>(N, 0);
  for (int i = 0; i < N; i++) {
    cpu_res[i] = a[i] + b[i];
  }

  bool ok = true;
  for (int i = 0; i < N; i++) {
    if (gpu_res[i] != cpu_res[i]) {
      ok = false;
      break;
    }
  }

  printf("%s\n", (ok ? "PASSED" : "FAILED"));
}
