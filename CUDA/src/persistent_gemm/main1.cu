#include <cstdio>
#include <cassert>
#include <vector>
#include <random>

#include <cstdlib>
#include <cuda_runtime.h>

#include "timer.h"

#define KERNEL_VERSION 1
#define TOSTR(x) #x
#define STRINGIFY(x) TOSTR(x)

#define WARP_SIZE 32
#define CHECK_CUDA_ERROR(apiCall)                                              \
  do {                                                                         \
    cudaError_t error = (apiCall);                                             \
    if (error != cudaSuccess) {                                                \
      auto errorName = cudaGetErrorName(error);                                \
      auto errorString = cudaGetErrorString(error);                            \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, errorName,         \
              errorString);                                                    \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)


__device__ void sync_cta() {
  asm volatile(
    "bar.sync 0;" : : : "memory"
  );
}

// No software pipeline
// Not persistent
template <uint32_t TILE_SIZE>
__global__ void gemm_kernel(const float *__restrict__ a,
                            const float *__restrict__ b, 
                            float *__restrict__ c,
                            uint32_t M, uint32_t N, uint32_t K) {
  assert(TILE_SIZE * TILE_SIZE == blockDim.x && "The number of threads in a CTA must be equal to the number of elements in a tile");

  // Indexed by threadIdx.x
  __shared__ float smem_a[TILE_SIZE * TILE_SIZE];
  __shared__ float smem_b[TILE_SIZE * TILE_SIZE];

  uint32_t tile_m = blockIdx.x;
  uint32_t tile_n = blockIdx.y;
  uint32_t tile_k_count = (K + TILE_SIZE - 1) / TILE_SIZE;

  uint32_t local_row = threadIdx.x / TILE_SIZE;
  uint32_t local_col = threadIdx.x % TILE_SIZE;
  uint32_t global_row_a = tile_m * TILE_SIZE + local_row;
  uint32_t global_col_b = tile_n * TILE_SIZE + local_col;

  // Each thread computes one element of the output tile
  float acc = 0;
  smem_a[threadIdx.x] = 0.0f;
  smem_b[threadIdx.x] = 0.0f;

  // Compute the output tile at (tile_m, tile_n) coordinate
  // Each thread computes a single element of an output tile
  for (uint32_t tile_k = 0; tile_k < tile_k_count; tile_k++) {
    // Load tile_m
    // Each thread loads one a[i][k] element
    //           |----|----|----|----|----|----|
    // tile_m -> |    |    |    |    |    |    |
    //           |----|----|----|----|----|----|
    //           |    |    |    |    |    |    |
    //           |----|----|----|----|----|----|
    //           |    |    |    |    |    |    |
    //           |----|----|----|----|----|----|

    uint32_t global_col_a = tile_k * TILE_SIZE + local_col;
    uint32_t gmem_index_a = global_row_a * K + global_col_a;

    // Handle OOB
    if (gmem_index_a < M * K) {
      smem_a[threadIdx.x] = a[gmem_index_a];
    }

    // Load tile_n
    uint32_t global_row_b = tile_k * TILE_SIZE + local_row;
    uint32_t gmem_index_b = global_row_b * N + global_col_b;

    // Handle OOB
    if (gmem_index_b < N * K) {
      smem_b[threadIdx.x] = b[gmem_index_b];
    }

    // Sync all threads in the same CTA
    // to ensure that they see the same smem_a and smem_b
    sync_cta();

    // Matmul smem_a @ smem_b
    // Each thread computes one acc
    #pragma unroll 4
    for (uint32_t k = 0; k < TILE_SIZE; k++) {
      acc += smem_a[local_row * TILE_SIZE + k] * smem_b[k * TILE_SIZE + local_col];
    }
  }

  // Epilogue
  // Store acc back to the resulting matrix c in gmem
  uint32_t global_row_c = tile_m * TILE_SIZE + local_row;
  uint32_t global_col_c = tile_n * TILE_SIZE + local_col;
  uint32_t gmem_index_c = global_row_c * N + global_col_c;
  c[gmem_index_c] = acc;
}


template <typename DataType>
std::vector<DataType> cpu_gemm(const DataType* a, const DataType* b, uint32_t M, uint32_t N, uint32_t K) {
  std::vector<DataType> c(M*N, 0);
  for (uint32_t i = 0; i < M; i++) {
    for (uint32_t j = 0; j < N; j++) {
      for (uint32_t k = 0; k < K; k++) {
        c[i * N + j] += a[i * K + k] * b[k * N + j];
      }
    }
  }

  return c;
}

int main(int argc, char **argv) {
  printf("KERNEL_VERSION = %d\n", KERNEL_VERSION);
  using DataType = float;
  constexpr uint32_t M = 129;
  constexpr uint32_t N = 257;
  constexpr uint32_t K = 512;
  
  constexpr uint32_t TILE_SIZE = 32;
  static_assert(TILE_SIZE * TILE_SIZE <= 1024 && "The number of elements in a tile should be less than or equal to 1024");

  constexpr uint32_t tile_m_count = (M + TILE_SIZE - 1) / TILE_SIZE;
  constexpr uint32_t tile_n_count = (N + TILE_SIZE - 1) / TILE_SIZE;
  constexpr std::size_t bytes_a = M*K * sizeof(DataType);
  constexpr std::size_t bytes_b = N*K * sizeof(DataType);
  constexpr std::size_t bytes_c = M*N * sizeof(DataType);

  auto a = std::vector<DataType>(M*K, 0);
  auto b = std::vector<DataType>(N*K, 0);

  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  for (int i = 0; i < M*K; i++) {
    a[i] = dist(gen);
  }

  for (int i = 0; i < N*K; i++) {
    b[i] = dist(gen);
  }

  DataType *dev_a_ptr = nullptr;
  DataType *dev_b_ptr = nullptr;
  DataType *dev_c_ptr = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&dev_a_ptr, bytes_a));
  CHECK_CUDA_ERROR(cudaMalloc(&dev_b_ptr, bytes_b));
  CHECK_CUDA_ERROR(cudaMalloc(&dev_c_ptr, bytes_c));

  // Copy inputs from CPU to GPU
  CHECK_CUDA_ERROR(
      cudaMemcpy(dev_a_ptr, a.data(), bytes_a, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(dev_b_ptr, b.data(), bytes_b, cudaMemcpyHostToDevice));

  dim3 gridSize(tile_m_count, tile_n_count, 1);
  dim3 blockSize(TILE_SIZE * TILE_SIZE, 1, 1);

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cudaEvent_t event_t0, event_t1;
  CHECK_CUDA_ERROR(cudaEventCreate(&event_t0));
  CHECK_CUDA_ERROR(cudaEventCreate(&event_t1));

  Timer timer;
  CHECK_CUDA_ERROR(cudaEventRecord(event_t0, stream));
  gemm_kernel<TILE_SIZE><<<gridSize, blockSize, 0, stream>>>(dev_a_ptr, dev_b_ptr, dev_c_ptr, M, N, K);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaEventRecord(event_t1, stream));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  float elapsed_time_ms;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_ms, event_t0, event_t1));
  printf("Kernel time measured by event: %6.3f ms\n", elapsed_time_ms);
  printf("Kernel time measured by timer: %6.3f ms\n", timer.elapsed_time_ms());

  auto gpu_res = std::vector<DataType>(M*N, 0);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_res.data(), dev_c_ptr, bytes_c, cudaMemcpyDeviceToHost));

  auto cpu_res = cpu_gemm<DataType>(a.data(), b.data(), M, N, K);

  bool ok = true;
  constexpr DataType epsilon = 1e-3;
  for (int i = 0; i < M*N; i++) {
    if (std::abs(gpu_res[i] - cpu_res[i]) > epsilon) {
      printf("GPU value %f != %f CPU value\n", gpu_res[i], cpu_res[i]);
      ok = false;
      break;
    }
  }

  printf("%s\n", (ok ? "PASSED" : "FAILED"));

  cudaFree(dev_a_ptr);
  cudaFree(dev_b_ptr);
  cudaFree(dev_c_ptr);
}
