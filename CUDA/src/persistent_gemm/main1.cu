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

__device__ void pred_store_global(uint32_t pred /* 0 or 1 */, float* ptr, float v) {
  asm volatile(
    "{\n"
    "\t\t.reg .pred %p;\n"
    "\t\tsetp.eq.u32 %p, %0, 1;\n"
    "\t\t@%p st.global.f32 [%1], %2;\n"
    "\t}"
    :
    : "r"(pred), "l"(ptr), "f"(v)
  );
}

// No software pipeline and non-persistent
template <uint32_t TILE_SIZE>
__global__ void gemm_kernel(const float *__restrict__ A,
                            const float *__restrict__ B, 
                            float *__restrict__ C,
                            const uint32_t C_ROWS, const uint32_t C_COLS,
                            const uint32_t M, const uint32_t N, const uint32_t K) {
  assert(TILE_SIZE * TILE_SIZE == blockDim.x && "The number of threads in a CTA must be equal to the number of elements in a tile");

  // Indexed by threadIdx.x
  __shared__ float sA[TILE_SIZE * TILE_SIZE];
  __shared__ float sB[TILE_SIZE * TILE_SIZE];

  uint32_t tile_m = blockIdx.x;
  uint32_t tile_n = blockIdx.y;
  uint32_t tile_k_count = (K + TILE_SIZE - 1) / TILE_SIZE;

  uint32_t local_row = threadIdx.x / TILE_SIZE;
  uint32_t local_col = threadIdx.x % TILE_SIZE;
  uint32_t global_row_a = tile_m * TILE_SIZE + local_row;
  uint32_t global_col_b = tile_n * TILE_SIZE + local_col;

  // Each thread computes one element of the output tile
  float acc = 0;

  // Compute the output tile at (tile_m, tile_n) coordinate
  // Each thread computes a single element of an output tile
  for (uint32_t tile_k = 0; tile_k < tile_k_count; tile_k++) {
    // Load tile_m
    // Each thread loads one A[i][k] element
    //           tile_k
    //             ↓
    //          |-----|-----|-----|-----|-----|-----|
    // tile_m → |     |     |     |     |     |     |
    //          |-----|-----|-----|-----|-----|-----|
    //          |     |     |     |     |     |     |
    //          |-----|-----|-----|-----|-----|-----|
    //          |     |     |     |     |     |     |
    //          |-----|-----|-----|-----|-----|-----|

    uint32_t global_col_a = tile_k * TILE_SIZE + local_col;
    uint32_t gmem_index_a = global_row_a * K + global_col_a;
    sA[threadIdx.x] = A[gmem_index_a];

    // Load tile_n
    // Each thread loads one B[k][j] element
    //           tile_n
    //             ↓
    //          |-----|-----|-----|-----|-----|-----|
    // tile_k → |     |     |     |     |     |     |
    //          |-----|-----|-----|-----|-----|-----|
    //          |     |     |     |     |     |     |
    //          |-----|-----|-----|-----|-----|-----|
    //          |     |     |     |     |     |     |
    //          |-----|-----|-----|-----|-----|-----|
    //          |     |     |     |     |     |     |
    //          |-----|-----|-----|-----|-----|-----|
    //          |     |     |     |     |     |     |
    //          |-----|-----|-----|-----|-----|-----|
    //          |     |     |     |     |     |     |
    //          |-----|-----|-----|-----|-----|-----|
    uint32_t global_row_b = tile_k * TILE_SIZE + local_row;
    uint32_t gmem_index_b = global_row_b * N + global_col_b;
    sB[threadIdx.x] = B[gmem_index_b];

    // Sync all threads in the same CTA
    // to ensure that they see the same sA and sB
    sync_cta();

    // Matmul sA @ sB
    // Each thread computes one acc
    #pragma unroll 4
    for (uint32_t k = 0; k < TILE_SIZE; k++) {
      acc += sA[local_row * TILE_SIZE + k] * sB[k * TILE_SIZE + local_col];
    }
  }

  // Epilogue
  // Store acc back to the resulting matrix C in gmem
  uint32_t global_row_c = tile_m * TILE_SIZE + local_row;
  uint32_t global_col_c = tile_n * TILE_SIZE + local_col;
  uint32_t pred = (global_row_c < C_ROWS && global_col_c < C_COLS) ? 1 : 0;
  uint32_t gmem_index_c = global_row_c * C_COLS + global_col_c;
  pred_store_global(pred, &C[gmem_index_c], acc);
}


template <typename DataType>
std::vector<DataType> cpu_gemm(const DataType* A, const DataType* B, uint32_t M, uint32_t N, uint32_t K) {
  std::vector<DataType> C(M*N, 0);
  for (uint32_t i = 0; i < M; i++) {
    for (uint32_t j = 0; j < N; j++) {
      for (uint32_t k = 0; k < K; k++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }

  return C;
}

int main(int argc, char **argv) {
  printf("KERNEL_VERSION = %d\n", KERNEL_VERSION);
  using DataType = float;
  uint32_t M = 1027;
  uint32_t N = 2026;
  uint32_t K = 1111;
  constexpr uint32_t TILE_SIZE = 32;
  assert(TILE_SIZE * TILE_SIZE <= 1024 && "The number of elements in a tile should be less than or equal to 1024");

  auto A = std::vector<DataType>(M*K, 0);
  auto B = std::vector<DataType>(N*K, 0);

  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> dist(0.0, 0.1);

  for (int i = 0; i < M*K; i++) {
    A[i] = dist(gen);
  }

  for (int i = 0; i < N*K; i++) {
    B[i] = dist(gen);
  }

  // CPU GEMM
  printf("Executing CPU GEMM\n");
  auto cpu_res = cpu_gemm<DataType>(A.data(), B.data(), M, N, K);
  printf("\n");


  //=====================================================================================
  // Padding M, N, and K
  //=====================================================================================
  printf("Padding M, N, and K\n");
  uint32_t pM = M;
  uint32_t rm = M % TILE_SIZE;
  if (rm != 0) {
    // M = qm * TILE_SIZE + rm
    // => M + (TILE_SIZE - rm) = qm * TILE_SIZE + rm + TILE_SIZE - rm 
    //                         = qm * TILE_SIZE + TILE_SIZE 
    //                         = (qm + 1) * TILE_SIZE
    pM = M + (TILE_SIZE - rm);
  }

  uint32_t pN = N;
  uint32_t rn = N % TILE_SIZE;
  if (rn != 0) {
    pN = N + (TILE_SIZE - rn);
  }

  uint32_t pK = K;
  uint32_t rk = K % TILE_SIZE;
  if (rk != 0) {
    pK = K + (TILE_SIZE - rk);
  }

  printf("M = %d, pM = %d\n", M, pM);
  printf("N = %d, pN = %d\n", N, pN);
  printf("K = %d, pK = %d\n", K, pK);

  // Fill the padded parts with 0
  if (rm + rk != 0) {
    printf("PRE: A has %d elements\n", A.size());

    // Initialize all elements to 0, 
    // so the padded part will be filled with 0
    std::vector<float> P(pM*pK, 0);

    // Next, we only need to copy M rows
    for (uint32_t i = 0; i < M; i++) {
      auto start = A.begin() + i * K;
      std::copy(start, start + K, P.begin() + i * pK);
    }

    // Swap
    A.clear();
    A = std::move(P);
    printf("NOW: A has %d elements\n", A.size());
  }

  if (rn + rk != 0) {
    printf("PRE: B has %d elements\n", B.size());
    std::vector<float> P(pK*pN);
    for (uint32_t k = 0; k < K; k++) {
      auto start = B.begin() + k * N;
      std::copy(start, start + N, P.begin() + k * pN);
    }

    // Swap
    B = std::move(P);
    printf("NOW: B has %d elements\n", B.size());
  }

  printf("\n");
  //=====================================================================================

  const uint32_t tile_m_count = (pM + TILE_SIZE - 1) / TILE_SIZE;
  const uint32_t tile_n_count = (pN + TILE_SIZE - 1) / TILE_SIZE;
  const std::size_t bytes_a = pM*pK * sizeof(DataType);
  const std::size_t bytes_b = pN*pK * sizeof(DataType);

  // Copy inputs from CPU to GPU
  DataType *dev_a_ptr = nullptr;
  DataType *dev_b_ptr = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&dev_a_ptr, bytes_a));
  CHECK_CUDA_ERROR(cudaMalloc(&dev_b_ptr, bytes_b));
  CHECK_CUDA_ERROR(cudaMemcpy(dev_a_ptr, A.data(), bytes_a, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(dev_b_ptr, B.data(), bytes_b, cudaMemcpyHostToDevice));

  // Allocate the output buffer
  // NOTE: using the original M and N
  const std::size_t bytes_c = M*N * sizeof(DataType);
  DataType *dev_c_ptr = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&dev_c_ptr, bytes_c));

  dim3 gridSize(tile_m_count, tile_n_count, 1);
  dim3 blockSize(TILE_SIZE * TILE_SIZE, 1, 1);

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cudaEvent_t event_t0, event_t1;
  CHECK_CUDA_ERROR(cudaEventCreate(&event_t0));
  CHECK_CUDA_ERROR(cudaEventCreate(&event_t1));

  printf("Launching gemm_kernel\n");
  Timer timer;
  CHECK_CUDA_ERROR(cudaEventRecord(event_t0, stream));
  gemm_kernel<TILE_SIZE><<<gridSize, blockSize, 0, stream>>>(dev_a_ptr, dev_b_ptr, dev_c_ptr, M, N, pM, pN, pK);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaEventRecord(event_t1, stream));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  float elapsed_time_ms;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_ms, event_t0, event_t1));
  printf("Kernel time measured by event: %6.3f ms\n", elapsed_time_ms);
  printf("Kernel time measured by timer: %6.3f ms\n", timer.elapsed_time_ms());

  auto gpu_res = std::vector<DataType>(M*N, 0);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_res.data(), dev_c_ptr, bytes_c, cudaMemcpyDeviceToHost));

  //=====================================================================================
  // Verify if GPU result matches CPU result
  //=====================================================================================
  bool ok = true;
  constexpr DataType epsilon = 1e-2;
  for (int i = 0; i < M*N; i++) {
    if (std::abs(gpu_res[i] - cpu_res[i]) > epsilon) {
      printf("i = %d: GPU value %f != %f CPU value\n", i, gpu_res[i], cpu_res[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", (ok ? "PASSED" : "FAILED"));

  cudaFree(dev_a_ptr);
  cudaFree(dev_b_ptr);
  cudaFree(dev_c_ptr);
}
