#include <cstdio>
#include <cassert>
#include <vector>
#include <random>

#include <cstdlib>
#include <cuda_runtime.h>

#include "timer.h"

#define KERNEL_VERSION 2
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


int get_sm_count() {
  cudaDeviceProp prop;
  auto errorCode = cudaGetDeviceProperties(&prop, 0);
  if (errorCode != cudaSuccess) {
    return 64;
  }

  return prop.multiProcessorCount;
}


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

// Persistent GEMM
template <uint32_t TILE_SIZE>
__global__ void persistent_gemm_kernel(const float *__restrict__ A,
                            const float *__restrict__ B, 
                            float *__restrict__ C,
                            const uint32_t OUT_ROWS, const uint32_t OUT_COLS,
                            const uint32_t M, const uint32_t N, const uint32_t K) {
  assert(TILE_SIZE * TILE_SIZE == blockDim.x && "The number of threads in a CTA must be equal to the number of elements in a tile");
  
  // Shared memory buffers for a_tile and b_tile;
  __shared__ float sA[TILE_SIZE * TILE_SIZE];
  __shared__ float sB[TILE_SIZE * TILE_SIZE];

  int m_tile_count = (M + TILE_SIZE - 1) / TILE_SIZE;
  int n_tile_count = (N + TILE_SIZE - 1) / TILE_SIZE;
  int k_tile_count = (K + TILE_SIZE - 1) / TILE_SIZE;

  // Total output tiles
  int total_tiles = m_tile_count * n_tile_count;

  // Each block is responsible for 1 output tile
  int tile_idx = blockIdx.x;

  // tile_idx must be strided by total blocks
  int blocks = gridDim.x;

  // Each thread is responsible for 1 element of output tile
  int local_row = threadIdx.x / TILE_SIZE;
  int local_col = threadIdx.x % TILE_SIZE;

  for (; tile_idx < total_tiles; tile_idx += blocks) {
    int m_tile = tile_idx / n_tile_count;
    int n_tile = tile_idx % n_tile_count;

    // Global row and col
    int global_row_a = m_tile * TILE_SIZE + local_row;
    int global_col_b = n_tile * TILE_SIZE + local_col;

    // Main loop
    float acc = 0;
    for (int k_tile = 0; k_tile < k_tile_count; k_tile++) {
      // Load A tile
      int global_col_a = k_tile * TILE_SIZE + local_col;
      sA[threadIdx.x] = A[global_row_a * K + global_col_a];

      // Load B tile
      int global_row_b = k_tile * TILE_SIZE + local_row;
      sB[threadIdx.x] = B[global_row_b * N + global_col_b];

      // Synchronize all threads in CTA to ensure that
      // all threads see the same sA and sB
      __syncthreads();

      // Compute dot product acc = sum(Aik * Bkj) for k = 0, ..., TILE_SIZE
      for (int k = 0; k < TILE_SIZE; k++) {
        acc += sA[local_row * TILE_SIZE + k] * sB[k * TILE_SIZE + local_col];
      }
    }

    // Epilogue: store acc back to GMEM
    // Each thread is responsible for 1 element of output tile
    if (global_row_a < OUT_ROWS && global_col_b < OUT_COLS) {
      C[global_row_a * OUT_COLS + global_col_b] = acc;
    }
  }
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

  const std::size_t bytes_a = pM*pK * sizeof(DataType);
  const std::size_t bytes_b = pN*pK * sizeof(DataType);

  // Allocate pinned memory buffers
  void* pinnedA;
  void* pinnedB;
  cudaMallocHost(&pinnedA, bytes_a);
  cudaMallocHost(&pinnedB, bytes_b);

  // Copy A and B to pinned memory locations;
  memcpy(pinnedA, A.data(), bytes_a);
  memcpy(pinnedB, B.data(), bytes_b);

  printf("\n");
  //=====================================================================================

  // Allocate input buffers
  DataType *dev_a_ptr = nullptr;
  DataType *dev_b_ptr = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&dev_a_ptr, bytes_a));
  CHECK_CUDA_ERROR(cudaMalloc(&dev_b_ptr, bytes_b));

  // Allocate the output buffer
  // NOTE: using the original M and N
  const std::size_t bytes_c = M*N * sizeof(DataType);
  DataType *dev_c_ptr = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&dev_c_ptr, bytes_c));

  int sm_count = get_sm_count();

  int threads = TILE_SIZE * TILE_SIZE;
  int per_sm_blocks = 1;
  CHECK_CUDA_ERROR(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_sm_blocks, persistent_gemm_kernel<TILE_SIZE>,
      threads, 0
    )
  );

  int blocks = sm_count * per_sm_blocks;
  printf("blocks = %d, threads = %d\n", blocks, threads);

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cudaEvent_t e0, e1;
  CHECK_CUDA_ERROR(cudaEventCreate(&e0));
  CHECK_CUDA_ERROR(cudaEventCreate(&e1));

  printf("Launching persistent_gemm_kernel\n");
  CHECK_CUDA_ERROR(cudaEventRecord(e0, stream));

  Timer timer;

  // Copy inputs from CPU to GPU
  CHECK_CUDA_ERROR(cudaMemcpyAsync(dev_a_ptr, pinnedA, bytes_a, cudaMemcpyHostToDevice, stream));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(dev_b_ptr, pinnedB, bytes_b, cudaMemcpyHostToDevice, stream));
  persistent_gemm_kernel<TILE_SIZE><<<blocks, threads, 0, stream>>>(dev_a_ptr, dev_b_ptr, dev_c_ptr, M, N, pM, pN, pK);
  // CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaEventRecord(e1, stream));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  float elapsed_time_ms;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_ms, e0, e1));
  printf("Total GPU time (memory + kernel) measured by event: %6.3f ms\n", elapsed_time_ms);
  printf("Total GPU time (memory + kernel) measured by timer: %6.3f ms\n", timer.elapsed_time_ms());

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
  cudaFreeHost(pinnedA);
  cudaFreeHost(pinnedB);
}
