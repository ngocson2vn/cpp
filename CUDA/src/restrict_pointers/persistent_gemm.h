#include <cassert>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

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
      std::exit(EXIT_FAILURE);                                                     \
    }                                                                          \
  } while (0)



uint32_t get_sm_count() {
  cudaDeviceProp prop;
  auto errorCode = cudaGetDeviceProperties(&prop, 0);
  if (errorCode != cudaSuccess) {
    return 64;
  }

  return prop.multiProcessorCount;
}


// Persistent GEMM
template <typename DataType, uint32_t TILE_SIZE>
__global__ void persistent_gemm_kernel(const DataType *__restrict__ A,
                            const DataType *__restrict__ B, 
                            DataType *__restrict__ C,
                            const uint32_t OUT_ROWS, const uint32_t OUT_COLS,
                            const uint32_t M, const uint32_t N, const uint32_t K) {
  // Shared memory buffers for a_tile and b_tile;
  __shared__ DataType sA[TILE_SIZE * TILE_SIZE];
  __shared__ DataType sB[TILE_SIZE * TILE_SIZE];

  uint32_t m_tile_count = (M + TILE_SIZE - 1) / TILE_SIZE;
  uint32_t n_tile_count = (N + TILE_SIZE - 1) / TILE_SIZE;
  uint32_t k_tile_count = (K + TILE_SIZE - 1) / TILE_SIZE;

  // Total output tiles
  uint32_t total_tiles = m_tile_count * n_tile_count;

  // Each block is responsible for 1 output tile
  uint32_t tile_idx = blockIdx.x;

  // tile_idx must be strided by total blocks
  uint32_t blocks = gridDim.x;

  // Each thread is responsible for 1 element of output tile
  uint32_t local_row = threadIdx.x / TILE_SIZE;
  uint32_t local_col = threadIdx.x % TILE_SIZE;

  for (; tile_idx < total_tiles; tile_idx += blocks) {
    uint32_t m_tile = tile_idx / n_tile_count;
    uint32_t n_tile = tile_idx % n_tile_count;

    // Global row and col
    uint32_t global_row_a = m_tile * TILE_SIZE + local_row;
    uint32_t global_col_b = n_tile * TILE_SIZE + local_col;

    // Main loop
    DataType acc = 0;
    for (uint32_t k_tile = 0; k_tile < k_tile_count; k_tile++) {
      // Load A tile
      uint32_t global_col_a = k_tile * TILE_SIZE + local_col;
      sA[threadIdx.x] = A[global_row_a * K + global_col_a];

      // Load B tile
      uint32_t global_row_b = k_tile * TILE_SIZE + local_row;
      sB[threadIdx.x] = B[global_row_b * N + global_col_b];

      // Synchronize all threads in CTA to ensure that
      // all threads see the same sA and sB
      __syncthreads();

      // Compute dot product acc = sum(Aik * Bkj) for k = 0, ..., TILE_SIZE
      for (uint32_t k = 0; k < TILE_SIZE; k++) {
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


template <typename DataType>
std::vector<DataType> persistent_gemm(const std::vector<DataType>& A, const std::vector<DataType>& B, const uint32_t M, const uint32_t N, const uint32_t K) {
  constexpr uint32_t TILE_SIZE = 32;
  assert(TILE_SIZE * TILE_SIZE <= 1024 && "The number of elements in a tile should be less than or equal to 1024");

  //=====================================================================================
  // Padding M, N, and K
  //=====================================================================================
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

  auto A_ptr = A.data();
  auto B_ptr = B.data();

  // Fill the padded parts with 0
  // Initialize all elements to 0, 
  // so the padded part will be filled with 0
  std::vector<DataType> pA(pM*pK, 0);
  if (rm + rk != 0) {
    // We only need to copy M rows
    for (uint32_t i = 0; i < M; i++) {
      auto start = A.begin() + i * K;
      std::copy(start, start + K, pA.begin() + i * pK);
    }

    A_ptr = pA.data();
  }

  std::vector<DataType> pB(pK*pN, 0);
  if (rn + rk != 0) {
    for (uint32_t k = 0; k < K; k++) {
      auto start = B.begin() + k * N;
      std::copy(start, start + N, pB.begin() + k * pN);
    }

    B_ptr = pB.data();
  }
  //=====================================================================================

  const std::size_t aBytes = pM*pK * sizeof(DataType);
  const std::size_t bBytes = pN*pK * sizeof(DataType);

  // Allocate input buffers
  DataType *dev_a_ptr = nullptr;
  DataType *dev_b_ptr = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&dev_a_ptr, aBytes));
  CHECK_CUDA_ERROR(cudaMalloc(&dev_b_ptr, bBytes));

  // Allocate the output buffer
  // NOTE: using the original M and N
  const std::size_t cBytes = M*N * sizeof(DataType);
  DataType *dev_c_ptr = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&dev_c_ptr, cBytes));

  auto kernel_fn = persistent_gemm_kernel<DataType, TILE_SIZE>;
  uint32_t sm_count = get_sm_count();

  int threads = TILE_SIZE * TILE_SIZE;
  int per_sm_blocks = 1;
  CHECK_CUDA_ERROR(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_sm_blocks, kernel_fn,
      threads, 0
    )
  );

  int blocks = sm_count * per_sm_blocks;

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Copy inputs from CPU to GPU
  CHECK_CUDA_ERROR(cudaMemcpyAsync(dev_a_ptr, A_ptr, aBytes, cudaMemcpyHostToDevice, stream));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(dev_b_ptr, B_ptr, bBytes, cudaMemcpyHostToDevice, stream));

  // Launch persistent_gemm_kernel
  kernel_fn<<<blocks, threads, 0, stream>>>(dev_a_ptr, dev_b_ptr, dev_c_ptr, M, N, pM, pN, pK);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  // Copy output from GPU to CPU
  auto gpu_res = std::vector<DataType>(M*N, 0);
  CHECK_CUDA_ERROR(cudaMemcpy(gpu_res.data(), dev_c_ptr, cBytes, cudaMemcpyDeviceToHost));

  // Return
  return gpu_res;
}
