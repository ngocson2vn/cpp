/*
Features:
  1. Persistent GEMM
*/

#include <cstdio>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(callExpr)                                             \
  do {                                                                         \
    cudaError_t status = callExpr;                                             \
    if (status != cudaSuccess) {                                               \
      auto errorName = cudaGetErrorName(status);                               \
      auto errorStr = cudaGetErrorString(status);                              \
      fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, errorName,        \
              errorStr);                                                       \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

template <typename DataType>
std::vector<DataType> cpuGemm(const std::vector<DataType> &A,
                              const std::vector<DataType> &B, const int M,
                              const int N, const int K) {
  std::vector<DataType> C(M * N, 0);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      auto acc = DataType(0);
      for (int k = 0; k < K; k++) {
        acc += A[i * K + k] * B[k * N + j];
      }

      C[i * N + j] = acc;
    }
  }

  return C;
}

template <typename DataType, int TILE_SIZE>
__global__ void persistentGemmKernel(const DataType *__restrict__ A,
                                     const DataType *__restrict__ B,
                                     DataType *__restrict__ C, 
                                     const int N,
                                     const int K, 
                                     const int nTileCount,
                                     const int kTileCount,
                                     const int totalTiles) {
  extern __shared__ DataType globalSmem[];

  DataType *sA = globalSmem;
  DataType *sB = globalSmem + (TILE_SIZE * TILE_SIZE);

  /*
  - Total blocks = total SMs
  - Divide matrix C into multiple TILE_SIZE x TILE_SIZE tiles
  - Each TILE_SIZE x TILE_SIZE tile has a linear index
  - Each block computes one TILE_SIZE x TILE_SIZE tile
    - Each thread computes one element of this tile
  - Each block moves to next tile with index = current tile index + total blocks
    - Each thread must check if its current tile index < total tiles
  */

  // Compute total blocks
  const auto blocks = gridDim.x;

  // Compute the first tileIdx
  auto tileIdx = blockIdx.x;

  while (tileIdx < totalTiles) {
    // Compute m_tile and n_tile based on the formula
    // tileIdx = mTileIdx * nTileCount + nTileIdx
    auto mTileIdx = tileIdx / nTileCount;
    auto nTileIdx = tileIdx % nTileCount;

    // Compute global row index for A and C
    auto localRowA = threadIdx.x / TILE_SIZE;
    auto globalRowA = mTileIdx * TILE_SIZE + localRowA;

    // Compute global col index for B and C
    auto localColB = threadIdx.x % TILE_SIZE;
    auto globalColB = nTileIdx * TILE_SIZE + localColB;

    DataType acc = DataType(0);
    for (int kTileIdx = 0; kTileIdx < kTileCount; kTileIdx++) {
      // To load an element of A or B, we need to know its global indices

      // Compute global col index for A
      auto localColA = threadIdx.x % TILE_SIZE;
      auto globalColA = kTileIdx * TILE_SIZE + localColA;

      // Compute global row index for B
      auto localRowB = threadIdx.x / TILE_SIZE;
      auto globalRowB = kTileIdx * TILE_SIZE + localRowB;

      // Now that we already know
      // - (globalRowA, globalColA)
      // - (globalRowB, globalColB)
      // we can load elements of A and B to SMEM
      sA[threadIdx.x] = A[globalRowA * K + globalColA];
      sB[threadIdx.x] = B[globalRowB * N + globalColB];

      // Synchronize all threads in CTA to ensure that
      // they all see the same images of sA and sB
      __syncthreads();

      // Compute dot product
      #pragma unroll TILE_SIZE
      for (int k = 0; k < TILE_SIZE; k++) {
        acc += sA[localRowA * TILE_SIZE + k] * sB[k * TILE_SIZE + localColB];
      }

      // Synchronize all threads in CTA to ensure that
      // they all finish reading sA and sB before they
      // start updating sA and sB again
      __syncthreads();
    }

    C[globalRowA * N + globalColB] = acc;

    // Compute next tile index
    tileIdx += blocks;
  }
}

int getSmCount() {
  int defaultSmCount = 64;
  cudaDeviceProp devProp;
  auto error = cudaGetDeviceProperties(&devProp, 0);
  if (error != cudaSuccess) {
    return defaultSmCount;
  }

  return devProp.multiProcessorCount;
}

template <typename DataType, int TILE_SIZE>
std::vector<DataType> gpuGemm(
                        const std::vector<DataType>& A,
                        const std::vector<DataType>& B,
                        const int M, const int N, const int K) {
  constexpr int elemBytes = sizeof(DataType);
  const int aNumBytes = M * K * elemBytes;
  const int bNumBytes = K * N * elemBytes;
  const int cNumBytes = M * N * elemBytes;

  // Dynamic SMEM size
  constexpr int smemBytes = 2 * (TILE_SIZE * TILE_SIZE) * elemBytes;

  // Get sm count
  auto smCount = getSmCount();
  printf("The number of SMs: %d\n", smCount);

  auto kernelFn = reinterpret_cast<const void *>(
                      persistentGemmKernel<DataType, TILE_SIZE>);

  // Compute an optimal number of blocks
  int threads = TILE_SIZE * TILE_SIZE;
  int blocks = smCount;
  int blocksPerSm = 1;
  {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, kernelFn,
                                                  threads, smemBytes);

    blocks = smCount * blocksPerSm;
  }
  printf("blocks = %d threads=%d\n", blocks, threads);

  //
  // Prepare arguments
  //

  // Allocate device buffers
  DataType *devPtrA;
  DataType *devPtrB;
  DataType *devPtrC;
  CHECK_CUDA_ERROR(cudaMalloc(&devPtrA, aNumBytes));
  CHECK_CUDA_ERROR(cudaMalloc(&devPtrB, bNumBytes));
  CHECK_CUDA_ERROR(cudaMalloc(&devPtrC, cNumBytes));

  // Copy A and B to GMEM
  CHECK_CUDA_ERROR(
      cudaMemcpy(devPtrA, A.data(), aNumBytes, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(devPtrB, B.data(), bNumBytes, cudaMemcpyHostToDevice));

  // Compute tile counts
  const auto mTileCount = (M + TILE_SIZE - 1) / TILE_SIZE; // ceil_div(M, TILE_SIZE)
  const auto nTileCount = (N + TILE_SIZE - 1) / TILE_SIZE; // ceil_div(N, TILE_SIZE)
  const auto kTileCount = (K + TILE_SIZE - 1) / TILE_SIZE; // ceil_div(K, TILE_SIZE)
  const auto totalTiles = mTileCount * nTileCount;

  void *kernelArgs[] = {
    &devPtrA,
    &devPtrB,
    &devPtrC,
    const_cast<int*>(&N),
    const_cast<int*>(&K),
    const_cast<int*>(&nTileCount),
    const_cast<int*>(&kTileCount),
    const_cast<int*>(&totalTiles),
  };

  // Launch CUDA kernel
  {
    // cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void
    // **args, size_t sharedMem, cudaStream_t stream)
    dim3 blockConfig(blocks);
    dim3 threadConfig(threads);
    auto status = cudaLaunchKernel(kernelFn, blockConfig, threadConfig,
                                   kernelArgs, smemBytes, 0);

    if (status != cudaSuccess) {
      auto errorName = cudaGetErrorName(status);
      auto errorStr = cudaGetErrorString(status);
      fprintf(
        stderr,
        "Failed to launch kernelFn: %s: %s\n",
        errorName,
        errorStr
      );

      std::exit(1);
    }
  }

  // Copy result from GMEM to CPU
  std::vector<DataType> gpuResult(M * N, DataType(0));
  CHECK_CUDA_ERROR(
      cudaMemcpy(gpuResult.data(), devPtrC, cNumBytes, cudaMemcpyDeviceToHost));

  return gpuResult;
}

int main(int argc, char **argv) {
  using DataType = float;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 0.1);

  constexpr int M = 512;
  constexpr int N = 1024;
  constexpr int K = 1024;
  constexpr int TILE_SIZE = 32;

  std::vector<float> A(M * K, 0);
  for (int i = 0; i < M * K; i++) A[i] = dist(gen);

  std::vector<float> B(N * K, 0);
  for (int i = 0; i < N * K; i++) B[i] = dist(gen);

  auto cpuResult = cpuGemm(A, B, M, N, K);
  auto gpuResult = gpuGemm<DataType, TILE_SIZE>(A, B, M, N, K);

  // Compare GPU result with CPU result
  DataType epsilon = 1e-2;
  for (int i = 0; i < M * N; i++) {
    if (std::abs(gpuResult[i] - cpuResult[i]) > epsilon) {
      printf("gpuResult[%d] %f != %f cpuResult[%d]\n", i, gpuResult[i],
             cpuResult[i], i);
      printf("FAILURE\n");
      return EXIT_FAILURE;
    }
  }

  printf("SUCCESS\n");
  return EXIT_SUCCESS;
}