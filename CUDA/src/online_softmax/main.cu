#include <cstdio>
#include <cassert>
#include <cfloat>

#include <random>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(callExpr)                                             \
  do {                                                                         \
    auto status = callExpr;                                                    \
    if (status != cudaSuccess) {                                               \
      auto errorName = cudaGetErrorName(status);                               \
      auto errorStr = cudaGetErrorString(status);                              \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, errorName,         \
              errorStr);                                                       \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__device__ __forceinline__ uint32_t nextPowerOf2(const uint32_t n) {
  auto zeros = __clz(n - 1);
  return 1U << (32 - zeros);
}

template <typename DataType>
__global__ void softmaxKernel(const DataType *inData, DataType *outData,
                              const uint32_t ROWS, const uint32_t COLS) {
  extern __shared__ char globalSmem[];

  // blockDim = (blockDim.x, 1, 1)
  const uint32_t threads = blockDim.x;
  const uint32_t blocks = gridDim.x;
  const uint32_t tid = threadIdx.x;

  // threads must be a multiple of 32
  const uint32_t warpCount = threads >> 5;
  const uint32_t warpIdx = tid >> 5;
  const uint32_t laneIdx = tid & 31;

  auto maxVals = reinterpret_cast<DataType*>(&globalSmem[0]);
  auto sumVals = reinterpret_cast<DataType*>(&globalSmem[0]) + warpCount;

  // Persistent CTA
  uint32_t rowIdx = blockIdx.x;
  while (rowIdx < ROWS) {
    // Each thread computes its own (m, s)
    const DataType* inRow = inData + rowIdx * COLS;
    DataType* outRow = outData + rowIdx * COLS;

    // It depends on COLS, each thread might 
    // read more than 1 elements of a row
    uint32_t i = tid;
    DataType m = -FLT_MAX;
    DataType newM = 0;
    DataType s = 0;
    DataType elemVal = DataType(0);
    while (i < COLS) {
      elemVal = inRow[i];
      newM = max(m, elemVal);
      s = s * exp(m - newM) + exp(elemVal - newM);
      m = newM;

      i += threads;
    }

    // Warp-level reduction
    // Reduce (m, s) within each warp
    for (uint32_t laneMask = 16; laneMask > 0; laneMask >>= 1) {
      DataType otherM = __shfl_xor_sync(0xFFFFFFFF, m, laneMask);
      DataType otherS = __shfl_xor_sync(0xFFFFFFFF, s, laneMask);
      newM = max(m, otherM);
      s = s * exp(m - newM) + otherS * exp(otherM - newM);
      m = newM;
    }

    if (laneIdx == 0) {
      maxVals[warpIdx] = m;
      sumVals[warpIdx] = s;
    }

    // Ensure that all warps have finished writing its (m, s) to SMEM
    __syncthreads();

    // Now that every thread in every warp has the same (m, s)

    // Reduce the final (m, s)
    const uint32_t P2 = nextPowerOf2(warpCount);
    if (tid == 0 && blockIdx.x == 0) {
      printf("P2 = %d, maxVals[0] = %f\n", P2, maxVals[0]);
    }
    __syncwarp();


    // Precondition
    // Since P2 divides a warp of 32 lanes evenly into groups of P2 lanes.
    //   P2  = 1 => 32 groups of  1 lane
    //   P2  = 2 => 16 groups of  2 lanes
    //   P2  = 4 =>  8 groups of  4 lanes
    //   P2  = 8 =>  4 groups of  8 lanes
    //   P2 = 16 =>  2 groups of 16 lanes
    //   P2 = 32 =>  1 group  of 32 lanes
    // Each group must load the same set of (m, s) pairs
    // For example P2 = 16, then a warp is divided into 2 groups. Each group has 16 lanes:
    //                                                                                 |
    //    0    1    2    3    4    5    6    7   8     9   10   11   12   13   14   15 | 16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31
    // |----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
    // |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
    // |----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
    //                                                                                 |
    //                                                                                 |
    // laneIdx & (P2 - 1) will produce the same set of indices for each group
    const uint32_t idx = laneIdx & (P2 - 1);

    // If idx >= warpCount, pad a pair of (-FLT_MAX, 0)
    // which will introduce nothing to the final (m, s)
    m = idx < warpCount ? maxVals[idx] : -FLT_MAX;
    s = idx < warpCount ? sumVals[idx] : 0;

    // NOTE: if warpCount == 1
    //   m = maxVals[0] for all threads
    //   s = sumVals[0] for all threads

    // laneMask divides each group into two group evenly 
    // until each individual thread holds the same pair (m, s)
    for (uint32_t laneMask = P2 >> 1; laneMask > 0; laneMask >>= 1) {
      DataType otherM = __shfl_xor_sync(0xFFFFFFFF, m, laneMask);
      DataType otherS = __shfl_xor_sync(0xFFFFFFFF, s, laneMask);
      newM = max(m, otherM);
      s = s * exp(m - newM) + otherS * exp(otherM - newM);
      m = newM;
    }

    // Now that every thread in every warp has the same final (m, s)

    // Compute softmax for each element inData inRow
    // NOTE: if results are `inf`, probably `s` is zero
    DataType invSum = s > DataType(0) ? 1 / s : DataType(0);
    i = tid;
    while (i < COLS) {
      elemVal = inRow[i];
      outRow[i] = exp(elemVal - m) * invSum;
      i += threads;
    }

    // Move to next row
    rowIdx += blocks;
  }
}

int main(int argc, char **argv) {
  using DataType = float;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  constexpr uint32_t ROWS = 1024;
  uint32_t COLS = 33;
  if (argc > 1) {
    COLS = std::stoi(argv[1]);
  }

  const uint32_t numBytes = ROWS * COLS * sizeof(DataType);

  assert(COLS >= 32 && "Row size must be >= 32!");

  std::vector<float> inData(ROWS * COLS, 0);
  for (int i = 0; i < ROWS * COLS; i++)
    inData[i] = dist(gen);

  DataType *devIn, *devOut;
  cudaMalloc(&devIn, numBytes);
  cudaMalloc(&devOut, numBytes);

  cudaMemcpy(devIn, inData.data(), numBytes, cudaMemcpyHostToDevice);

  int threads = (COLS / 32) * 32;
  if (threads > 1024) {
    threads = 1024;
  }

  uint32_t warpCount = threads / 32;
  uint32_t smemBytes = 2 * warpCount * sizeof(DataType);

  // Get heuristic blocks
  int blocksPerSM = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocksPerSM, 
    softmaxKernel<DataType>, 
    threads, 
    smemBytes
  );

  // Get SM count
  uint32_t smCount = 64;
  {
    cudaDeviceProp devProp;
    auto status = cudaGetDeviceProperties(&devProp, 0);
    if (status == cudaSuccess) {
      smCount = devProp.multiProcessorCount;
    }
  }

  uint32_t heuBlocks = smCount * blocksPerSM;
  uint32_t blocks = (heuBlocks > ROWS) ? ROWS : heuBlocks;
  printf("blocks = %d threads = %d\n", ROWS, threads);


  softmaxKernel<DataType><<<blocks, threads, smemBytes>>>(devIn, devOut, ROWS, COLS);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  std::vector<DataType> outData(ROWS * COLS, 0);
  cudaMemcpy(outData.data(), devOut, numBytes, cudaMemcpyDeviceToHost);

  // printf("Input tensor:\n");
  // for (uint32_t i = 0; i < ROWS; i++) {
  //   // printf("Row %d: ", i);
  //   printf("[%.3f", inData[i * COLS + 0]);
  //   for (uint32_t j = 1; j < COLS; j++) {
  //     printf(", %.3f", inData[i * COLS + j]);
  //   }
  //   printf("]\n\n");
  // }

  printf("\nOutput tensor:\n");
  for (uint32_t i = 0; i < ROWS; i++) {
    // printf("Row %d: ", i);
    printf("[%f", outData[i * COLS + 0]);
    for (uint32_t j = 1; j < COLS; j++) {
      printf(", %f", outData[i * COLS + j]);
    }
    printf("]\n\n");
  }
}
