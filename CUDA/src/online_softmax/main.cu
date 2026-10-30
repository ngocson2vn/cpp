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
  const uint32_t tid = threadIdx.x;

  // threads must be a multiple of 32
  const uint32_t warpCount = threads >> 5;
  const uint32_t warpIdx = tid >> 5;
  const uint32_t laneIdx = tid & 31;

  auto maxVals = reinterpret_cast<DataType*>(&globalSmem[0]);
  auto sumVals = reinterpret_cast<DataType*>(&globalSmem[0]) + warpCount;

  // Each thread computes its own (m, s)
  const DataType *inRow = inData + blockIdx.x * COLS;

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
  if (tid == 0) {
    printf("P2 = %d, maxVals[0] = %f\n", P2, maxVals[0]);
  }
  __syncwarp();


  // Precondition
  // laneIdx and (laneIdx XOR P2) must load the same values
  // 
  // Prove that (laneIdx XOR P2) % P2 = laneIdx % P2
  //   Case 1: laneIdx < P2
  //     laneIdx XOR P2 = laneIdx + P2
  //       => (laneIdx XOR P2) % P2 = (laneIdx + P2) % P2 = laneIdx % P2 = laneIdx
  //   Case 2: laneIdx == P2
  //     laneIdx XOR P2 = P2 XOR P2 = 0
  //       => (laneIdx XOR P2) % P2 = laneIdx % P2 = 0
  //   Case 3: laneIdx > P2
  //     laneIdx = nP2 + P2 + r (n = 0, 1, 2, ...; r < P2)
  //       => laneIdx XOR P2 = (nP2 + P2 + r) XOR P2 = nP2 + r
  //       => (nP2 + r) % P2 = r
  //     laneIdx % P2 = (nP2 + P2 + r) % P2 = r
  const uint32_t idx = laneIdx & (P2 - 1);
  m = idx < warpCount ? maxVals[idx] : -FLT_MAX;
  s = idx < warpCount ? sumVals[idx] : 0;

  // NOTE: if warpCount == 1
  //   m = maxVals[0] for all threads
  //   s = sumVals[0] for all threads

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
  DataType *outRow = outData + blockIdx.x * COLS;
  DataType invSum = s > DataType(0) ? 1 / s : DataType(0);
  i = tid;
  while (i < COLS) {
    elemVal = inRow[i];
    outRow[i] = exp(elemVal - m) * invSum;
    i += threads;
  }
}

int main(int argc, char **argv) {
  using DataType = float;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  constexpr uint32_t ROWS = 3;
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

  uint32_t threads = (COLS / 32) * 32;
  if (threads > 1024) {
    threads = 1024;
  }

  uint32_t blocks = ROWS;
  printf("blocks = %d threads = %d\n", ROWS, threads);

  uint32_t warpCount = threads / 32;
  uint32_t smemBytes = 2 * warpCount * sizeof(DataType);
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
