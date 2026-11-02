#include <cstdio>
#include <cstdint>
#include <cfloat>
#include <random>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <common/common.h>

#include "constants.h"

__device__ uint32_t nextPowerOf2(uint32_t n) {
  auto zeros = __clz(n - 1);
  return 1U << (32 - zeros);
}

template <typename DataType>
__global__ void rmsNormKernel(DataType* outData,
                              const DataType* inData,
                              const DataType* w,
                              uint32_t ROWS,
                              uint32_t COLS) {
  extern __shared__ char globalSmem[];
  DataType* sumVals = reinterpret_cast<DataType*>(globalSmem);

  const uint32_t blocks = gridDim.x;
  const uint32_t threads = blockDim.x;
  const uint32_t warpCount = threads >> 5;
  const uint32_t warpIdx = threadIdx.x >> 5;
  const uint32_t laneIdx = threadIdx.x & 31;

  uint32_t rowIdx = blockIdx.x;
  while (rowIdx < ROWS) {
    const DataType* inRow  = inData  + rowIdx * COLS;
    /***/ DataType* outRow = outData + rowIdx * COLS;

    uint32_t i = threadIdx.x;
    DataType s = DataType(0);
    while (i < COLS) {
      s += inRow[i] * inRow[i];
      i += threads;
    }

    // Warp-level reduction
    for (uint32_t laneMask = 16; laneMask > 0; laneMask >>= 1) {
      s += __shfl_xor_sync(0xFFFFFFFF, s, laneMask);
    }

    if (laneIdx == 0) {
      // Since division is an expensive op, do it here.
      // So later threads do not need to do it
      sumVals[warpIdx] = s / COLS;
    }

    // Ensure that all threads see the same image of sumVals
    __syncthreads();

    // Every warp computes the final sum
    const uint32_t P2 = nextPowerOf2(warpCount);
    const uint32_t idx = laneIdx & (P2 - 1);
    DataType ms = idx < warpCount ? sumVals[idx] : 0;
    for (uint32_t laneMask = P2 >> 1; laneMask > 0; laneMask >>= 1) {
      ms += __shfl_xor_sync(0xFFFFFFFF, ms, laneMask);
    }

    // At this point, every thread has the same final sum

    // Compute RMS
    DataType eps = DataType(0);
    DataType rms = sqrtf(ms + eps);
    
    // Normalize
    DataType invRms = 1 / rms;
    i = threadIdx.x;
    while (i < COLS) {
      outRow[i] = inRow[i] * invRms * w[i];
      i += threads;
    }

    rowIdx += blocks;
  }
}

int main(int argc, char** argv) {
  if (argc > 2) {
    try {
      ROWS = std::stoi(argv[1]);
      COLS = std::stoi(argv[2]);
    } catch(const std::exception& ex) {
      printf("Default ROWS = %d COLS = %d\n", ROWS, COLS);
    }
  }
  printf("ROWS = %d COLS = %d\n", ROWS, COLS);

  // const uint32_t numBytes = ROWS * COLS * sizeof(DataType);

  assert(COLS >= 32 && "Row size must be >= 32!");

  // Copy inputs
  DVec<DataType> input = genDVec<DataType>(ROWS * COLS, "input");
  DVec<DataType> gamma = genDVec<DataType>(COLS, "gamma");

  // Allocate output
  DVec<DataType> devOut(ROWS * COLS, DataType(0));


  // 
  // Compute optimal blocks and threads
  // 

  // Get SM count
  uint32_t smCount = 64;
  cudaDeviceProp devProp;
  auto status = cudaGetDeviceProperties(&devProp, 0);
  if (status == cudaSuccess) {
    smCount = devProp.multiProcessorCount;
  }

  // Compute optimal threads
  uint32_t threads = (COLS >> 5) * 32;
  threads = (threads > 1024) ? 1024 : threads;

  // Compute optimal blocks
  int heuBlocks = 1;
  const uint32_t smemBytes = (threads >> 5) * sizeof(DataType);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &heuBlocks, 
    rmsNormKernel<DataType>,
    threads,
    smemBytes
  );
  uint32_t blocks = smCount * heuBlocks;
  blocks = (blocks > ROWS) ? ROWS : blocks;

  printf("blocks = %d threads = %d warps = %d\n", blocks, threads, threads >> 5);

  rmsNormKernel<<<blocks, threads, smemBytes>>>(
    devOut.data(),
    input.data(),
    gamma.data(),
    ROWS,
    COLS
  );

  status = cudaGetLastError();
  if (status != cudaSuccess) {
    auto errorName = cudaGetErrorName(status);
    auto errorString = cudaGetErrorString(status);
    fprintf(stderr, "%s: %s\n", errorName, errorString);
  }

  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    auto errorName = cudaGetErrorName(status);
    auto errorString = cudaGetErrorString(status);
    fprintf(stderr, "%s: %s\n", errorName, errorString);
  }

  HVec<DataType> hostOut = devOut.cpu();

  printf("\n[SONY] RMSNorm output:\n");
  for (uint32_t i = 0; i < ROWS; i++) {
    printf("[%f", hostOut[i * COLS + 0]);
    for (uint32_t j = 1; j < COLS; j++) {
      printf(", %f", hostOut[i * COLS + j]);
    }
    printf("]\n\n");
  }


  // Compare with PyTorch

}
