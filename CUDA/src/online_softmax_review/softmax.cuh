#include <common/common.h>
#include <cfloat>
#include <cstdlib>

using u32 = uint32_t;

__device__ u32 nextPowerOf2(u32 n) {
  auto zeros = __clz(n - 1);
  return 1U << (32 - zeros);
}

// y = exp(x - max) / sum(exp(x - max))
__global__ void softmaxKernel(float* inData,
                              float* outData,
                              const u32 M,
                              const u32 N) {
  extern __shared__ char globalSmem[];
  
  const u32 threads = blockDim.x;
  const u32 warpCount = threads >> 5;
  const u32 warpIdx = threadIdx.x >> 5;
  const u32 laneIdx = threadIdx.x & 31;

  auto maxVals = reinterpret_cast<float*>(globalSmem);
  auto sumVals = maxVals + warpCount;

  const u32 blocks = gridDim.x;

  u32 rowIdx = blockIdx.x;
  while (rowIdx < M) {
    float* inRow = inData + rowIdx * N;
    float* outRow = outData + rowIdx * N;

    // Each thread compute a pair of (m, s)
    u32 i = threadIdx.x;
    float m = -FLT_MAX;
    float s = 0;
    while (i < N) {
      float x = inRow[i];
      float newM = max(m, x);
      s = s * exp(m - newM) + exp(x - newM);
      m = newM;
      i += threads;
    }

    // Reduce (m, s) within every warp
    for (u32 laneMask = 16; laneMask > 0; laneMask >>= 1) {
      float otherM = __shfl_xor_sync(0xFFFFFFFF, m, laneMask);
      float otherS = __shfl_xor_sync(0xFFFFFFFF, s, laneMask);
      float newM = max(m, otherM);
      s = s * exp(m - newM) + otherS * exp(otherM - newM);
      m = newM;
    }

    // Save (m, s) to SMEM
    if (laneIdx == 0) {
      maxVals[warpIdx] = m;
      sumVals[warpIdx] = s;
    }

    // Ensure that all warps have finished
    __syncthreads();

    // Every warp reduces the same final (m, s)
    u32 P2 = nextPowerOf2(warpCount);
    u32 idx = laneIdx % P2;
    m = idx < warpCount ? maxVals[idx] : -FLT_MAX;
    s = idx < warpCount ? sumVals[idx] : 0;
    for (u32 laneMask = P2 >> 1; laneMask > 0; laneMask >>= 1) {
      float otherM = __shfl_xor_sync(0xFFFFFFFF, m, laneMask);
      float otherS = __shfl_xor_sync(0xFFFFFFFF, s, laneMask);
      float newM = max(m, otherM);
      s = s * exp(m - newM) + otherS * exp(otherM - newM);
      m = newM;
    }

    // Now that every thread has the same final (m, s)
    // Compute softmax for each element
    float invSum = s > 0 ? 1 / s : 0;
    i = threadIdx.x;
    while (i < N) {
      outRow[i] = exp(inRow[i] - m) * invSum;
      i += threads;
    }

    // Move to next row
    rowIdx += blocks;
  }
}

HVec<float> gpuSoftmax(const HVec<float>& hInp, const u32 M, const u32 N) {
  printf("\nLaunching GPU softmax\n");

  // Compute SM count
  u32 smCount = 64;
  cudaDeviceProp devProp;
  auto status = cudaGetDeviceProperties(&devProp, 0);
  if (status == cudaSuccess) {
    smCount = devProp.multiProcessorCount;
  }
  printf("SM count: %d\n", smCount);

  // Compute threads and blocks
  int threads = (N >> 5) * 32;
  threads = (threads > 1024) ? 1024 : threads;

  u32 smemBytes = 2 * (threads >> 5) * sizeof(float);

  int blocksPerSM = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocksPerSM,
    softmaxKernel,
    threads,
    smemBytes
  );
  int blocks = smCount * blocksPerSM;
  printf("blocks = %d threads = %d warpCount = %d\n",
                        blocks, threads, threads >> 5);

  DVec<float> dOut(M * N, 0);

  // Launch kernel
  softmaxKernel<<<blocks, threads, smemBytes>>>(
    hInp.gpu().data(),
    dOut.data(),
    M,
    N
  );

  status = cudaGetLastError();
  if (status != cudaSuccess) {
    auto errName = cudaGetErrorName(status);
    auto errString = cudaGetErrorString(status);
    fprintf(stderr, "%s: %s\n", errName, errString);
    std::exit(1);
  }

  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    auto errName = cudaGetErrorName(status);
    auto errString = cudaGetErrorString(status);
    fprintf(stderr, "%s: %s\n", errName, errString);
    std::exit(1);
  }

  return dOut.cpu();
}