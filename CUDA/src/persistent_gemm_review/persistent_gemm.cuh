#pragma once

#include <cstdint>
#include <cstdlib>

#include <common/common.h>

using u32 = uint32_t;

template <u32 TILE_SIZE>
__global__ void persistentGemm(const float* A,
                               const float* B,
                                     float* C,
                               const u32 M,
                               const u32 N,
                               const u32 K,
                               const u32 tileCountN,
                               const u32 tileCountK,
                               const u32 totalTiles) {
  static_assert(TILE_SIZE <= 32, "TILE_SIZE must be <= 32");
  
  // Global SMEM
  extern __shared__ char globalSmem[];

  // Tile A and B
  // Each tile has TILE_SIZE * TILE_SIZE elements
  auto sA = reinterpret_cast<float*>(globalSmem);
  auto sB = sA + TILE_SIZE * TILE_SIZE;

  const u32 blocks = gridDim.x;

  // Current tile index
  u32 tileIdx = blockIdx.x;

  // Each CTA may compute several tiles
  while (tileIdx < totalTiles) {
    // Compute coordinate (tileM, tileN) of the current tile
    u32 tileM = tileIdx / tileCountN;
    u32 tileN = tileIdx % tileCountN;

    // Compute local indices
    // Each thread is responsible for one element of the current tile
    u32 lm = threadIdx.x / TILE_SIZE;
    u32 ln = threadIdx.x % TILE_SIZE;

    // Compute global indices
    u32 m = tileM * TILE_SIZE + lm;
    u32 n = tileN * TILE_SIZE + ln;

    // Accumulator of the current element
    float acc = 0;

    // Loop over tiles along K dimension
    u32 dk = 0;
    u32 kA = 0;
    u32 kB = 0;
    for (u32 tileK = 0; tileK < tileCountK; tileK++) {
      // Compute k indices for A and B
      dk = tileK * TILE_SIZE;
      kA = dk + ln;
      kB = dk + lm;

      // Load sA and sB
      // Each thread loads one element from A and one element from B
      sA[threadIdx.x] = (m < M && kA < K) ? A[m * K + kA] : 0;
      sB[threadIdx.x] = (n < N && kB < K) ? B[kB * N + n] : 0;

      // Ensure that sA and sB are fully loaded
      __syncthreads();

      // Compute dot product sA @ sB
      for (u32 lk = 0; lk < TILE_SIZE; lk++) {
        acc += sA[lm * TILE_SIZE + lk] * sB[lk * TILE_SIZE + ln];
      }

      // Ensure that dot products are done
      __syncthreads();
    }

    // Store acc back to GMEM
    if (m < M && n < N) {
      C[m * N + n] = acc;
    }

    // Move to next tile
    tileIdx += blocks;
  }
}

void checkCudaError(cudaError_t err) {
  if (err != cudaSuccess) {
    auto errName = cudaGetErrorName(err);
    auto errString = cudaGetErrorString(err);
    fprintf(stderr, "%s: %s\n", errName, errString);
    std::exit(1);
  }
}

HVec<float> gpuGemm(const HVec<float>& A,
                    const HVec<float>& B,
                    const uint32_t M,
                    const uint32_t N,
                    const uint32_t K) {
  printf("\nLaunching GPU GEMM\n");
  
  constexpr u32 TILE_SIZE = 32;
  const u32 tileCountM = (M + TILE_SIZE - 1) / TILE_SIZE;
  const u32 tileCountN = (N + TILE_SIZE - 1) / TILE_SIZE;
  const u32 tileCountK = (K + TILE_SIZE - 1) / TILE_SIZE;
  const u32 totalTiles = tileCountM * tileCountN;

  // Get SM count
  u32 smCount = 64;
  cudaDeviceProp devProp;
  auto err = cudaGetDeviceProperties(&devProp, 0);
  checkCudaError(err);
  smCount = devProp.multiProcessorCount;

  // Compute threads and blocks
  constexpr int threads = TILE_SIZE * TILE_SIZE;
  static_assert(threads <= 1024, "threads must be <= 1024");

  // SMEM bytes
  u32 smemBytes = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

  int blocksPerSM = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocksPerSM,
    persistentGemm<TILE_SIZE>,
    threads,
    smemBytes
  );

  int blocks = smCount * blocksPerSM;
  printf("blocks = %d threads = %d TILE_SIZE = %d\n", blocks, threads, TILE_SIZE);

  // Allocate C
  DVec<float> dC(M * N, 0);

  // Launch kernel
  persistentGemm<TILE_SIZE><<<blocks, threads, smemBytes>>>(
    A.gpu().data(),
    B.gpu().data(),
    dC.data(),
    M, N, K,
    tileCountN,
    tileCountK,
    totalTiles
  );

  err = cudaGetLastError();
  checkCudaError(err);

  err = cudaDeviceSynchronize();
  checkCudaError(err);

  return dC.cpu();
}
