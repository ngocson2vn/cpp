#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

#include <cuda_runtime.h>

#include "persistent_gemm.h"

#define KERNEL_VERSION 3

extern "C" {

__device__ float ld_nc_f32(const float* ptr) {
  float out;
  asm volatile(
    "ld.global.nc.f32 %0, [%1];"
    : "=f"(out)
    : "l"(ptr)
  );

  return out;
}

// Each block computes one row of matrix c
__global__ void gemm_v3(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float* __restrict__ c,
                        const uint32_t M,
                        const uint32_t N,
                        const uint32_t K) {
  uint32_t const TOTAL_THREADS = gridDim.x * blockDim.x;
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t row = 0;
  uint32_t col = 0;
  float acc = 0;

  #pragma unroll 1
  while (idx < M * N) {
    row = idx / N;
    col = idx % N;
    acc = 0;

    #pragma unroll 1
    for (uint32_t k = 0; k < K; k++) {
      acc += ld_nc_f32(&a[row * K + k]) * ld_nc_f32(&b[k * N + col]);
    }
    c[idx] = acc;

    idx += TOTAL_THREADS;
  }
}

}

int main(int argc, char **argv) {
  const uint32_t M = 1024;
  const uint32_t N = 512;
  const uint32_t K = 1024 * 32;

  const std::size_t aBytes = M * K * sizeof(float);
  const std::size_t bBytes = K * N * sizeof(float);
  const std::size_t cBytes = M * N * sizeof(float);

  printf("KERNEL_VERSION = %d\n", KERNEL_VERSION);

  auto a = std::vector<float>(M*K, 0);
  auto b = std::vector<float>(K*N, 0);
  auto c = std::vector<float>(M*N, 0);
  auto gpu_res = std::vector<float>(M*N, 0);

  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> dist(0, 0.1);

  for (int i = 0; i < M*K; i++) a[i] = dist(gen);
  for (int i = 0; i < K*N; i++) b[i] = dist(gen);

  // Baseline GEMM
  auto base_res = persistent_gemm(a, b, M, N, K);

  float *dev_a_ptr = nullptr;
  float *dev_b_ptr = nullptr;
  float *dev_c_ptr = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&dev_a_ptr, aBytes));
  CHECK_CUDA_ERROR(cudaMalloc(&dev_b_ptr, bBytes));
  CHECK_CUDA_ERROR(cudaMalloc(&dev_c_ptr, cBytes));

  // Copy inputs from CPU to GPU
  CHECK_CUDA_ERROR(
      cudaMemcpy(dev_a_ptr, a.data(), aBytes, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(dev_b_ptr, b.data(), bBytes, cudaMemcpyHostToDevice));

  int blocks = 1;
  int threads = 256;
  cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, gemm_v3, 0, 0);
  printf("[PRE] blocks = %d threads = %d\n", blocks, threads);

  // Optimize blocks and threads
  if (M*N < threads) {
    blocks = 1;
    threads = M*N;
  } else if (M*N < blocks * threads) {
    blocks = (M*N + threads - 1) / threads;
  }
  printf("[OPT] blocks = %d threads = %d\n", blocks, threads);

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);

  cudaEventRecord(e0, 0);
  gemm_v3<<<blocks, threads, 0, 0>>>(dev_a_ptr, dev_b_ptr, dev_c_ptr, M, N, K);
  cudaEventRecord(e1, 0);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  float elapsed_time_ms;
  cudaEventElapsedTime(&elapsed_time_ms, e0, e1);
  printf("Kernel time: %f ms\n", elapsed_time_ms);

  CHECK_CUDA_ERROR(
      cudaMemcpy(gpu_res.data(), dev_c_ptr, cBytes, cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < M*N; i++) {
    if (std::abs(gpu_res[i] - base_res[i]) > 1e-3) {
      printf("i = %d: GPU value %f != %f CPU value\n", i, gpu_res[i], base_res[i]);
      ok = false;
      break;
    }
  }

  printf("%s\n", (ok ? "PASSED" : "FAILED"));

  cudaFree(dev_a_ptr);
  cudaFree(dev_b_ptr);
  cudaFree(dev_c_ptr);
}
