#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>


__device__ void stg_s32(int* gmem_ptr, int v) {
  asm volatile(
    "st.global.release.gpu.s32 [%0], %1;"
    :
    : "l"(gmem_ptr), "r"(v)
  );
}

__device__ int ldg_s32(int* gmem_ptr) {
  int out = 0;
  asm volatile(
    "ld.global.acquire.gpu.s32 %0, [%1];"
    : "=r"(out)
    : "l"(gmem_ptr)
  );

  return out;
}

__device__ __forceinline__ void sync_blocks(uint64_t* global_counter_ptr, uint64_t local_counter) {
  uint64_t global_counter = 0;
  while (global_counter < local_counter) {
    asm volatile(
      "atom.global.release.gpu.add.u64 %0, [%1], 0;\n"
      : "=l"(global_counter)
      : "l"(global_counter_ptr)
    );

    __nanosleep(1);
  }
}

__global__ void bitonic_sort_kernel(int* a, const int j, const int k, const int N) {
  const int TOTAL_THREADS = gridDim.x * blockDim.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int p = i ^ j;
  while (i < N && p < N) {
    if (i < p) {
      int ai = a[i];
      int ap = a[p];
      bool ascending = (i & k) == 0;
      if (ascending) {
        if (ai > ap) {
          a[i] = ap;
          a[p] = ai;
        }
      } else {
        if (ap > ai) {
          a[i] = ap;
          a[p] = ai;
        }
      }
    }

    i += TOTAL_THREADS;
    p = i ^ j;
  }
}


int main() {
  int N = 1024;

  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<int> dist(0, 10000);
  
  std::vector<int> host_vec(N, 0);
  for (auto& e : host_vec) {
    e = dist(gen);
  }

  int* dev_vec;
  cudaMalloc(&dev_vec, N * sizeof(int));
  cudaMemcpy(dev_vec, host_vec.data(), N * sizeof(int), cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = 1;
  cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, bitonic_sort_kernel, 0, 0);
  printf("[PRE] blocks=%d threads=%d\n", blocks, threads);

  if (N < threads) {
    blocks = 1;
    threads = N;
  } else if (N < blocks*threads) {
    blocks = (N + threads - 1) / threads;
  }
  printf("[OPT] blocks=%d threads=%d\n", blocks, threads);


  for (int k = 2; k <= N; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      bitonic_sort_kernel<<<blocks, threads, 0, 0>>>(dev_vec, j, k, N);
      cudaDeviceSynchronize();
    }
  }

  std::vector<int> gpu_vec(N, 0);
  cudaMemcpy(gpu_vec.data(), dev_vec, N * sizeof(int), cudaMemcpyDeviceToHost);

  for (auto& e : gpu_vec) {
    printf("%d ", e);
  }
  printf("\n");

  std::sort(host_vec.begin(), host_vec.end());
  for (int i = 0; i < N; i++) {
    if (gpu_vec[i] != host_vec[i]) {
      printf("FAILED\n");
      return EXIT_FAILURE;
    }
  }

  printf("PASSED\n");
}
