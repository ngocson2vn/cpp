#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>


#include <cuda_runtime.h>

__device__ int ldg_s32(int* ptr) {
  int out = 0;
  asm volatile(
    "ld.global.acquire.gpu.s32 %0, [%1];"
    : "=r"(out)
    : "l"(ptr)
  );

  return out;
}

__device__ void sync_blocks(uint64_t* global_counter_ptr, uint64_t& local_counter) {
  // Increase local counter
  local_counter += gridDim.x * blockDim.x;

  // Increase global counter
  asm volatile(
    "{\n"
    "\t\t.reg .u64 %tmp;\n"
    "\t\tatom.global.release.gpu.add.u64 %tmp, [%0], 1;\n"
    "\t}"
    :
    : "l"(global_counter_ptr)
  );

  // Read global counter
  uint64_t global_counter = 0;
  do {
    __nanosleep(1);
    asm volatile(
      "ld.global.acquire.gpu.u64 %0, [%1];"
      : "=l"(global_counter)
      : "l"(global_counter_ptr)
    );
  } while (global_counter < local_counter);
}

__global__ void some_kernel(int* a, const int N, uint64_t* global_counter_ptr) {
  uint64_t local_counter = 0;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  for (int k = 2; k <= N; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      int p = i ^ j;
      bool ascending = (i & k) == 0;

      // Only allow the left thread to do swapping
      if (i < p && i < N && p < N) {
        auto ai = ldg_s32(&a[i]);
        auto ap = ldg_s32(&a[p]);
        if (ascending) {
          if (ai > ap) {
            a[i] = ap;
            a[p] = ai;
          }
        } else { // descending
          if (ai < ap) {
            a[i] = ap;
            a[p] = ai;
          }
        }
      }

      sync_blocks(global_counter_ptr, local_counter);
    }
  }
}

int get_sm_count() {
  cudaDeviceProp prop;
  auto errorCode = cudaGetDeviceProperties(&prop, 0);
  if (errorCode != cudaSuccess) {
    return 64;
  }

  return prop.multiProcessorCount;
}

int get_max_grid_x() {
  cudaDeviceProp prop;
  auto errorCode = cudaGetDeviceProperties(&prop, 0);
  if (errorCode != cudaSuccess) {
    return ((1U << 31) - 1);
  }

  return prop.maxGridSize[0];
}


int main() {
  // N must be power of 2
  constexpr int N = 1 << 16;

  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<int> dist(0, 10000);
  
  std::vector<int> host_vec(N, 0);
  for (auto& e : host_vec) {
    e = dist(gen);
  }

  int* dev_vec_ptr = nullptr;
  cudaMalloc(&dev_vec_ptr, N * sizeof(int));
  cudaMemcpy(dev_vec_ptr, host_vec.data(), N * sizeof(int), cudaMemcpyHostToDevice);
  
  uint64_t* global_counter_ptr = nullptr;
  cudaMalloc(&global_counter_ptr, sizeof(uint64_t));
  uint64_t init_val = 0;
  cudaMemcpy(global_counter_ptr, &init_val, sizeof(uint64_t), cudaMemcpyHostToDevice);

  int maxBlocks = get_max_grid_x();

  int heuBlocks = 1;
  int heuThreads = 256;
  cudaOccupancyMaxPotentialBlockSize(&heuBlocks, &heuThreads, some_kernel, 0, 0);

  int threads = std::min(N, heuThreads);
  int blocks = (uint64_t(N) + threads - 1) / threads;
  if (blocks > maxBlocks) {
    printf("blocks = %d > %d max blocks\n", blocks, maxBlocks);
    return EXIT_FAILURE;
  }

  printf("blocks = %d threads = %d\n", blocks, threads);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0, 0);
  some_kernel<<<blocks, threads, 0, 0>>>(dev_vec_ptr, N, global_counter_ptr);
  cudaEventRecord(e1, 0);
  cudaEventSynchronize(e1);

  float elapsed_time_ms;
  cudaEventElapsedTime(&elapsed_time_ms, e0, e1);
  printf("Kernel time: %f ms\n", elapsed_time_ms);

  std::vector<int> gpu_result(N, 0);
  cudaMemcpy(gpu_result.data(), dev_vec_ptr, N * sizeof(int), cudaMemcpyDeviceToHost);

  std::sort(host_vec.begin(), host_vec.end());
  std::vector<int> cpu_result = std::move(host_vec);

  for (int i = 0; i < N; i++) {
    if (gpu_result[i] != cpu_result[i]) {
      printf("FAILED\n");
      return EXIT_FAILURE;
    }
  }

  printf("PASSED\n");
}
