#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void simple_sort(const int* in, int* out, const int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < N) {
    int val = in[i];
    int rank = 0;
    for (int j = 0; j < N; j++) {
      if (in[j] < val || (in[j] == val && j < i)) {
        rank++;
      }
    }

    out[rank] = val;

    i += gridDim.x + blockDim.x;
  }
}

int main() {
  constexpr int N = 1024 * 1024 + 5;
  std::vector<int> host_vec(N, 0);

  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<int> dist(0, 10000);

  for (auto& e : host_vec) {
    e = dist(gen);
  }

  int* dev_vec_ptr = nullptr;
  cudaMalloc(&dev_vec_ptr, N * sizeof(int));
  cudaMemcpy(dev_vec_ptr, host_vec.data(), N * sizeof(int), cudaMemcpyHostToDevice);

  int* dev_out_ptr = nullptr;
  cudaMalloc(&dev_out_ptr, N * sizeof(int));

  int heuBlocks = 1;
  int heuThreads = 246;
  cudaOccupancyMaxPotentialBlockSize(&heuBlocks, &heuThreads, simple_sort, 0, 0);

  int threads = std::min(N, heuThreads);
  int blocks = std::min(heuBlocks, (N + threads - 1) / threads);
  printf("blocks = %d threads = %d\n", blocks, threads);

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);

  printf("GPU sorting\n");
  cudaEventRecord(e0, 0);
  simple_sort<<<blocks, threads>>>(dev_vec_ptr, dev_out_ptr, N);
  cudaEventRecord(e1, 0);
  cudaEventSynchronize(e1);

  float kernel_time_ms = 0;
  cudaEventElapsedTime(&kernel_time_ms, e0, e1);
  printf("Kernel time: %f ms\n", kernel_time_ms);

  std::vector<int> gpu_result(N, 0);
  cudaMemcpy(gpu_result.data(), dev_out_ptr, N * sizeof(int), cudaMemcpyDeviceToHost);

  // printf("CPU sorting\n");
  // std::sort(host_vec.begin(), host_vec.end());
  // std::vector<int> cpu_result = std::move(host_vec);
  
  thrust::device_vector<int> dev_vec(N, 0);
  cudaMemcpy(dev_vec.data().get(), host_vec.data(), N * sizeof(int), cudaMemcpyHostToDevice);
  printf("thrust sorting\n");
  thrust::sort(dev_vec.begin(), dev_vec.end());
  thrust::host_vector<int> thrust_result = dev_vec;


  printf("Verifying results\n");
  for (int i = 0; i < N; i++) {
    if (gpu_result[i] != thrust_result[i]) {
      printf("FAILED\n");
      return EXIT_FAILURE;
    }
  }

  printf("PASSED\n");
}
