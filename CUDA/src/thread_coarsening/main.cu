#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

thrust::host_vector<float> cpu_add_vectors(const float* __restrict__ a, const float* __restrict__ b, const int N) {
  thrust::host_vector<float> out(N, 0);
  for (int i = 0; i < N; i++) {
    out[i] = a[i] + b[i];
  }

  return out;
}

__global__ void gpu_add_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, const int N) {
  // Each block covers `blockDim.x` elements,
  // so idx must be strided by `blockDim.x`
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Total threads in the grid
  int stride = gridDim.x * blockDim.x;

  // idx must be strided by total threads in the grid
  // so that each thread will compute an unique idx
  for (; idx < N; idx += stride) {
    out[idx] = a[idx] + b [idx];
  }
}

int main() {
  constexpr int N = 1024 * 1024 * 16;

  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> dist(0.0, 0.1);

  thrust::host_vector<float> host_vec1(N, 0);
  for (auto& e : host_vec1) {
    e = dist(gen);
  }

  thrust::host_vector<float> host_vec2(N, 0);
  for (auto& e : host_vec2) {
    e = dist(gen);
  }

  auto cpu_result = cpu_add_vectors(host_vec1.data(), host_vec2.data(), N);

  thrust::device_vector<float> dev_vec1 = host_vec1;
  thrust::device_vector<float> dev_vec2 = host_vec2;
  thrust::device_vector<float> dev_out(N, 0);

  int blocks = 1;
  int threads = 256;
  cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, gpu_add_vectors, 0, 0);
  printf("[PRE] blocks = %d, threads = %d\n", blocks, threads);
  threads = std::min(N, threads);
  blocks = std::min(blocks, (N + threads - 1) / threads);
  printf("[OPT] blocks = %d, threads = %d\n", blocks, threads);

  gpu_add_vectors<<<blocks, threads, 0, 0>>>(dev_vec1.data().get(), dev_vec2.data().get(), dev_out.data().get(), N);
  cudaDeviceSynchronize();

  thrust::host_vector<float> gpu_result = dev_out;

  for (int i = 0; i < N; i++) {
    if (std::abs(gpu_result[i] - cpu_result[i]) > 1e-2) {
      printf("i = %d: GPU value %f != %f CPU value\n", i, gpu_result[i], cpu_result[i]);
      return EXIT_FAILURE;
    }
  }

  printf("PASSED\n");
}