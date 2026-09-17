#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstdlib>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <int FILTER_SIZE>
thrust::host_vector<float> cpu_kernel(const float* __restrict__ in, const float* __restrict__ weights, const int n) { 
  thrust::host_vector<float> out(n, 0);
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < FILTER_SIZE; k++) {
      if (i + k < n) {
        out[i] += in[i + k] * weights[k];
      }
    }
  }

  return out;
}


template <int TILE_SIZE, int FILTER_SIZE>
__global__ void gpu_kernel_v1(const float* __restrict__ in, const float* __restrict__ weights, float* __restrict__ out, const int n) {
  int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int start_idx = tile_idx * TILE_SIZE;
  
  int i = start_idx;
  for (int t = 0; t < TILE_SIZE; t++) {
    i = start_idx + t;
    if (i < n) {
      float acc = 0;
      for (int k = 0; k < FILTER_SIZE; k++) {
        if (i + k < n) {
          acc += in[i + k] * weights[k];
        }
      }

      out[i] = acc;
    }
  }
}

/*
Reorder loops
  - Swap the innermost loop with the outermost loop
  - weights[k] will be loaded into a register 
  - weights[k] will be re-used TILE_SIZE times

NOTE: Pay attention to the index of acc
*/
template <int TILE_SIZE, int FILTER_SIZE>
__global__ void gpu_kernel_v2(const float* __restrict__ in, const float* __restrict__ weights, float* __restrict__ out, const int n) {
  int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int start_idx = tile_idx * TILE_SIZE;

  float acc[TILE_SIZE];
  #pragma unroll TILE_SIZE
  for (int t = 0; t < TILE_SIZE; t++) {
    acc[t] = 0;
  }

  for (int k = 0; k < FILTER_SIZE; k++) {
    float wk = weights[k];
    for (int t = 0; t < TILE_SIZE; t++) {
      int i = start_idx + t;
      if (i < n && i + k < n) {
        acc[t] += in[i + k] * wk;
      } 
    }
  }

  for (int t = 0; t < TILE_SIZE; t++) {
    int i = start_idx + t;
    if (i < n) {
      out[i] = acc[t];
    }
  }
}

int main() {
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> dist(0.0, 0.1);

  constexpr int N = 1024 * 1024 * 16;
  constexpr int TILE_SIZE = 32;
  constexpr int FILTER_SIZE = 32;
  constexpr float eps = 1e-2;
  thrust::host_vector<float> host_vec1(N, 0);
  for (auto& e : host_vec1) {
    e = dist(gen);
  }

  thrust::host_vector<float> host_vec2(FILTER_SIZE, 0);
  for (auto& e : host_vec2) {
    e = dist(gen);
  }

  printf("Executing cpu_kernel\n");
  thrust::host_vector<float> cpu_out_vec = cpu_kernel<FILTER_SIZE>(host_vec1.data(), host_vec2.data(), N);

  int total_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
  int threads = 512;
  if (total_tiles < threads) {
    threads = total_tiles;
  }

  int blocks = (total_tiles + threads - 1) / threads;
  printf("blocks = %d, threads = %d\n", blocks, threads);

  for (int i = 0; i < 4; i++) {
    printf("\n");
    printf("========================================================\n");

    float t1 = 0;
    printf("Launching gpu_kernel_v1\n");
    {
      thrust::device_vector<float> dev_in_vec1 = host_vec1;
      thrust::device_vector<float> dev_in_vec2 = host_vec2;
      thrust::device_vector<float> dev_out_vec(N, 0);

      cudaEvent_t e0, e1;
      cudaEventCreate(&e0);
      cudaEventCreate(&e1);
      
      cudaEventRecord(e0, 0);
      gpu_kernel_v1<TILE_SIZE, FILTER_SIZE><<<blocks, threads, 0, 0>>>(
        dev_in_vec1.data().get(), dev_in_vec2.data().get(), dev_out_vec.data().get(), N);
      cudaEventRecord(e1, 0);
      cudaDeviceSynchronize();

      cudaEventElapsedTime(&t1, e0, e1);
      printf("Kernel time v1: %f ms\n", t1);

      thrust::host_vector<float> gpu_out_vec = dev_out_vec;
      for (int i = 0; i < N; i++) {
        if (std::abs(gpu_out_vec[i] - cpu_out_vec[i]) > eps) {
          printf("i = %d: GPU value %f != %f CPU value\n", i, gpu_out_vec[i], cpu_out_vec[i]);
          printf("gpu_kernel_v1 FAILED\n");
          return EXIT_FAILURE;
        }
      }

      printf("gpu_kernel_v1 PASSED\n");
    }

    printf("\n");

    float t2 = 0;
    printf("Launching gpu_kernel_v2\n");
    {
      thrust::device_vector<float> dev_in_vec1 = host_vec1;
      thrust::device_vector<float> dev_in_vec2 = host_vec2;
      thrust::device_vector<float> dev_out_vec(N, 0);

      cudaEvent_t e0, e1;
      cudaEventCreate(&e0);
      cudaEventCreate(&e1);
      
      cudaEventRecord(e0, 0);
      gpu_kernel_v2<TILE_SIZE, FILTER_SIZE><<<blocks, threads, 0, 0>>>(
        dev_in_vec1.data().get(), dev_in_vec2.data().get(), dev_out_vec.data().get(), N);
      cudaEventRecord(e1, 0);
      cudaDeviceSynchronize();

      cudaEventElapsedTime(&t2, e0, e1);
      printf("Kernel time v2: %f ms\n", t2);

      thrust::host_vector<float> gpu_out_vec = dev_out_vec;
      for (int i = 0; i < N; i++) {
        if (std::abs(gpu_out_vec[i] - cpu_out_vec[i]) > eps) {
          printf("i = %d: GPU value %f != %f CPU value\n", i, gpu_out_vec[i], cpu_out_vec[i]);
          printf("gpu_kernel_v2 FAILED\n");
          return EXIT_FAILURE;
        }
      }

      printf("gpu_kernel_v2 PASSED\n");
    }

    printf("\n");
    printf("t1 - t2 = %f ms\n", t1 - t2);
    printf("========================================================\n");
  }
}
