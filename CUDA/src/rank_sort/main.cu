#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

extern "C" {

__global__ void rank_sort(const int* __restrict__ in, int* __restrict__ out, int n) {
  int threads = gridDim.x * blockDim.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Apply Thread Coarsening
  // Each thread may handle multiple elements
  for (; i < n; i+=threads) {
    int val = in[i];
    int rank = 0;
    for (int j = 0; j < n; j++) {
      if (in[j] < val || (in[j] == val && j < i)) {
        rank++;
      }
    }

    out[rank] = val;
  }
}

}

int main() {
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<int> dist(0, 10000);

  const int n = 23961;
  thrust::host_vector<int> host_vec(n, 0);
  for (auto& e : host_vec) {
    e = dist(gen);
  }

  if (n <= 1024) {
    printf("Unsorted vec:\n");
    for (auto& e : host_vec) {
      printf("%d ", e);
    }
    printf("\n\n");
  }

  thrust::device_vector<int> dev_in = host_vec;
  thrust::device_vector<int> dev_out(n, 0);

  int threads = 512;
  int blocks = 64;
  cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, rank_sort, 0, 0);
  printf("[PRE] blocks = %d, threads = %d\n", blocks, threads);

  if (n <= threads) {
    threads = n;
    blocks = 1;
  } else if (n < blocks * threads) {
    blocks = (n + threads - 1) / threads;
  }
  printf("[OPT] blocks = %d, threads = %d\n", blocks, threads);

  rank_sort<<<blocks, threads>>>(dev_in.data().get(), dev_out.data().get(), n);
  cudaDeviceSynchronize();

  thrust::host_vector<int> sorted_vec = dev_out;
  if (n <= 1024) {
    printf("Sorted vec:\n");
    for (auto& e : sorted_vec) {
      printf("%d ", e);
    }
    printf("\n");
  }

  thrust::sort(dev_in.begin(), dev_in.end());
  cudaDeviceSynchronize();

  for (int i = 0; i < n; i++) {
    if (dev_out[i] - dev_in[i] != 0) {
      printf("FAILED");
      return EXIT_FAILURE;
    }
  }

  printf("PASSED\n");
}
