#include <cstdio>
#include <random>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

extern "C" {

__global__ void rank_sort(const int* __restrict__ in, int* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  int val = in[i];
  int rank = 0;

  #pragma unroll
  for (int j = 0; j < n; j++) {
    auto& v = in[j];

    // Count how many elements sit before element i;
    // rank must be unique
    if (v < val || (v == val && j < i)) {
      rank++;
    }
  }

  out[rank] = val;
}

}

int main() {
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<int> dist(0, 10000);

  const int n = 4000;
  thrust::host_vector<int> host_vec(n, 0);
  for (auto& e : host_vec) {
    e = dist(gen);
  }

  printf("Unsorted vec:\n");
  for (auto& e : host_vec) {
    printf("%d ", e);
  }
  printf("\n\n");

  thrust::device_vector<int> dev_in_vec = host_vec;
  thrust::device_vector<int> dev_out_vec(n, 0);

  int threads = 512;
  if (n <= 512) {
    threads = n;
  }
  int blocks = (n + threads - 1) / threads;
  printf("blocks = %d, threads = %d\n", blocks, threads);

  rank_sort<<<blocks, threads>>>(dev_in_vec.data().get(), dev_out_vec.data().get(), n);
  cudaDeviceSynchronize();

  printf("Sorted vec:\n");
  thrust::host_vector<int> sorted_vec = dev_out_vec;
  for (auto& e : sorted_vec) {
    printf("%d ", e);
  }
  printf("\n");
}
