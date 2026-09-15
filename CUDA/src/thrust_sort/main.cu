#include <cstdio>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/sort.h>

int main(int argc, char** argv) {
  thrust::host_vector<int> h_vector = {30, 23, 2, 34, 1, 4, 6, 9, 41, 39, 13};
  thrust::device_vector<int> d_vector = h_vector;

  thrust::sort(d_vector.begin(), d_vector.end());

  thrust::host_vector<int> result = d_vector;
  printf("Sorted vector:\n");
  for (auto& e : result) {
    printf("%d ", e);
  }
  printf("\n");
}