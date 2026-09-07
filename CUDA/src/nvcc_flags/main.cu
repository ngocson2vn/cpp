#include <cuda_runtime.h>

__device__ void add_vectors(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
  auto idx = threadIdx.x
  c[idx] = a[idx] + b[idx];
}
