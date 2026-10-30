#include <cstdio>
#include <cstdint>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

__global__ void testKernel(const uint32_t n, uint32_t* out) {
  if (threadIdx.x == 0) {
    auto zeros = __clz(n - 1);
    out[0] = 1U << (32 - zeros);
  }
}

__global__ void shflDownKernel(const float* inData, float* outData) {
  // const uint32_t laneIdx = threadIdx.x & 31;
  float v = inData[threadIdx.x];
  for (uint32_t delta = 16; delta > 0; delta >>= 1) {
    v += __shfl_down_sync(0xFFFFFFFF, v, delta);
  }

  if (threadIdx.x == 0) {
    outData[0] = v;
  }
}

__global__ void shflUpKernel(const float* inData, float* outData) {
  // const uint32_t laneIdx = threadIdx.x & 31;
  float v = inData[threadIdx.x];
  for (uint32_t delta = 16; delta > 0; delta >>= 1) {
    v += __shfl_up_sync(0xFFFFFFFF, v, delta);
  }

  if (threadIdx.x == 31) {
    outData[0] = v;
  }
}

int main(int argc, char** argv) {
  uint32_t n = 6;
  if (argc > 1) {
    try {
      n = std::stoi(argv[1]);
      printf("Input n = %d\n", n);
    } catch(const std::exception& ex) {
      printf("Default n = %d\n", n);
    }
  }

  thrust::device_vector<float> hostVec(32, 0);
  for (uint32_t i = 0; i < 32; i++) hostVec[i] = float(i);

  thrust::device_vector<float> devIn = hostVec;

  thrust::device_vector<float> devDownOut(1, 0);
  // testKernel<<<1, 32>>>(n, devOut.data().get());
  shflDownKernel<<<1, 32>>>(devIn.data().get(), devDownOut.data().get());
  cudaDeviceSynchronize();

  thrust::host_vector<float> downOut = devDownOut;
  printf("Down out: %f\n", downOut[0]);

  thrust::device_vector<float> devUpOut(1, 0);
  shflUpKernel<<<1, 32>>>(devIn.data().get(), devUpOut.data().get());
  cudaDeviceSynchronize();

  thrust::host_vector<float> upOut = devUpOut;
  printf("Up out: %f\n", upOut[0]);
}