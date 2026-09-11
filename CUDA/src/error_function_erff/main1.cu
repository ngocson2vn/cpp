#include <vector>
#include <random>

#include <cuda_runtime.h>

extern "C" {

__global__ void gelu(const float* __restrict__ input, float* __restrict__ output, int length) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < length) {
    float elem = input[tid];
    float res = 0.5f * elem * (1.0f + erff(elem * 0.7071067811f));
    output[tid] = res;
  }
}

}

int main(int argc, char** argv) {
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  int length = 1024;
  std::vector<float> x(length, 0.0f);
  for (int i = 0; i < length; i++) {
    x[i] = dist(gen);
  }

  float* dev_inp_ptr = nullptr;
  float* dev_out_ptr = nullptr;
  cudaMalloc(&dev_inp_ptr, length * sizeof(float));
  cudaMalloc(&dev_out_ptr, length * sizeof(float));
  cudaMemcpy(dev_inp_ptr, x.data(), length * sizeof(float), cudaMemcpyHostToDevice);

  auto threads = 512;
  auto blocks = (length + threads - 1) / threads;
  gelu<<<blocks, threads>>>(dev_inp_ptr, dev_out_ptr, length);
  cudaDeviceSynchronize();
}
