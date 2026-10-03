#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(apiCall)                                              \
  do {                                                                         \
    cudaError_t error = (apiCall);                                             \
    if (error != cudaSuccess) {                                                \
      auto errorName = cudaGetErrorName(error);                                \
      auto errorString = cudaGetErrorString(error);                            \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, errorName,         \
              errorString);                                                    \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

__global__ void repeat_kernel(const float* in, float* out, int NROWS, int NCOLS) {
  const int TOTAL_THREADS = gridDim.x * blockDim.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  while (i < NROWS * NCOLS) {
    out[i] = in[i % NCOLS];
    i += TOTAL_THREADS;
  }
}


int main(int argc, char** argv) {
  constexpr int NCOLS = 32;

  int NROWS = 1024 * 1024 * 32;
  if (argc > 1) {
    NROWS = std::stoi(argv[1]);
  }
  printf("NROWS = %d\n", NROWS);
  
  std::vector<float> host_in(NCOLS, 0);
  for (int i = 0; i < NCOLS; i++) {
    host_in[i] = float(i);
  }

  float* dev_in_ptr;
  cudaMalloc(&dev_in_ptr, NCOLS * sizeof(float));
  cudaMemcpy(dev_in_ptr, host_in.data(), NCOLS * sizeof(float), cudaMemcpyHostToDevice);

  float* dev_out_ptr;
  cudaMalloc(&dev_out_ptr, NROWS * NCOLS * sizeof(float));

  int threads = 256;
  int blocks = 1;
  cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, repeat_kernel, 0, 0);
  printf("[PRE] blocks=%d threads=%d\n", blocks, threads);


  const int N = NROWS * NCOLS;
  threads = (N < threads) ? N : threads;
  blocks = (N < blocks * threads) ? (N + threads - 1) / threads : blocks;
  printf("[OPT] blocks=%d threads=%d\n", blocks, threads);

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0, 0);
  repeat_kernel<<<blocks, threads, 0, 0>>>(dev_in_ptr, dev_out_ptr, NROWS, NCOLS);
  cudaEventRecord(e1, 0);
  CHECK_CUDA_ERROR(cudaEventSynchronize(e1));
  
  float kernel_time_ms;
  cudaEventElapsedTime(&kernel_time_ms, e0, e1);
  printf("Kernel time: %f ms\n\n", kernel_time_ms);

  constexpr int N_OUT_ROWS = 10;
  std::vector<float> host_out(N_OUT_ROWS, 0);
  auto dev_ptr = dev_out_ptr + (NROWS - N_OUT_ROWS) * NCOLS;
  cudaMemcpy(host_out.data(), dev_ptr, N_OUT_ROWS * NCOLS * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N_OUT_ROWS; i++) {
    printf("i = %04d: ", (NROWS - N_OUT_ROWS) + i);
    for (int j = 0; j < NCOLS; j++) {
      printf("%4.1f ", host_out[i * NCOLS + j]);
    }
    printf("\n");
  }
  printf("\n");
}
