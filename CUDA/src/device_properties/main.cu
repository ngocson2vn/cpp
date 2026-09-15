#include <cstdio>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
  int device = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  printf("cc = %d.%d, sm = %d\n", prop.major, prop.minor, prop.multiProcessorCount);
}
