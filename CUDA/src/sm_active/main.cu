/* Launch a CUDA kernel that utilizes all available SMs

 */

#include "helper_cuda.h"
#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <vector>

#include "gpu_monitor.h"

#define LOG_INFO(format, ...) fprintf(stdout, format, ##__VA_ARGS__);

__global__ void makeAllSMsActive(uint64_t N) {
  uint64_t tmp = 0;
  uint64_t prev = 0;
  volatile uint64_t next = 1;
  while (true) {
    for (uint64_t i = 1; i < N; i++) {
      tmp = prev + next;
      prev = next;
      next = tmp;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  int dev_count = 0;
  checkCudaErrors(cudaGetDeviceCount(&dev_count));

  std::vector<cudaEvent_t> cudaEvents;
  int totalSMs = 0;

  gpu::monitor::GpuMonitor gpuMonitor;

  for (int dev = 0; dev < dev_count; dev++) {
    checkCudaErrors(cudaSetDevice(dev));

    if (dev == 0) {
      cudaDeviceProp deviceProp;
      checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
      LOG_INFO("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

      int driverVersion = 0;
      checkCudaErrors(cudaDriverGetVersion(&driverVersion));

      int runtimeVersion = 0;
      checkCudaErrors(cudaRuntimeGetVersion(&runtimeVersion));

      LOG_INFO("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
              driverVersion / 1000, (driverVersion % 100) / 10,
              runtimeVersion / 1000, (runtimeVersion % 100) / 10);

      LOG_INFO("  CUDA Capability Major/Minor version number:    %d.%d\n",
              deviceProp.major, deviceProp.minor);

      LOG_INFO("  Number of SMs:                                 %d\n",
              deviceProp.multiProcessorCount);

      totalSMs = deviceProp.multiProcessorCount;
    }

    // Create a stream
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Kernel launch config
    cudaLaunchConfig_t config = {0};
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 1; // Cluster size in X-dimension
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    // The grid dimension is not affected by cluster launch, and is still
    // enumerated using number of blocks. The grid dimension should be a multiple
    // of cluster size.
    config.gridDim = dim3(totalSMs, 1, 1);

    // Threadblock: 128 threads (4 warps) for one warp-group
    constexpr int THREADS_PER_BLOCK = 32;
    config.blockDim = dim3(THREADS_PER_BLOCK);
    config.stream = stream;

    cudaEvent_t event;
    checkCudaErrors(cudaEventCreate(&event));

    uint64_t N = 1000000;
    checkCudaErrors(cudaLaunchKernelEx(&config, &makeAllSMsActive, N));

    LOG_INFO("\nLaunched makeAllSMsActive on device %d\n", dev);
    checkCudaErrors(cudaEventRecord(event, stream));

    cudaEvents.push_back(event);
  }

  gpuMonitor.start(0);

  for (auto& event : cudaEvents) {
    checkCudaErrors(cudaEventSynchronize(event));
  }

  LOG_INFO("DONE\n");

  // finish
  exit(EXIT_SUCCESS);
}
