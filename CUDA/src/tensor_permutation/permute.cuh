#include <cstdint>

#include <cuda_runtime.h>

#include <common/common.h>

// Using thread-coarsening
__global__ void permuteKernel(const float* inData,
                              const int64_t* inStride,
                              const int64_t* perm,
                              float* outData,
                              const int64_t* outStride,
                              const int64_t NDIMS,
                              const int64_t TOTAL_ELEMS) {
  // Old linear index
  int64_t inIdx = blockIdx.x * blockDim.x + threadIdx.x;
  // inIdx = d0 * stride[0] + d1 * stride[1] + d2 * stride[2] + ... + d{NDIMS - 1} * stride[NDIMS - 1]
  // How to recover [d0, d1, d2, ..., d{NDIMS - 1}]?
  // 
  // For 3D case,
  // inIdx = d0 * stride[0] + d1 * stride[1] + d2 * stride[2]
  // subIdx = inIdx
  // d0 = subIdx // stride[0]
  // 
  // subIdx = (inIdx - d0 * stride[0])
  // d1 = subIdx // stride[1]
  // 
  // subIdx = subIdx - d1 * stride[1]
  // d2 = subIdx // stride[2]

  const int64_t totalThreads = gridDim.x * blockDim.x;

  while (inIdx < TOTAL_ELEMS) {
    // Compute output linear index
    int64_t subIdx = inIdx;
    int64_t d = 0;
    int64_t outIdx = 0;
    for (int i = 0; i < NDIMS; i++) {
      // Derive logical index for i-th dimension
      d = subIdx / inStride[i];

      // Since the permutation maps i to perm[i], 
      // d becomes logical index for perm[i]-th dimension.
      // Therefore, we need to derive the corresponding outStride 
      // for perm[i]-th dimension
      outIdx += d * outStride[perm[i]];

      // Update subIdx
      subIdx -= d * inStride[i];
    }

    outData[outIdx] = inData[inIdx];

    inIdx += totalThreads;
  }
}

HVec<float> gpuPermute(HVec<float> dataVec, const std::vector<int64_t>& dimVec, const std::vector<int64_t>& permVec) {
  printf("\nLaunching GPU permute\n");
  auto inStride = HVec<int64_t>(dimVec.size(), 0);
  int64_t s = 1;
  for (int64_t i = 0; i < dimVec.size(); i++) {
    inStride[dimVec.size() - 1 - i] = s;
    s = s * dimVec[dimVec.size() - 1 - i];
  }

  auto perm = HVec<int64_t>(permVec);

  auto outDimVec = HVec<int64_t>(dimVec.size(), 0);
  for (uint32_t i = 0; i < dimVec.size(); i++) {
    // i -> perm[i]
    outDimVec[permVec[i]] = dimVec[i];
  }

  auto outStride = HVec<int64_t>(outDimVec.size(), 0);
  s = 1;
  for (uint32_t i = 0; i < outDimVec.size(); i++) {
    outStride[outDimVec.size() - 1 - i] = s;
    s = s * outDimVec[outDimVec.size() - 1 - i];
  }

  // Confirm dimVec, inStride, perm, outDimVec, and outStride
  printf("dimVec: ");
  HVec<int64_t>(dimVec).print();

  printf("inStride: ");
  inStride.print();

  printf("perm: ");
  perm.print();

  printf("outDimVec: ");
  outDimVec.print();

  printf("outStride: ");
  outStride.print();

  // Get SM count
  uint32_t smCount = 64;
  cudaDeviceProp devProp;
  auto status = cudaGetDeviceProperties(&devProp, 0);
  if (status == cudaSuccess) {
    smCount = devProp.multiProcessorCount;
  }

  auto kernelFn = reinterpret_cast<const void*>(permuteKernel);

  // Get max blocks per SM
  int threads = 1024;
  int blocksPerSM = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocksPerSM,
    kernelFn,
    threads,
    0
  );
  int heuBlocks = blocksPerSM * smCount;

  // Find optimal threads and blocks
  threads = (dataVec.size() > 1024) ? 1024 : dataVec.size();
  int blocks = dataVec.size() + (threads - 1) / threads;
  blocks = (heuBlocks > blocks) ? blocks : heuBlocks;
  printf("threads = %d blocks = %d\n", threads, blocks);

  auto outData = DVec<float>(dataVec.size(), 0);

  // Launch kernel
  permuteKernel<<<blocks, threads>>>(
    dataVec.gpu().data(),
    inStride.gpu().data(),
    perm.gpu().data(),
    outData.data(),
    outStride.gpu().data(),
    dimVec.size(),
    dataVec.size()
  );

  CHECK_CUDA_ERROR(cudaGetLastError());
  cudaDeviceSynchronize();

  return outData.cpu();
}