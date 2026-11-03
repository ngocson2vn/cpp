#include <common/common.h>
#include <torch/torch.h>

template <typename T>
HVec<T> torchGemm(const HVec<T>& A,
                  const HVec<T>& B,
                  const uint32_t M,
                  const uint32_t N,
                  const uint32_t K) {
  printf("\nLaunching TORCH GEMM\n");
  torch::Device device(c10::kCUDA, 0);
  torch::TensorOptions options(device);
  auto tensorA = torch::from_blob(
    A.gpu().data(),
    {M, K},
    {K, 1},
    options
  );

  auto tensorB = torch::from_blob(
    B.gpu().data(),
    {K, N},
    {N, 1},
    options
  );

  auto tensorC = torch::matmul(tensorA, tensorB);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  auto ptrC = tensorC.data_ptr<T>();
  return DVec<T>(ptrC, M * N).cpu();
}

template HVec<float> torchGemm(const HVec<float>& A,
                               const HVec<float>& B,
                               const uint32_t M,
                               const uint32_t N,
                               const uint32_t K);


template <typename T>
HVec<T> torchSoftmax(const HVec<T>& hInp,
                     const uint32_t M,
                     const uint32_t N) {
  printf("\nLaunching TORCH softmax\n");
  torch::Device device(c10::kCUDA, 0);
  torch::TensorOptions options(device);
  auto inpTensor = torch::from_blob(
    hInp.gpu().data(),
    {M, N},
    {N, 1},
    options
  );

  auto outTensor = inpTensor.softmax(-1);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  auto devPtr = outTensor.data_ptr();
  return DVec<T>(reinterpret_cast<T*>(devPtr), M * N).cpu();
}

template HVec<float> torchSoftmax(const HVec<float>& hInp,
                                  const uint32_t M,
                                  const uint32_t N);
