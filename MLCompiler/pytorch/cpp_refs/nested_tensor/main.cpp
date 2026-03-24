#include <vector>
#include <cuda_runtime.h>
#include <torch/torch.h>

int main(int argc, char** argv) {
  int device_id = 0;
  void* data_;
  int size = 256 * 1024*1024;
  TORCH_CHECK(cudaSetDevice(device_id) == cudaSuccess);
  TORCH_CHECK(cudaMalloc(&data_, size) == cudaSuccess);

  // 
  // Create nested tensors
  // 

  // Create a CPU nested tensor
  torch::Tensor cpu_tensor;
  torch::Tensor t1 = torch::randn({3, 512}); 
  torch::Tensor t2 = torch::randn({5, 512}); 
  torch::Tensor t3 = torch::randn({2, 512});
  std::vector<torch::Tensor> tensor_list = {t1, t2, t3};
  cpu_tensor = torch::nested::nested_tensor(tensor_list);
  auto dtype = cpu_tensor.dtype();

  std::cout << "cpu_tensor is nested? " << (cpu_tensor.is_nested() ? "Yes" : "No") << std::endl;
  std::cout << "cpu_tensor.numel()? " << cpu_tensor.numel() << std::endl;
  std::cout << "cpu_tensor._nested_tensor_size()? " << std::endl;
  std::cout << cpu_tensor._nested_tensor_size() << std::endl;

  // Create a GPU nested tensor
  // Allocate an empty, uninitialized 1D buffer directly on the GPU
  auto device = torch::Device(at::kCUDA, static_cast<c10::DeviceIndex>(device_id));
  auto gpu_buffer = torch::empty(
      {cpu_tensor.numel()}, 
      torch::TensorOptions().dtype(dtype).device(device)
  );

  // Manually wrap the GPU buffer and the CPU metadata into a Nested Tensor
  auto gpu_tensor = at::detail::make_tensor<at::native::NestedTensorImpl>(
      gpu_buffer,
      cpu_tensor._nested_tensor_size()
  );
  std::cout << "gpu_tensor is nested? " << (gpu_tensor.is_nested() ? "Yes" : "No") << std::endl;

  // Copy CPU -> GPU
  gpu_tensor.copy_(cpu_tensor, true);
  std::cout << "gpu_tensor is nested? " << (gpu_tensor.is_nested() ? "Yes" : "No") << std::endl;
  std::cout << "gpu_tensor.numel()? " << gpu_tensor.numel() << std::endl;
  std::cout << "gpu_tensor._nested_tensor_size()? " << std::endl;
  std::cout << gpu_tensor._nested_tensor_size() << std::endl;

  std::cout << "DONE" << std::endl;
}
