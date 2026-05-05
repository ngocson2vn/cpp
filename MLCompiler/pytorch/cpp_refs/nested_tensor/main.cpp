#include <vector>
#include <cuda_runtime.h>
#include <torch/torch.h>

namespace sony {

std::string timestamp() {
  // 1. Get current time as a time_point
  auto now = std::chrono::system_clock::now();

  // 2. Convert to time_t (legacy timestamp)
  std::time_t t_c = std::chrono::system_clock::to_time_t(now);

  // 3. Convert to broken-down time (local)
  std::tm* now_tm = std::localtime(&t_c);

  // 4. Format using strftime
  char buffer[80];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", now_tm);

  return std::string(buffer);
}

void print_key_tensor(const std::string& name, const torch::Tensor& tensor) {
  auto cpu_tensor = tensor.cpu().contiguous();

  if (cpu_tensor.is_nested()) {
    std::cout << sony::timestamp() << " " << name << ":\n";
    std::vector<torch::Tensor> rows = cpu_tensor.unbind(0);
    for (size_t i = 0; i < rows.size(); ++i) {
      torch::Tensor char_tensor = rows[i].contiguous();
      auto seq_len = char_tensor.size(0);
      auto raw_data = reinterpret_cast<const char*>(char_tensor.data_ptr<uint8_t>());
      std::string row_str(raw_data, seq_len);
      std::cout << "  Row " << i << ": " << row_str << std::endl;
    }
  } else {
    auto seq_len = cpu_tensor.size(0);
    auto raw_data = reinterpret_cast<const char*>(cpu_tensor.data_ptr<uint8_t>());
    std::string row_str(raw_data, seq_len);
    std::cout << sony::timestamp() << " " << name << ": " << row_str << std::endl;
  }
}

}

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
