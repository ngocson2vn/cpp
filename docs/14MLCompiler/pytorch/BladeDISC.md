
# RAL
## ral_kernel_launch___gpu___pvoid_ppvoid_i64_pvoid_i64_i64_i64_i64_i64_i64_i32_pvoid_i32_ppvoid___void
mlir/ral/context/base/cuda/cuda_context_impl.cc
```C++
TAO_RAL_API(tao::ral::gpu::kRalGpuLaunch, "gpu", ral_base_cuda_launch);
```
<br/>

pvoid_ppvoid_i64_pvoid_i64_i64_i64_i64_i64_i64_i32_pvoid_i32_ppvoid___void is the signature of function `ral_base_cuda_launch`:
```C++
void ral_base_cuda_launch(ExecutionContext* ctx, void** blobs, size_t num_blobs,
                          const char* kernel_name, intptr_t gridX,
                          intptr_t gridY, intptr_t gridZ, intptr_t blockX,
                          intptr_t blockY, intptr_t blockZ,
                          int32_t smem,        /* sharedMemBytes */
                          void* stream_handle, /* stream */
                          int32_t num_args, void** params /* kernel params */);
```
<br/>

mlir/ral/ral_helper.h
```C++
// Macros used to define TAO_RAL apis.
#define TAO_RAL_API(name, device, ...) \
  TAO_RAL_API_UNIQ_HELPER(name, device, __COUNTER__, __VA_ARGS__)

#define TAO_RAL_API_UNIQ_HELPER(name, device, ctr, ...) \
  TAO_RAL_API_UNIQ(name, device, ctr, __VA_ARGS__)

#define TAO_RAL_API_UNIQ(name, device, ctr, ...)                               \
  static bool unused_ret_val_##ctr =                                           \
      ::tao::ral::TaoRalApiRegistry::Global().Register(                        \
          ::tao::ral::TaoRalApiFuncNameHelper<decltype(&__VA_ARGS__)>::Invoke( \
              std::string(name) + "___" + std::string(device)),                \
          std::string(name),                                                   \
          ::tao::ral::TaoRalApiFuncInvoker<decltype(&__VA_ARGS__),             \
                                           &__VA_ARGS__>::Invoke);
```
<br/>

# disc_shape.SymbolicDim
<br/>

# AStich
## DiscFusionPass
mlir/disc/transforms/lhlo_fusion.cc
tao_compiler/bazel-out/k8-opt/bin/external/mlir-hlo/lhlo/IR/lhlo_ops.h.inc
