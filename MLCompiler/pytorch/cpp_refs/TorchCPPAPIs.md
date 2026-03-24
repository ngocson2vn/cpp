# Tensor
## Copy
```C++
Tensor::copy_(const at::Tensor & src, bool non_blocking)
```

Call Stack for nested tensors:
```C++
pytorch/aten/src/ATen/native/Copy.cpp
  -> Tensor& copy_(Tensor& self, const Tensor& src, bool non_blocking)
    -> static Tensor & copy_impl(Tensor & self, const Tensor & src, bool non_blocking)
      -> ...
        -> Tensor& copy_nested_(Tensor& self, const Tensor& src, bool non_blocking)
          -> inline NestedTensorImpl* get_nested_tensor_impl(const at::Tensor& tensor)
              TORCH_CHECK(
                  tensor.is_nested(), "get_nested_tensor_impl requires a NestedTensor.");
```
