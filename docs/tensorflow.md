# dtf
https://github.com/dissecting-tensorflow/dtf/tree/master

# CUDA_ERROR_ILLEGAL_ADDRESS and BFC
Sometimes, CUDA_ERROR_ILLEGAL_ADDRESS occurs with a specific set of input tensors. However, when running the model with the same set of input tensors, CUDA_ERROR_ILLEGAL_ADDRESS may not be reproduced. The reason is that even though the culprit CUDA kernel accesses a wrong tensor linear index, the corresponding GPU memory address is still in BFC pool. 

The CUDA_ERROR_ILLEGAL_ADDRESS only occurs when the GPU memory address is not in BFC pool. To quickly reproduce the CUDA_ERROR_ILLEGAL_ADDRESS, we should reduce GPU memory fraction:
```C++
gpu_options->set_per_process_gpu_memory_fraction(static_cast<float>(options.gpu_memory_percent) / 100.0);
```

# Search for an op definition:
Op definition:
```
REGISTER_OP("TensorScatterUpdate")
```

Kernel definition:
```
Name("TensorScatterUpdate")
```