# CUDA kernels
Given CPU kernel -> write GPU kernel

## 1. Memory
Status | Topic                                                          | Doc
-------|----------------------------------------------------------------|--------------------------------------------------------------------------
   OK  | GPU architecture                                               | [./01_gpu_architecture.md](./01_gpu_architecture.md)
   OK  | Memory Hierarchy                                               | [./MEMORY.md](./MEMORY.md)
   OK  | SM Architecture                                                | [./02_gpu_sm.md](./02_gpu_sm.md)
   OK  | Register File                                                  | [./03_gpu_sm_register_file.md](./03_gpu_sm_register_file.md)
   OK  | Restrictive Pointers                                           | [../src/restrict_pointers/README.md](../src/restrict_pointers/README.md)
   OK  | Loop unrolling                                                 | [../src/loop_unroll/README.md](../src/loop_unroll/README.md)
   OK  | Read global memory using vectorized instructions               | [../src/vectorized_load_store/README.md](../src/vectorized_load_store/README.md)
   OK  | How to bypass SM's private L1                                  | [../src/bypass_L1/](../src/bypass_L1/)
   OK  | Atomicity                                                      | [../src/atomic_read_modify_write/](../src/atomic_read_modify_write/)
   OK  | Default memory address alignment of cudaMalloc                 | [./CUDA_cudaMalloc.md](./CUDA_cudaMalloc.md)
   OK  | Pinned Memory                                                  | [./CUDA_Pinned_Memory.md](./CUDA_Pinned_Memory.md)
   OK  | Static shared memory usage                                     | [../src/persistent_gemm/main2.cu](../src/persistent_gemm/main2.cu#L83)
   OK  | Thread coarsening                                              | [../src/thread_coarsening/main.cu](../src/thread_coarsening/main.cu#L27)
   OK  | Dynamic shared memory usage (use `extern` keyword)             | [../src/persistent_gemm/main3.cu](../src/persistent_gemm/main3.cu#L72)
   OK  | Set desired dynamic shared memory using `cudaFuncSetAttribute` | [../src/persistent_gemm/main3.cu](../src/persistent_gemm/main3.cu#L275)
   \-  | Shared memory bank conflicts                                   | \-
   \-  | TMA                                                            | \-
   \-  | The usage of `__grid_constant__`                               | \-


## 2. Control Flow
Status | Topic                                                          | Doc
-------|----------------------------------------------------------------|--------------------------------------------------------------------------
   OK  | CUDA Event                                                     | [./CUDA_Event.md](./CUDA_Event.md)
   \-  | Memory barriers                                                | \-
   \-  | Scoped synchronizations (warp, cluster, CTA, CTAs)             | \-


## 3. Tensor Core
Status | Topic                                                          | Doc
-------|----------------------------------------------------------------|--------------------------------------------------------------------------
   \-  | MMA                                                            | \-


## 4. GEMM
### Persistent GEMM
Doc: [./Persistent_GEMM.md](./Persistent_GEMM.md) <br/>
Code: [../src/persistent_gemm/](../src/persistent_gemm/)

**NOTE:**
- Syncing CTAs before dot product to ensure that all threads see the same sA and sB tiles.
- Syncing CTAs after dot product to ensure that all threads have finished reading sA and sB tiles.

<br/>

## 5. Inline PTX
How to use inline PTX: [../src/vectorized_load_store/main3.cu](../src/vectorized_load_store/main3.cu#L28)


## 6. Sorting
Status | Topic                                                          | Doc
-------|----------------------------------------------------------------|--------------------------------------------------------------------------
   OK  | Sort on GPU using `thrust::sort()`                             | [../src/thrust_sort/](../src/thrust_sort/)
   OK  | Manually implement rank sort                                   | [../src/rank_sort/](../src/rank_sort/)
   OK  | Implement bitonic sort                                         | [../src/bitonic_sort/](../src/bitonic_sort/)


## 7. Intrinsics and Builtins
### Integer Mathematical Functions
https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INT.html



## 8. Inspect a CUBIN file
Using `cuobjdump`:
```bash
# List ELF files
cuobjdump -lelf main

# Dump SASS
cuobjdump -sass main
```


## 9. CUDA APIs
### How to call `cudaLaunchKernel`
Code: [../src/persistent_gemm/main2.cu](../src/persistent_gemm/main2.cu#277)

### How to call `cudaLaunchKernelEx`
Code: [../src/persistent_gemm/main3.cu](../src/persistent_gemm/main3.cu#287)
