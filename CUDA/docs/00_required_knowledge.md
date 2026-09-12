# CUDA kernels
Given CPU kernel -> write GPU kernel

## 1. Memory
Status | Topic                                               | Doc
-------|-----------------------------------------------------|--------------------------------------------------------------------------
   OK  | GPU architecture                                    | [./01_gpu_architecture.md](./01_gpu_architecture.md)
   OK  | Memory Hierarchy                                    | [./MEMORY.md](./MEMORY.md)
   OK  | SM Architecture                                     | [./02_gpu_sm.md](./02_gpu_sm.md)
   OK  | Register File                                       | [./03_gpu_sm_register_file.md](./03_gpu_sm_register_file.md)
   OK  | Restrictive Pointers                                | [../src/restrict_pointers/README.md](../src/restrict_pointers/README.md)
   OK  | Loop unrolling                                      | [../src/loop_unroll/README.md](../src/loop_unroll/README.md)
   OK  | Read global memory using vectorized instructions    | [../src/vectorized_load_store/README.md](../src/vectorized_load_store/README.md)
   OK  | How to bypass SM's private L1                       | [../src/bypass_L1/](../src/bypass_L1/)
   OK  | Atomicity                                           | [../src/atomic_read_modify_write/](../src/atomic_read_modify_write/)
   OK  | Default memory address alignment of cudaMalloc      | [./CUDA_cudaMalloc.md](./CUDA_cudaMalloc.md)
   \-  | Shared memory bank conflicts                        | \-
   \-  | Static and dynamic shared memory usages             | \-
   \-  | TMA                                                 | \-
   \-  | Pinned Memory                                       | \-


## 2. Control Flow
Status | Topic                                               | Doc
-------|-----------------------------------------------------|--------------------------------------------------------------------------
   \-  | Memory barriers                                     | \-
   \-  | Scoped synchronizations (warp, cluster, CTA, CTAs)  | \-


## 3. Tensor Core
Status | Topic                                               | Doc
-------|-----------------------------------------------------|--------------------------------------------------------------------------
   \-  | MMA                                                 | \-

## 4. GEMM
Status | Topic                                               | Doc
-------|-----------------------------------------------------|--------------------------------------------------------------------------
   \-  | Persistent GEMM                                     | \-


## 5. Inline PTX
- How to execute a PTX instruction?

## 6. Inspect a CUBIN file

## 7. Builtins
- Sort on GPU

## 8. CUDA APIs
- CUDA APIs
