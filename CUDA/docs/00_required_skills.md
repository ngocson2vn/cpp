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
   \-  | How to bypass SM's private L1                       | \-
   \-  | Memory barrier                                      | \-
   \-  | Scoped synchronizations                             | \-
   \-  | Default memory address alignment of cudaMalloc      | \-


## 2. Inline PTX
- How to execute a PTX instruction?

## 3. Inspect a CUBIN file

## 4. Builtins
- Sort on GPU

## 5. CUDA APIs
- CUDA APIs

