# L1 and L2 and HBM
Architecture:
<img src="./architecture.png">
Ref: https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#gpu-arch

At a high level, NVIDIA® GPUs consist of a number of Streaming Multiprocessors (SMs), on-chip L2 cache, and high-bandwidth DRAM. Arithmetic and other instructions are executed by the SMs; data and code are accessed from DRAM via the L2 cache. As an example, an NVIDIA A100 GPU contains 108 SMs, a 40 MB L2 cache, and up to 2039 GB/s bandwidth from 80 GB of HBM2 memory.

**L1 note:**<br/>
On some GPU devices: the L1 cache and shared memory may use the same hardware resources.<br/>
Ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g6c9cc78ca80490386cf593b4baa35a15

# Memory Address Alignment
Ref: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
>Memory allocated through the CUDA Runtime API, such as via cudaMalloc(), is guaranteed to be aligned to at least 256 bytes. 
