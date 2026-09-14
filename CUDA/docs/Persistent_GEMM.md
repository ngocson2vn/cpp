# Persistent GEMM
In a standard General Matrix-Matrix Multiplication (GEMM) implementation, you divide the output matrix into tiles and launch a CUDA grid containing one thread block for every tile. The GPU's hardware scheduler assigns these blocks to Streaming Multiprocessors (SMs) as resources become available.

A **persistent GEMM kernel** (often called "persistent threads" or "software scheduling") flips this model. Instead of relying on the hardware scheduler, you launch exactly enough thread blocks to fill the GPU's available SMs. Once running, these blocks do not exit after computing a single tile. Instead, they run a `while` loop, dynamically fetching and computing new tiles until the entire workload is finished.

Here is a breakdown of how it is implemented in CUDA C++ and why modern high-performance libraries like NVIDIA's CUTLASS use it.

---

## The Architecture of a Persistent GEMM

To implement a persistent kernel, you must move the scheduling logic from the host (via grid dimensions) to the device (via a work-fetching loop).

### 1. The Host-Side Launch

Instead of calculating the grid size based on the matrix dimensions, you query the GPU for its SM count and launch a grid that exactly matches the hardware's maximum concurrency.

```cpp
// 1. Query the device for its capacity
int num_SMs;
cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0);

// Assume we tuned our block size and shared memory so 
// exactly 2 blocks can run concurrently per SM.
int blocks_per_SM = 2; 
int grid_size = num_SMs * blocks_per_SM;
int block_size = 256;

// 2. Calculate total work (number of tiles to compute)
int total_tiles_m = (M + TILE_M - 1) / TILE_M;
int total_tiles_n = (N + TILE_N - 1) / TILE_N;
int total_tiles = total_tiles_m * total_tiles_n;

// 3. Launch the persistent grid
// (Optionally pass an allocated atomic counter initialized to 0)
persistent_gemm_kernel<<<grid_size, block_size>>>(A, B, C, M, N, K, total_tiles, d_atomic_counter);

```

### 2. The Device-Side Kernel

Inside the kernel, threads enter a loop. There are two primary ways to distribute the work among the persistent blocks: **Grid-Stride Loops**.

#### Deterministic Grid-Stride (Lock-Free)

The simplest way to schedule persistent threads is to use the `blockIdx.x` as the starting tile, and stride by `gridDim.x`. This requires no synchronization overhead.

```cpp
__global__ void persistent_gemm_grid_stride(
    const float* A, const float* B, float* C, 
    int M, int N, int K, 
    int total_tiles) 
{
    // Each block starts at its physical index and strides by the grid size
    int tile_idx = blockIdx.x;

    while (tile_idx < total_tiles) {
        // 1. Map 1D tile_idx to 2D matrix coordinates
        int tiles_per_row = (N + TILE_N - 1) / TILE_N;
        int tile_row = (tile_idx / tiles_per_row);
        int tile_col = (tile_idx % tiles_per_row);

        // 2. Perform standard tile computation
        // (Load to shared memory, compute dot products, write to C)
        compute_tile(A, B, C, tile_row, tile_col, M, N, K);

        // 3. Move to the next tile in the schedule
        tile_idx += gridDim.x;
    }
}
```

## Why go to this effort?

Implementing a persistent kernel adds complexity, but it solves several critical bottlenecks in high-performance computing:

* **Eliminates Wave Quantization:** In standard scheduling, if you have 100 tiles of work and your GPU can compute 80 at a time, the GPU processes one "wave" of 80 tiles. Then, a second wave of 20 tiles launches. During that second wave, 60% of your GPU's SMs sit completely idle waiting for the kernel to finish. Persistent kernels with work-stealing keep all SMs fed until the very last tile is claimed.
* **Warp Specialization (Hopper/CUDA 12):** Modern architectures like NVIDIA Hopper use asynchronous DMA (Tensor Memory Accelerator or TMA). Persistent threads allow you to split roles within a single block: some warps *only* issue memory fetches, while other warps *only* execute math (WGMMA). This producer-consumer model requires threads to stay alive continuously to manage synchronization barriers.
* **Enables Stream-K:** A technique popularized by CUTLASS where work is partitioned not just by output tiles ($M \times N$), but along the reduction dimension ($K$). Persistent threads allow partial results from different blocks to be synchronized and accumulated in software, which is impossible with the standard CUDA hardware scheduler.