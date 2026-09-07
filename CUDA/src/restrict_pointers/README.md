# Restrict Pointers
## ld.global.b32
Given the following kernel:
```cpp
extern "C" {

// c = (a + b) * a
__global__ void add_mul_vectors(const float *a, const float *b, float *c,
                                int totalElems) {
  auto numThreads = blockDim.x;
  auto numElems = (totalElems + numThreads - 1) / numThreads;
  auto tid = threadIdx.x;

  // numElems = 4
  //  tid=0  |  tid=1  |  tid=2
  // 0 1 2 3 | 4 5 6 7 | 8 9 10 11

  // Ensure that `maxIdx` is not greater than `totalElems`
  auto maxIdx = min((tid + 1) * numElems, totalElems);
  for (int idx = tid * numElems; idx < maxIdx; idx++) {
    c[idx] = a[idx] + b[idx];
  }

  for (int idx = tid * numElems; idx < maxIdx; idx++) {
    c[idx] = c[idx] * a[idx];
  }
}

}
```

`nvcc` will generate PTX instructions for reading pointers `a` and `b` as follows:
```mlir
  // Define 18 64-bit registers %rd0, %rd1, ..., %rd17
  .reg .b64 	%rd<18>;

  // Load the value at memory location `[%rd1]` into register `%r26`
  ld.global.b32 	%r26, [%rd1];
```

`ld.global.b32` --> L1 --> L2 --> Global
- First, try L1
- If L1 cache hit, read L1
- If L1 cache miss, try L2
- If L2 cache hit, read L2
- If L2 cache miss, try Global

## ld.global.nc.b32
If we mark pointers with `__restrict__` attributes as follows:
```cpp
__global__ void add_mul_vectors(const float *__restrict__ a,
                                const float *__restrict__ b,
                                float *__restrict__ c, int totalElems)
```
By using the `__restrict__` attribute with a pointer, we are making a strict promise with the compiler: <br/>
**"For the lifetime of this pointer, the memory it points to will only be accessed through this pointer."**


`nvcc` will generates `ld.global.nc.b32` PTX instructions for reading `a` and `b` as follows:
```mlir
	ld.global.nc.b32 	%r26, [%rd1];
	ld.global.nc.b32 	%r27, [%rd2];
```

The modifier `.nc` means **non-coherence**. It is a powerful performance optimization that routes the load through the **read-only cache**.

Here is exactly why routing global memory reads through the read-only cache boosts performance.

### 1. Global Memory is Far Away (DRAM vs. SRAM)

Global memory (the VRAM on the graphics card) is physically located off the processor chip. Fetching data from there is incredibly slow—it can take **400 to 800 clock cycles** for a single read.

Caches (like the L1 cache and the Read-Only cache) are made of ultra-fast SRAM physically located *inside* the Streaming Multiprocessor (SM) right next to the cores doing the math. Fetching data from a cache takes roughly **20 to 30 clock cycles**.

When you route a read through the read-only cache:

1. The GPU fetches the data from slow global memory **exactly once**.
2. It stores a temporary copy of that data in the on-chip Read-Only cache.
3. If that thread (or any nearby thread in the same block) needs that data again, it reads it instantly from the fast cache, bypassing global memory entirely.

### 2. Unlocking "Parallel" Memory Pipes

This is the real secret behind the bandwidth boost.

In many GPU architectures, the L1 data cache and the Read-Only (Texture) cache are separate hardware pathways.

* If you do not use `__restrict__`, the compiler puts *everything*—your reads, your writes, and your local thread variables—into the standard L1 cache. The L1 cache gets congested.
* By using `__restrict__`, you split the traffic. Your reads go down the Read-Only hardware pipe, while your writes and local variables use the L1 hardware pipe. You are literally utilizing more physical silicon at the same time, increasing your overall on-chip bandwidth.

### 3. Preventing "Cache Thrashing"

Caches are very small (often just 64KB to 128KB per SM).

If your reads and writes share the same L1 cache, they fight for space. A large burst of writes to your `result` array might force the GPU to kick out (evict) the `a` and `b` array data that it just loaded. When the GPU needs `a` and `b` for the next loop iteration, it has to suffer the 800-cycle penalty to fetch them from global memory all over again.

By sending read-only data to its own dedicated cache space (or treating it differently in a unified cache), you protect your incoming read data from being overwritten by your outgoing write data.
