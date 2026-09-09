# SRAM (Static RAM) vs DRAM (Dynamic RAM)
- SRAM uses 4 to 6 transistors per bit in a flip-flop circuit, while DRAM uses just one transistor and one capacitor per bit.
- Speed: SRAM has very low latency and is significantly faster than DRAM.
- Refreshing: SRAM retains data as long as power is on without needing updates; DRAM leaks charge and must be refreshed thousands of times per second.
- Density & Cost: DRAM is much denser (fits more memory in less space) and costs less per bit, making it ideal for large capacities.
- Common Use: SRAM is used for high-speed CPU cache memory (L1, L2, L3), while DRAM is used for main system memory (RAM).


# GPU Cache Hierarchy
<img src="./gpu_architecture_details.png" width="60%" />

* **L1 Cache (Private):** Each Streaming Multiprocessor (SM) has its own L1 cache. It is incredibly fast but only serves the threads running on that specific SM.
* **L2 Cache (Shared):** A larger, unified L2 cache sits between the SMs and the global memory (VRAM). All SMs share this cache.
* **Global Memory (VRAM):** The main pool of high-capacity memory.


# Cache Coherence
NVIDIA GPUs do not have full, automatic hardware cache coherence** for global memory across their L1 caches like modern CPUs do.

Instead of relying on hardware protocols (like MESI) that automatically keep all caches synchronized, NVIDIA GPUs use a **software-managed, relaxed memory consistency model** combined with a shared L2 cache.

### The Coherence Problem

Because the L2 cache is shared, it is inherently coherent across the entire GPU. If all memory requests bypassed L1 and went straight to L2, there would be no coherence issue.

However, because each SM has its own L1 cache, a data race can occur. If SM A reads a variable from global memory, it gets cached in SM A's L1. If SM B then overwrites that variable in global memory (updating the L2), SM A's L1 cache does not automatically get notified. SM A will continue to read the stale, outdated value from its own L1 cache.

### Why Not Use CPU-Style Coherence?

CPUs use complex hardware protocols (like bus snooping or directory-based MESI protocols) to ensure that when one core modifies a variable, all other cores immediately invalidate their cached copies of it.

GPUs cannot do this because of their scale. A modern CPU might have 16 or 32 cores, while a modern NVIDIA GPU has over a hundred SMs and tens of thousands of concurrent threads. If the GPU had to broadcast invalidation signals across the chip every time a thread updated memory, the interconnect would be completely paralyzed by coherence traffic.

### How NVIDIA Manages Memory Consistency

Because the hardware does not automatically synchronize the L1 caches, the burden is placed on the programmer and the compiler to manage it explicitly.

* **Bypassing L1 (Volatile/Atomics):** If multiple SMs need to communicate via global memory, the programmer can use the `volatile` keyword or atomic operations. This instructs the compiler to skip or invalidate the L1 cache for that specific variable, forcing the reads and writes to go directly to the shared, coherent L2 cache.
* **Memory Fences (Barriers):** CUDA provides memory fence instructions (like `__threadfence()`). A memory fence stalls the thread until all its previous global memory writes are guaranteed to be visible to all other SMs (usually by flushing the local cache to L2).
* **Scoped Memory Model:** In modern architectures (Volta and newer), NVIDIA uses a "scoped" memory consistency model. When you synchronize threads, you specify the "scope" of the synchronization (e.g., Thread Block, GPU Device, or System). The hardware only flushes and invalidates caches to the level necessary to guarantee coherence for that specific scope, avoiding full-chip performance penalties.


# Read-Only Cache
Read-only cache is a hardware component, which is separate from L1 cache.

The following PTX instruction fetches data from read-only cache:
```mlir
ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];
```
The `.nc` modifier is a powerful performance optimization. By routing the load through the read-only data cache, you gain several benefits:

1. **Higher Bandwidth:** The read-only cache has a separate memory pipeline. Using it relieves pressure on the standard L1 data cache, effectively increasing the total cache bandwidth available to your streaming multiprocessor (SM).
2. **Looser Alignment Rules:** The read-only cache is often better optimized for unaligned or scattered memory access patterns than standard global loads.
3. **No Coherency Overhead:** Because the GPU knows it doesn't have to monitor this cache for writes from other threads, it saves hardware tracking overhead.


# Cache Line
In an NVIDIA GPU, a **cache line** is the fundamental, atomic unit of data transfer between the slower global memory (VRAM) and the faster on-chip caches (L1 and L2).

Whenever a thread requests a single piece of data from memory, the GPU doesn't just fetch that one byte or float. Instead, it pulls in an entire cache line containing the requested data and the data immediately surrounding it.

Here is how cache lines dictate GPU memory architecture and performance.

### Size and Structure

On modern NVIDIA architectures (like Ampere, Hopper, and Ada Lovelace), a standard cache line is **128 bytes**.

To optimize bandwidth, this 128-byte line is typically subdivided into four **32-byte sectors**. If a memory request only touches data within one or two of those 32-byte sectors, the GPU is smart enough to only fetch the required sectors, saving memory bandwidth.

### Why Cache Lines Matter: The Warp

To understand why the 128-byte size is significant, you have to look at how GPUs process data. NVIDIA GPUs group execution threads into bundles of 32, known as a **warp**.

When a warp executes a memory read instruction, all 32 threads are requesting data at the same time. If each thread requests a standard 32-bit (4-byte) floating-point number, the total data requested by the warp is exactly:
**32 threads × 4 bytes = 128 bytes.**

This perfect alignment is not an accident—the cache line is sized precisely to serve an entire warp in a single, efficient memory transaction.

---

Here is a visual representation of how threads in a warp map to memory addresses:

---

### Memory Coalescing

The relationship between warp execution and cache lines leads to the most important performance optimization in CUDA programming: **memory coalescing**.

* **Coalesced Access (High Performance):** If all 32 threads in a warp request consecutive memory addresses, the hardware "coalesces" (combines) these 32 individual requests into a single 128-byte cache line fetch. The memory bus is used at 100% efficiency.
* **Uncoalesced/Scattered Access (Low Performance):** If the threads request scattered data (e.g., thread 0 wants address 10, thread 1 wants address 500, etc.), the requested data spans multiple different cache lines. The GPU is forced to fetch several 128-byte chunks (or 32-byte sectors) just to retrieve the scattered values, throwing away the unused data in those chunks. This can drop your memory bandwidth efficiency to 20% or lower, creating a massive bottleneck.


```text
If threads in a warp access memory addresses that are not in the same cache line, will the memory controller fetch all different cache lines?
```
**Yes.** If the threads in a warp request data scattered across multiple different cache lines, the memory controller has no choice but to fetch *every single one* of those required cache lines.

Here is exactly what happens in the hardware during this scenario (known as **uncoalesced access**):

* **Address Calculation:** When the memory instruction is executed, the Load/Store Unit (LSU) computes the memory addresses needed by all 32 threads in the warp.
* **Transaction Serialization:** The hardware determines how many 128-byte cache lines (and their internal 32-byte sectors) are needed to fulfill the 32 requests. If the threads request addresses spanning 15 different cache lines, the memory controller must issue 15 separate memory transactions. In the worst-case scenario, it will issue 32 separate transactions.
* **Massive Bandwidth Waste:** The memory controller can only fetch data in minimum chunk sizes (usually a 32-byte sector). If a single thread needs a 4-byte float from a sector, the GPU still fetches all 32 bytes, discards the other 28 bytes, and moves on. In this scenario, **87.5% of your memory bandwidth is wasted** transferring useless data.

Because the warp cannot continue executing until all of these separate, serialized memory fetches complete, this creates a massive stall and drops your memory throughput drastically.
