# NVML Metrics
[/usr/local/cuda-13.1/targets/x86_64-linux/include/nvml.h](/usr/local/cuda-13.1/targets/x86_64-linux/include/nvml.h)

## SM Metrics
```c
/**
 * GPM Metric Identifiers
 */
typedef enum
{
    // ...
    NVML_GPM_METRIC_SM_UTIL                     = 2,    //!< Percentage of SMs that were busy. 0.0 - 100.0
    NVML_GPM_METRIC_SM_OCCUPANCY                = 3,    //!< Percentage of warps that were active vs theoretical maximum. 0.0 - 100.0
    // ...
} nvmlGpmMetricId_t;
```

## Theoretical Maximum Warps
In the context of NVIDIA GPUs and the `NVML_GPM_METRIC_SM_OCCUPANCY` metric, the **theoretical maximum** refers to the **absolute hardware limit of concurrent warps that a single Streaming Multiprocessor (SM) can physically support at one time**, assuming no other resource bottlenecks.

Because a warp always consists of 32 threads, this limit is directly tied to the maximum number of resident threads an SM can handle.

### How the Theoretical Maximum is Determined

The theoretical maximum is completely fixed by the GPU's microarchitecture (its "Compute Capability"). It does not depend on your code.

The theoretical maximums for recent NVIDIA GPU architectures is described in: [../../docs/compute_capabilities.md](../../docs/compute_capabilities.md)


### What the Metric is Actually Telling You

When NVML reports `NVML_GPM_METRIC_SM_OCCUPANCY = 50.0` on a Hopper H100 GPU, it means the SM had an average of **32 active warps** running on it (out of its hardware ceiling of 64).

Even heavily optimized kernels often fail to reach 100% occupancy (the theoretical maximum) due to physical resource limits on the SM:

1. **Register Usage:** If each thread in your kernel requires a large number of registers, the SM will run out of register file space before it hits the maximum warp count.
2. **Shared Memory Allocation:** If your thread blocks allocate a lot of shared memory, the SM won't have enough shared memory capacity to host the maximum number of warps.
3. **Block Size Constraints:** If your thread block sizes don't divide cleanly into the SM's maximum capacities, "wasted" space can leave warps unallocated.

In short, this metric tells you how efficiently you are packing work into the SM relative to the ultimate physical ceiling of the silicon you are running on.
