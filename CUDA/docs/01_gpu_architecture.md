# GPU Architecture
<img src="./gpu_architecture_details.png" width="100%" />
<br/><br/>

## Programming Model
A CUDA kernel is launched with multiple thread blocks.

Questions:
1. Thread blocks will be distributed over the same Streaming Multiprocessor or different Streaming Multiprocessors, right?

2. Must a thread block be executed by the same Streaming Multiprocessor?

<br/>

Here are the direct answers to how CUDA distributes and executes thread blocks across a GPU:

### 1. Are thread blocks distributed over the same SM or different SMs?

**Both.** Thread blocks are distributed across **different** SMs, but multiple thread blocks will also be assigned to the **same** SM.

When you launch a kernel, the GPU's hardware scheduler (often called the GigaThread Engine) distributes the blocks to available SMs.

* **Across different SMs:** To maximize parallel performance, the scheduler spreads the blocks across all available SMs on the GPU.
* **On the same SM:** A single SM is designed to execute multiple thread blocks concurrently. As long as the SM has enough hardware resources (registers, shared memory, and maximum thread capacity) available, it will accept multiple blocks at the same time to keep its processing cores busy while other blocks are waiting on memory fetches.

### 2. Must a thread block be executed by the same SM?

**Yes, absolutely.**

Once a thread block is assigned to an specific SM, its entire lifecycle happens on that SM. It **never migrates** to a different SM, and its threads are never split across multiple SMs.

This strict hardware rule exists for two critical reasons:

1. **Shared Memory:** Threads within a block often need to share data. They do this using "Shared Memory," which is ultra-fast physical memory located directly on the SM itself. If threads were on different SMs, they couldn't access the same physical cache.
2. **Synchronization:** Threads within the same block use `__syncthreads()` to wait for each other. This barrier synchronization relies on the hardware of a single SM. If a block were split across SMs, the hardware could not guarantee this fast, unified synchronization.

<br/>

**For a thread block, does the hardware scheduler schedule warp by warp?**

**Yes, absolutely.**

While a thread block is assigned to a Streaming Multiprocessor (SM) as a single logical group, the SM's hardware does not execute the block as one giant monolithic unit. Instead, it breaks the block down and schedules it **warp by warp**.

Here is how that process works under the hood:

#### 1. Division into Warps

As soon as a thread block is loaded onto an SM, the hardware immediately divides its threads into groups of 32, called **warps**. (For example, a block of 256 threads is split into 8 warps). From this point on, the warp is the fundamental unit of scheduling and execution.

#### 2. The Warp Scheduler

Each SM contains specialized hardware called **Warp Schedulers**. Every clock cycle, the warp scheduler looks at all the active warps currently residing on the SM and selects one (or more, depending on the architecture) that is ready to execute its next instruction.

#### 3. SIMT Execution (Lockstep)

When a warp is scheduled for execution, all 32 threads within that warp execute the exact same instruction at the exact same time, but on their own individual data. This is known as **SIMT** (Single Instruction, Multiple Threads) architecture.

#### 4. Zero-Overhead Context Switching (Latency Hiding)

Scheduling by warp is the secret to a GPU's massive performance. If Warp A needs to read data from global memory (which takes hundreds of clock cycles), it stalls. The warp scheduler doesn't wait; it instantly swaps Warp A out and schedules Warp B, which has its data ready and is waiting to do math.

Because the hardware keeps the exact state (registers, program counters) of all active warps saved on the SM, swapping between warps takes **zero clock cycles**. The GPU hides the latency of slow memory operations by simply keeping the processing cores busy with other warps.

<br/>

**Does the Warp Scheduler schedule warps concurrently?**
**Yes, they do.**

In modern CUDA architectures, warps are scheduled and executed concurrently, primarily because a single Streaming Multiprocessor (SM) actually contains **multiple warp schedulers**.

Here is how the concurrency breaks down at the hardware level:

### 1. Multiple Schedulers per SM

A modern SM is not a single monolithic processing unit. It is divided into sub-partitions (usually 4 sub-blocks in architectures like Ampere, Hopper, or Ada Lovelace).

Each of these 4 sub-partitions has its own dedicated **Warp Scheduler** and its own set of execution cores (FP32, INT32, Tensor Cores, etc.). Because they operate independently, an SM can schedule and issue instructions for **4 different warps simultaneously** in a single clock cycle.

### 2. Dual-Issue Dispatch (Instruction-Level Concurrency)

Depending on the specific GPU architecture, a single warp scheduler might also have multiple dispatch units.

If a scheduler looks at a warp and sees that its next two instructions are independent and use different hardware units (for example, one instruction is a math calculation and the other is a memory load), the scheduler can use **dual-issue** to dispatch both instructions concurrently from the *same* warp in a single clock cycle.

### 3. Concurrent Execution via Pipelining

Even when a scheduler only issues one instruction per clock cycle, the execution is still highly concurrent.

* Mathematical operations take several clock cycles to complete.
* Because the execution units are pipelined, the scheduler issues an instruction for Warp A on cycle 1, Warp B on cycle 2, and Warp C on cycle 3.
* By cycle 3, Warp A, B, and C are all concurrently flowing through different stages of the execution pipelines.

### Summary

While a *single* warp scheduler typically selects one ready warp to issue an instruction for in a given clock cycle, the GPU achieves massive concurrency because **multiple schedulers are acting in parallel**, and they are feeding pipelines that keep dozens of warps in flight simultaneously across the SM.
