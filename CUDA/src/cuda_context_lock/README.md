# CUDA Context Lock
## Case 1: Launch gelu kernels serially
Use 1 thread to launch gelu kernels serially. There is no lock:

<img src="./lock_no.png" />

## Case 2: Launch gelu kernels concurrently
Use 4 threads to launch gelu kernels concurrently. There are many write locks:

<img src="./lock_yes.png" />

OS Runtime Libraries Track: Above CUDA API calls, there are continuous, elongated blocks labeled `pthread_rwlock_wrlock`. This is the POSIX standard for acquiring a write lock on a read-write lock.

Because 4 C++ threads are simultaneously bombarding the same context with gelu launches, they are physically unable to execute concurrently on the CPU. The operating system grants the lock to one thread, while the other three are trapped inside `pthread_rwlock_wrlock` waiting for their turn. This inflates the API call from a few microseconds to a massive stall, leaving the GPU waiting for work.

## Case 3: Multi-CUDA Context
Each thread uses a separate CUDA context => No lock contention

Use MPS to funnel multiple MPS client contexts into a single MPS server CUDA context => bypass time-sliced scheduling bottleneck.

To fully understand how your multi-context architecture achieves lock-free concurrency, we need to look under the hood of how NVIDIA implemented MPS.

The terminology can be slightly confusing because NVIDIA separates the system into three distinct entities: **The Client(s)** (your custom contexts), **The Control Daemon**, and **The Server Process**.

Here is the exact step-by-step pipeline of how your multiple CUDA contexts communicate with the hardware through the MPS ecosystem, specifically focusing on modern Volta, Ampere, Ada, and Hopper architectures.

---

### Phase 1: The MPS Control Daemon (The Gatekeeper)

Before your application even runs, the system administrator starts the `nvidia-cuda-mps-control` daemon.

* **Role:** The daemon acts as a lightweight background listener and gatekeeper. It listens on a designated Unix domain socket (defined by the `CUDA_MPS_PIPE_DIRECTORY` environment variable).
* **Action:** When your application starts, it does not immediately talk to the GPU. Instead, the CUDA driver inside your process detects the MPS pipe and pings the daemon.

### Phase 2: Client Registration & The MPS Server

When your main thread calls `cuInit` and subsequently `cuCtxCreate` (or `cuDevicePrimaryCtxRetain`), the pipeline springs into action.

1. **Context Creation = Client Request:** Every time you call `cuCtxCreate`, the CUDA driver sends a message to the Control Daemon saying, *"I need a new isolated environment."*
2. **Server Spawning:** If this is the first context requesting access to the GPU, the Control Daemon spawns a new, separate OS process called `nvidia-cuda-mps-server`. There is exactly one Server process per GPU.
3. **The Shared Hardware Context:** The MPS Server process talks to the GPU and creates a single, massive **Hardware Context**. This is the actual state on the silicon.
4. **Client Slot Allocation:** The Control Daemon hands your `cuCtxCreate` request to the Server. The Server allocates one of its hard-coded client slots (e.g., 1 out of 48) to this new software context.

Because your thread pool creates 4 custom contexts (plus 1 primary context), your single application registers as **5 distinct clients** to the MPS Server.

---

### Phase 3: Memory and Resource Provisioning

This is where the magic of modern MPS (Volta architecture and newer) happens.

In older architectures (Kepler/Pascal), the MPS Server had to act as a middleman for every single kernel launch, causing immense CPU overhead. In modern architectures, the Server acts only as a resource manager to establish **Direct Client Communication**.

When the Server accepts your custom contexts (Clients):

* It partitions the GPU's memory address space so all clients can share it (Unified Virtual Addressing).
* It allocates a dedicated **hardware work queue** (pushbuffer) directly on the GPU silicon for *each* of your 5 contexts.
* It maps these hardware queues directly into the memory space of your CPU process.

Once this setup is complete, the Control Daemon and Server step out of the critical execution path.

---

### Phase 4: Execution (Bypassing the Host Lock)

Now, your worker threads begin picking up tasks and executing kernels. Here is how your architecture bypasses the lock:

1. **Thread Binds Context:** A worker thread calls `cuCtxPushCurrent(isolated_context)`.
2. **Kernel Launch:** The thread calls `myKernel<<<...>>>`.
3. **Direct Hardware Submission:** Because each `isolated_context` was assigned its own distinct hardware queue by the MPS server, the CUDA driver does not need to acquire a global process lock. It writes the kernel launch commands *directly* to that context's specific hardware queue.
4. **Parallel Host Submission:** If all 4 worker threads call `myKernel<<<...>>>` at the exact same microsecond, they write to 4 different queues simultaneously. Zero host-side lock contention.
5. **Hardware Scheduling:** On the GPU silicon, the hardware scheduler (managing the single Shared Hardware Context created by the Server) sees 4 independent streams of work arriving instantly. It time-slices or spatially partitions the Streaming Multiprocessors (SMs) to run all 4 kernels side-by-side in real-time.

### Summary of the Data Flow

To visualize the pipeline during an active kernel launch:

**Your C++ Thread (Context A)** -> *writes directly to* -> **GPU Hardware Queue A**
**Your C++ Thread (Context B)** -> *writes directly to* -> **GPU Hardware Queue B**

The **MPS Daemon** and **MPS Server** are not involved in this hot path. They only exist to perform the initial setup that allows Context A and Context B to safely write to the same physical GPU simultaneously without crashing the driver.


Here is a visual mapping of both pipelines.

By comparing them side-by-side, you can clearly see where the standard driver enforces security boundaries, and where MPS bypasses them using its privileged access to the kernel.

### 1. NATIVE DRIVER PIPELINE (Strict Isolation, No Concurrency)

This is what happens when you run your program natively (without MPS) and 2 threads call `cuCtxCreate`. The public driver (`libcuda.so`) forces complete separation.

```text
       [ CPU Thread A ]                  [ CPU Thread B ]
              │ cuCtxCreate                     │ cuCtxCreate
              ▼                                 ▼
======================================================================
 USER SPACE
           [ libcuda.so (Public CUDA Driver) ]
           Rule: "One Context = One Isolated Environment"
======================================================================
 KERNEL SPACE
           [ nvidia.ko (Kernel Driver) ]
              │                                 │
              ▼                                 ▼
======================================================================
 GPU SILICON
      [ Hardware Context A ]            [ Hardware Context B ]
      (Private Page Tables)             (Private Page Tables)
      Contains: Hardware Queue A        Contains: Hardware Queue B
              │                                 │
              ▼                                 ▼
       [ GPU Scheduler ]                 [ GPU Scheduler ]
     (Runs Context A for 2ms)          (Switches to Context B)
                                      
         RESULT: TIME-SLICING (No Simultaneous Execution)

```

* **The Bottleneck:** The GPU silicon physically cannot read from Hardware Queue A and Hardware Queue B at the same time because they live in completely different security environments (different Page Tables). It must context-switch between them.

---

### 2. MPS PIPELINE (Shared Environment, True Concurrency)

This is what happens when you start the MPS daemon and run the exact same C++ code. The MPS Server intercepts your calls and uses undocumented APIs to ask the Kernel Driver for a special shared configuration.

```text
       [ CPU Thread A ]                  [ CPU Thread B ]
              │ cuCtxCreate                     │ cuCtxCreate
              └────────────────┐ ┌──────────────┘
                               ▼ ▼
======================================================================
 USER SPACE
           [ nvidia-cuda-mps-server (Privileged Daemon) ]
           Action: "Merge these clients into my existing Context"
                               │
======================================================================
 KERNEL SPACE
           [ nvidia.ko (Kernel Driver) ]
           Action: Allocates new Queues, maps them to ONE Context
                               │
======================================================================
 GPU SILICON
                [ SINGLE Shared Hardware Context ]
                     (Shared Page Tables)
                     /                  \
       [ Hardware Queue A ]        [ Hardware Queue B ]
       (Mapped to Thread A)        (Mapped to Thread B)
                   │                    │
                   ▼                    ▼
             [ Streaming Multiprocessors (SMs) ]
             (Runs Queue A & B Side-by-Side)
                                      
         RESULT: SPATIAL SHARING (True Hardware Concurrency)

```

### The Key Differences at a Glance

| Feature | Native Driver Pipeline | MPS Pipeline |
| --- | --- | --- |
| **Middleman** | None (direct to `libcuda.so`) | MPS Server daemon intercepts calls |
| **Page Tables on GPU** | 1 set *per* C++ Context | 1 set *total* (Shared by all) |
| **Hardware Queues** | Isolated inside separate Contexts | Grouped inside a single Context |
| **GPU Execution** | Time-sliced (Context Switching) | Concurrent (Spatial Sharing on SMs) |
| **Locking** | CPU threads wait on Global Lock | Threads write directly to MMIO Queues |