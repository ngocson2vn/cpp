# Kernel Fusion
In CUDA programming, may the following claims be all true?

Implement a post-inductor fusion strategy to reduce the number of kernel launches => reduce kernel launch overhead => reduce overall inference latency. In addition, when two kernels are fused into a larger kernel, the larger kernel may keep SMs stay active for more consecutive cycles instead of leaving them idle while waiting for new workloads to be submitted by next kernel launch.

Yes, **all of those claims are absolutely true.** In fact, they perfectly describe the core motivations behind kernel fusion in modern deep learning compilers like PyTorch's TorchInductor, XLA, and TensorRT.

Here is a breakdown of why your reasoning is exactly correct from a CUDA hardware and software perspective:

### 1. Fewer Launches = Less Launch Overhead

Every time the CPU launches a CUDA kernel, there is a fixed cost. The CPU has to execute the CUDA runtime API, the NVIDIA driver has to process the command, and the command must be pushed into the GPU's hardware queue.

* If you have dozens of small operations (like element-wise additions, activations, or layer norms), the GPU might finish calculating the math faster than the CPU can queue up the next operation.
* This makes the workload **CPU-bound** (or "launch-bound"). Fusing these into one kernel removes those redundant CPU-side API calls.

### 2. Reduced Overhead = Reduced Inference Latency

Because deep learning inference is highly sequential, the CPU must queue the next layer, and the GPU must execute it before moving on. By eliminating the microsecond-level delays of kernel launch overhead across hundreds of layers in a model, you directly shave off milliseconds of total end-to-end inference latency.

### 3. Keeping SMs Active (Eliminating Inter-Kernel Gaps)

Your claim about Streaming Multiprocessors (SMs) staying active is highly accurate and refers to the **"inter-kernel gap"** and the **"tail effect"**.

* **The Tail Effect:** As a kernel finishes, some thread blocks complete before others. During this "tail," many SMs sit idle because there are no more thread blocks left to schedule for that specific kernel.
* **The Gap:** Once the kernel fully finishes, the GPU scheduler must fetch the next kernel from the queue, map its grid, and dispatch new thread blocks. This leaves a gap where the entire GPU does nothing.
* **The Fusion Fix:** By fusing kernels, you give the GPU a larger grid of work. As soon as a thread block finishes calculating the first operation (e.g., a matrix multiplication), it immediately moves into the next operation (e.g., a ReLU activation) within the same thread. The SMs never have to drain and wait for a new kernel to be scheduled.

### The Hidden Bonus: Memory Locality

While you correctly identified the launch and scheduling benefits, there is one more massive advantage to your fusion strategy: **VRAM bandwidth**.

Without fusion, Kernel A writes its output to High Bandwidth Memory (HBM/VRAM), and Kernel B immediately reads it back. With fusion, the output of operation A can be kept directly in the SM's ultra-fast **registers** or **shared memory** and fed right into operation B. Skipping that slow round-trip to VRAM often saves more latency than the launch overhead itself.
