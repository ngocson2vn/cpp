# Skill Set
## 1. Inference & Compute Optimization
Design and implement highly optimized inference pipelines and computational kernels to accelerate neural network workloads, leveraging low-level techniques such as SIMD vectorization, cache-aware memory access patterns, and hardware-specific tuning.

## 2. Neural Network Compression & Model Optimization
Research and implement **pruning**, **quantization**, and other compression techniques to reduce model size and accelerate inference while preserving accuracy across Large Recommendation Model (LRM) workloads.

## 3. Profiling & Observability
Build and utilize advanced profiling tools to identify bottlenecks across the inference stack—from memory bandwidth and cache utilization to CPU-side data preprocessing stalls and end-to-end pipeline throughput.

## 4. Evaluation & Benchmarking
Design and maintain rigorous evaluation and benchmarking frameworks for systematic model comparison across optimization configurations. Develop automated pipelines (e.g., LLM-as-a-judge) to measure the impact of optimization techniques on model quality and performance.


# Others
The work is hands on, alongside ML researchers, with ownership across inference stack. It includes
- writing fused kernels,
- developing an holistic optimization pipeline to apply a series of optimization passes
- developing a ML compiler to fuse TF ops and lower them to CUBIN
- digging into framework or compiler internals,
- improving memory access, or 
- removing the communication bottleneck holding back an inference session

