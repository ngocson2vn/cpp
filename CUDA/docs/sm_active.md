# SM Active

## How SM Active is measured
<img src="./sam_active.jpeg" width="50%"/>

NVIDIA GPUs are composed of multiple **Streaming Multiprocessors (SMs)**. Each SM is responsible for executing threads in groups of 32, known as warps.

**SM Active** (often labeled as `sm__active_cycles.avg.pct_of_peak_sustained_elapsed` in NVIDIA's profiling tools) represents the percentage of time that at least one warp is active—meaning it is resident on the SM and either executing, waiting on memory, or stalling.

This metric is typically aggregated across the entire GPU. You can define the relationship mathematically as:

$$\text{Overall SM Active} = \frac{\sum_{i=1}^{N_{\text{SM}}} \text{Active Cycles for SM}_i}{N_{\text{SM}} \times \text{Elapsed Cycles}} \times 100$$

Where $N_{\text{SM}}$ is the total number of Streaming Multiprocessors on the GPU.

If your GPU has 100 SMs and only 45 of them have warps assigned to them while the rest sit completely idle, your overall SM Active will report as 45%, even if those 45 SMs are working as hard as possible.

