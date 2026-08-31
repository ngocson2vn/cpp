# tl.max_contiguous
```python
rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
```

### `tl.max_contiguous(..., BLOCK_M)` (The Contiguity Hint)

* **What it does:** This is another compiler hint that asserts: *"Even though I just used a modulo operation, I guarantee these indices are perfectly contiguous (step-by-step sequential) for at least `BLOCK_M` elements."*. For example, `[0, 1, 2, 3, 8, 9, 10, 11]` has contiguity 4.

* **Why it matters:** Because the compiler previously lost track of contiguity at the `% M` step, this function forcefully restores that knowledge.

---

### The Big Picture: Why do this?

If you simply wrote `offs_a_m = rm % M`, the math would be perfectly correct, but the **performance would tank (fail completely)**.

Because the Triton compiler cannot mathematically prove that `rm % M` is contiguous and aligned, it plays it safe. It will fall back to **scalar memory accesses**—instructing the GPU to load elements one by one.

By wrapping the calculation in `tl.max_contiguous` and `tl.multiple_of`, you are restoring the compiler's confidence. This unlocks the compiler's ability to emit highly optimized PTX assembly code, such as:

* **Vectorized Loads:** e.g., fetching 4 floats at a time using `ld.global.v4`.
* **Asynchronous Copies:** Using `cp.async` (on Ampere GPUs) or TMA (Tensor Memory Accelerator on Hopper GPUs) to pipeline memory fetches straight into shared memory without tying up the compute cores.

In short: **It calculates the wrapped block indices while guaranteeing memory alignment and contiguity, allowing Triton to maximize memory bandwidth.**