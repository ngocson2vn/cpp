# Online Softmax Kernel
Given a 2D tensor and let `x` denote row `r`, <br/>
the softmax formula for row `x` is as follows:
```
        exp(x - max)     
y = ---------------------
      sum(exp(x - max))
```

where
- `max = max(x)` is per row `r`
- `sum(exp(x - max))` is per row `r`
The softmax is computed for each row independently.

```
                         exp(x0 - max)                                   
y0 = ----------------------------------------------------------
      exp(x0 - max) + exp(x1 - max) + ... + exp(x_{N-1} - max)
```

First we focus on computing a **running max** `m` and a **running sum** `s`,

## One kernel with only 1 CTA per row
- Each thread computes a chunk of a row
- Need to reduce both `m` and `s` across warps
- Lane 0 of each warp writes (m, s) pair to `smemMaxs[warpIdx]` and `smemSums[warpIdx]`
- Sync all threads in CTA
- Warp 0 reduces `smemMaxs` and `smemSums`; lane 0 writes the final (m, s) to SMEM
- Sync all threads in CTA
- Every thread reads the final (m, s) and compute $`y_i = \exp(x_i - m) / s`$

## Two kernels with many CTAs per row
**The first kernel**: <br/>
- Each CTA is responsible for a chunk of a row
- Each CTA computes its own (m, s) for its chunk of the row
- Eahc CTA writes its own (m, s) to GMEM

**The second kernel**: <br/>
- Each CTA computes the final (m, s) for its row => Other CTAs for the same row recompute the final (m, s) too.
- Each CTA writes the final (m, s) to SMEM
- Each thread in a CTA read the final (m, s) and computes $`y_i = \exp(x_i - m) / s`$
