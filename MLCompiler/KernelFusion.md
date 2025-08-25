# Vertical fusion across multiple ops
```Python
buffers:
  x: f32[M, K]
  w: f32[M, K]
  b: f32[1]        # scalar/broadcast
  z: f32[M, K]

loops:
  for m in 0..M:
    for k in 0..K:
      t0 = x[m, k] * w[m, k]
      t1 = t0 + b[0]
      t2 = 1.0 / (1.0 + exp(-t1))   # sigmoid
      z[m, k] = t2
```
Notes:
- No temporaries are materialized as buffers; they are SSA scalars within the loop body.
- If multiple outputs share the same iteration space, they appear as additional stores in the same loop.