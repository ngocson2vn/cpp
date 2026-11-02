# RMS Norm
Given a tensor `x`,
```cpp
// Step 1: Compute Mean Square along dimension `d`
// Usually, `d` is the last (innermost) dimension
ms = sum_d(e_d * e_d) / N // `e_d` are elements along dimension `d`

// Step 2: Compute Root Mean Square
rms(x) = sqrt(ms + eps)

// Step 3: Normalize `x`
y = x / rms(x) * w // <-- Elementwise multiply
```

If `x` is a 2D tensor, then compute RMSNorm for the last dimension, i.e, RMSNorm per row.
