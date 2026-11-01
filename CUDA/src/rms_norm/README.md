# RMS Norm
```
m = sum_i(x_i * x_i) / N
rms(x) = sqrt(m + eps)
y = x / rms(x) * w <-- Elementwise multiply
```

If `x` is a 2D tensor, then compute RMSNorm per row.
