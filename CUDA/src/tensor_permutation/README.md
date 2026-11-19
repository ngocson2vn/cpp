# Tensor Permutation
The main purpose of a permutation operation is to reorder dimensions.

Given tensor `t1`:
```python
import torch

M = 5
N = 32
K = 128
t1 = torch.rand(M, N, K, dtype=torch.float32)
```
`t1` has shape `(M, N, K)` and strides `(N * K, N, 1)`.

## Case 1: Permute `t1`
```python
t2 = t1.permute(2, 1, 0)
```
- `t2` shares the same buffer with `t1`.
- Both shape and strides of `t1` are permuted.
- `t2` has shape `(K, N, M)` and strides `(1, N, N * K)`.
- t1[m, n, k] == t2[k, n, m]


## Case 2: Permute `t1` and copy it
```python
t2 = t1.permute(2, 1, 0).contiguous()
```
- `t2` has a completely new buffer.
- The shape of `t2` is the permuted shape of `t1`.
- `t2` has shape `(K, N, M)` and contiguous strides `(N * M, N, 1)`.
- `t1[m, n, k]` is copied to `t2[k, n, m]`.

In memory, `t1`'s element at offset `m*4096 + n*128 + k` is written to offset `k*160 + n*5 + m` in the new buffer.
