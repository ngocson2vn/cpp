# Tensor Slicing
```python
offs_k = tl.arange(0, BLOCK_K)

# Later
for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
    a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
    b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
```

`offs_k` is 1D, but `offs_k[None, :]` is still valid. Triton tensors implement a small slice of NumPy/PyTorch indexing: `:` is a no-op and `None` is `newaxis`.

## What the expression actually does

```python
offs_k = tl.arange(0, BLOCK_K)   # shape: (BLOCK_K,)
offs_k[None, :]                  # shape: (1, BLOCK_K)
```

`None` inserts a new axis of length 1. 

**Mental model**:

Think of it as building an outer-product grid of indices, not as Python list indexing:

```text
offs_k          = [0, 1, 2, ..., BLOCK_K-1]

offs_k[None, :] = [[0, 1, 2, ..., BLOCK_K-1]]          # 1 row, BLOCK_K cols

offs_k[:, None] = [[0],
                   [1],
                   [2],
                   ...]                                # BLOCK_K rows, 1 col
```

Adding those two shapes is how Triton constructs blocked 2D pointer arithmetic without an explicit nested loop.

The two forms you see everywhere are:

| Expression | Equivalent | Resulting shape |
|---|---|---|
| `offs_k[None, :]` | `offs_k.expand_dims(0)` | `(1, BLOCK_K)` — a row |
| `offs_k[:, None]` | `offs_k.expand_dims(1)` | `(BLOCK_K, 1)` — a column |

So it is not “indexing into a 1D array with two indices.” It is reshaping by adding a unit dimension.

## Why kernels write it that way

You need a 2D grid of addresses, not a 1D vector. In the standard matmul tutorial:

```python
offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
offs_k  = tl.arange(0, BLOCK_K)                    # (BLOCK_K,)

a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#                   (BLOCK_M, 1)                    (1, BLOCK_K)
#                   ---------------- broadcast ----------------
#                              (BLOCK_M, BLOCK_K)
```

Broadcasting rules are the same as NumPy: a dim of size 1 stretches to match the other operand.

- `offs_am[:, None]` repeats each row offset across every K column.
- `offs_k[None, :]` repeats each K offset down every M row.
- The sum is the full 2D pointer tile for `A[m:m+BLOCK_M, k:k+BLOCK_K]`.

Same idea for the mask:

```python
mask = offs_k[None, :] < K   # (1, BLOCK_K) broadcasts over the loaded tile
```
