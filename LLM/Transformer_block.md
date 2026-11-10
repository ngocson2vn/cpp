# Transformer Block
Below is one Transformer block taken apart op by op. I'm using Llama 3 8B's real dimensions and the same 5-token prompt, so every step has concrete shapes.

**Setup:** T = 5 tokens ("The cat sat on the"), `d_model = 4096`, `32 query heads` and `8 key/value heads` (grouped-query attention), `head_dim = 128`, MLP hidden size = 14,336.

---

## Step 0: Embedding (once, before block 1)

```
ids = [791, 8415, 7731, 389, 279]      # illustrative token ids
x   = E[ids]                           # E: [128256, 4096] → x: [5, 4096]
```

This is just a row lookup. Row *i* depends only on which token sits at position *i*. That's what "in isolation" means. Llama adds no positional vector here; position enters later, at step 3.

`x` is the **residual stream**. Everything below either reads from it or adds to it.

---

## Attention half: where positions talk to each other

### 1. RMSNorm
```
h = g₁ ⊙ x / sqrt(mean(x², axis=-1) + 1e-5)      # [5, 4096]
```
Each position is normalized independently, so nothing mixes yet. `x` itself is left unchanged; the block works on the normalized copy `h`.

### 2. Project to Q, K, V
```
Q = h @ W_Q    # [4096, 4096] → [5, 4096] → reshape [5, 32, 128]
K = h @ W_K    # [4096, 1024] → [5, 1024] → reshape [5,  8, 128]
V = h @ W_V    # [4096, 1024] → [5, 1024] → reshape [5,  8, 128]
```
Every position produces three things:
- a **query**: "what am I looking for?"
- a **key**: "what do I contain?"
- a **value**: "what I'll hand over if picked."

This step is still per-position.

### 3. RoPE (rotary position encoding)
Each 128-dim q and k vector is split into 64 pairs of numbers. Pair *j* at position *p* is rotated by the angle p·θⱼ. V is not rotated. Because of the rotation, q_i · k_j ends up depending on the relative offset i − j. This is the only place position information enters the model.

### 4. Attention scores
```
S[h, i, j] = Q[i, h] · K[j, h//4] / sqrt(128)     # [32, 5, 5]
```
Each of the 8 KV heads is shared by 4 query heads (32 / 8). Every query is scored against every key.

Here is each symbol, with its range in the 5-token example:

| Symbol | Meaning | Range / shape |
|---|---|---|
| `h` | Which **query head** | 0 … 31 |
| `i` | The **query position**: the position doing the looking | 0 … 4 |
| `j` | The **key position**: the position being looked at | 0 … 4 |
| `Q[i, h]` | Query vector of position *i* for head *h* | 128 numbers |
| `K[j, h//4]` | Key vector of position *j* from KV head *h//4* | 128 numbers |


**The result: one number per (head, looker, lookee)**

`S[h, i, j]` is a single scalar: how strongly position *i*, through head *h*, wants to read from position *j*. This is before masking and softmax. The full tensor holds 32 × 5 × 5 = 800 of these numbers.

A concrete entry is `S[6, 4, 1]`:
- head 6,
- the query comes from position 4 ("the"),
- the key comes from position 1 ("cat"),
- the KV head used is 6 // 4 = 1.

```
S[6, 4, 1] = Q[4, 6, :] · K[1, 1, :] / sqrt(128)
```

Two properties are worth noting:

- **Asymmetry.** `S[h, i, j]` is generally not equal to `S[h, j, i]`. One side uses a query and the other a key, and those come from different weight matrices (W_Q vs. W_K). "How much *the* wants *cat*" is a different question from "how much *cat* wants *the*". With the causal mask, `S[h, 1, 4]` gets set to −∞ anyway.
- **Matrix form.** For one head, the whole `[5, 5]` slice is a single matrix multiply, with rows indexed by *i* and columns by *j*:
  ```
  S[h] = Q[:, h, :] @ K[:, h//4, :].T      # [5, 128] @ [128, 5] → [5, 5]
  ```

**5. Causal mask**
`S[h, i, j] = −∞` wherever `j > i`:
```
            keys:  The  cat  sat  on   the
query The           ✓    ·    ·    ·    ·
query cat           ✓    ✓    ·    ·    ·
query sat           ✓    ✓    ✓    ·    ·
query on            ✓    ✓    ✓    ✓    ·
query the           ✓    ✓    ✓    ✓    ✓
```

**6. Softmax, row by row**
```
A = softmax(S, axis=-1)                            # [32, 5, 5]
```
Row *i* becomes a set of weights over positions 0..i that sum to 1. For example, one head's row for "the" might come out as [0.05, 0.60, 0.15, 0.10, 0.10] (numbers made up for illustration). That head would be mostly "looking at" "cat".

**7. Weighted sum of values: the step where information moves**
```
O[i, h] = Σⱼ A[h, i, j] · v[j, h//4]               # [5, 32, 128]
```
Position *i* receives a blend of value vectors from earlier positions. **This is the only operation in the whole block that moves information between positions.**

**8. Concatenate heads and project**
```
attn_out = concat(O) @ W_O                         # [5, 4096] @ [4096, 4096] → [5, 4096]
```

**9. Residual add**
```
x = x + attn_out                                   # [5, 4096]
```
The attention half doesn't replace `x`. It writes a correction on top of it.

---

## MLP half: per-position processing

**10. RMSNorm** (its own learned gain `g₂`)
```
h = g₂ ⊙ x / sqrt(mean(x², axis=-1) + 1e-5)        # [5, 4096]
```

**11. Gate and up projections**
```
gate = h @ W_gate      # [4096, 14336] → [5, 14336]
up   = h @ W_up        # [4096, 14336] → [5, 14336]
```

**12. SwiGLU activation**
```
a = silu(gate) ⊙ up                                # [5, 14336]
```
The gate decides which of the 14,336 hidden features fire for this token.

**13. Down projection**
```
mlp_out = a @ W_down   # [14336, 4096] → [5, 4096]
```

**14. Residual add**
```
x = x + mlp_out                                    # [5, 4096]
```

The MLP treats each position separately. It works on whatever attention just gathered into that position's vector. The MLP is also where most of the parameters live: about 176M per block versus about 42M for attention.

---

## Stacking: why "one running stream"

Repeat steps 1–14 for 32 blocks. Each block has its own weights. Unrolling the residual adds gives:

```
x_final = x₀ + (attn₁ + mlp₁) + (attn₂ + mlp₂) + … + (attn₃₂ + mlp₃₂)
```

So the final vector at each position is literally **the token's embedding plus 64 additive updates**. This sum view explains several things:
- Each sublayer reads a normalized snapshot of everything written so far and adds its own delta.
- A late layer can directly use something an early layer wrote, with no chain of transformations in between.
- Gradients get a direct identity path back through all 64 additions, which is part of why deep stacks train stably.
- The logit lens works because the stream stays in one shared 4096-d space throughout.

---

## Summary: which steps mix positions

| Step | Operation | Mixes positions? |
|---|---|---|
| 1, 10 | RMSNorm | No |
| 2, 8, 11–13 | Matmuls | No (each of the T rows is independent) |
| 3 | RoPE | No (uses the position index, but per row) |
| 4–7 | Scores → mask → softmax → weighted sum | **Yes** |
| 9, 14 | Residual add | No |

Most of the block is the same computation run independently on T rows, which parallelizes trivially. Only steps 4–7 have a T×T dependency. During generation, the new token has T = 1, so step 4 becomes one query against the cached K and V from all earlier positions. That T×T dependency is why the KV cache exists.

If it would help, I can write a small runnable NumPy script that pushes random weights through one block and prints every shape, or walk through RoPE with actual numbers.