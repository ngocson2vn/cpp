# Could not guard on data-dependent expression u0 > 8192 (unhinted: u0 > 8192)
```Python
def get_bucket_m(m):
    target = 8192
    while target < m and target < 2**18:
        target *= 2
    
    return target
```
This graph break occurs because PyTorch Dynamo cannot compile a `while` loop when the loop condition depends on a symbolic input (`m`). <br/>
The compiler needs to know exactly how many times the loop will run to unroll it into the computation graph, but since m is dynamic, this is impossible.

**Solution**: To fix this, you must rewrite the iterative logic (the while loop) into a closed-form mathematical expression using standard arithmetic operations (max, min, log2). Dynamo can trace these operations into a symbolic graph without breaking.
```Python
def get_bucket_m(m):
    target = 8192
    m_floored = max(m, 8192)
    
    # Case 1: m < 8192
    # m_floored = 8192 = 2**13
    # math.log2(m_floored) = 13.0
    # math.ceil(13.0) = 13
    # next_pow2 = 2**13
    # target = 2**13
    
    # Case 2: m > 8192
    # m_floored = m
    # math.ceil(math.log2(m_floored)) = X
    # next_pow2 = 2**X
    # target = min(next_pow2, 2**18)
    next_pow2 = 2 ** math.ceil(math.log2(m_floored))
    target = min(next_pow2, 2**18)

    return target
```

# torch.einsum
```Python
query_proj = torch.einsum('bld,lde->ble', self._cast(query), self._cast(self.dense)) + self._cast(self.bias)
```

**Solution**:
```Python
query_proj = torch.matmul(self._cast(query).permute(1, 0, 2), self._cast(self.dense)).permute(1, 0, 2) + self._cast(self.bias)
```

The two operations are equivalent because they both perform a **position-wise matrix multiplication** where the weight matrix varies depending on the sequence position index.

Here is a breakdown of why they produce the exact same result.

### 1. Shape Assumptions

To make this clear, let's define the dimensions based on the Einstein summation string:

* $b$ = Batch Size ($B$)
* $l$ = Sequence Length ($L$)
* $d$ = Input Dimension ($D$)
* $e$ = Output Dimension ($E$)

We assume the input shapes are:

* `query`: $(B, L, D)$
* `self.dense` (Weight): $(L, D, E)$
* `self.bias`: $(E)$ or $(1, 1, E)$ (broadcastable)

---

### 2. Operation 1: The `einsum` Approach

```python
torch.einsum('bld,lde->ble', query, dense)
```

This is the most direct mathematical definition of the operation. The string `'bld,lde->ble'` tells PyTorch exactly how to compute the output:

* **Inputs:**
* Term 1 (`bld`): The query tensor at batch $b$, position $l$, feature $d$.
* Term 2 (`lde`): The weight tensor at position $l$, input feature $d$, output feature $e$.


* **Computation:**
It iterates over indices $b, l, e$ and sums over the contraction index $d$.
The mathematical formula for a single output element is:<br/>
$\hspace{1cm} \text{Output}_{b,l,e} = \sum_{d} \text{Query}_{b,l,d} \times \text{Weight}_{l,d,e}$

**Key Characteristic:** The index $l$ appears in both inputs. This means the operation uses a **unique weight matrix** for every position  in the sequence.

**Why $d$ is the contraction index?**<br/>
To understand why $d$ is the contraction index, we need to look at how the Einstein Summation (einsum) notation defines the relationship between inputs and outputs.

In `einsum`, the role of an index is determined entirely by **whether it appears in the output string (the part after `->`).**

### The Rule of Contraction

The notation follows a strict rule:

1. **Free Indices:** If an index appears in the input *and* the output (e.g., `b`, `l`, `e`), it is preserved. This dimension determines the shape of the result.
2. **Summation (Contraction) Indices:** If an index appears in the input strings but **does not** appear in the output string, PyTorch automatically sums over that dimension.

Let's apply this to your string: `'bld,lde->ble'`.

| Index | In Input 1 (`bld`)? | In Input 2 (`lde`)? | In Output (`ble`)? | Role |
| --- | --- | --- | --- | --- |
| **b** | Yes | No | **Yes** | **Free Index** (Preserved Batch dimension) |
| **l** | Yes | Yes | **Yes** | **Free Index** (Preserved Sequence dimension) |
| **e** | No | Yes | **Yes** | **Free Index** (Preserved Output dimension) |
| **d** | **Yes** | **Yes** | **NO** | **Contraction Index** (Summed over) |

Because `d` is missing from the right side of the arrow (`->ble`), the operation knows to multiply the values along the `d` axis and then add them all up to produce a single value for the output.

### Visualizing the Math

If we were to write this out as a nested loop (which is essentially what the GPU does in parallel), it would look like this:

```python
# b = batch, l = length, d = input_dim, e = output_dim
output = torch.zeros(B, L, E)

for b in range(B):          # Free index
    for l in range(L):      # Free index
        for e in range(E):  # Free index
            sum_val = 0
            # THIS is the contraction over 'd'
            for d in range(D):
                sum_val += query[b, l, d] * dense[l, d, e]
            
            output[b, l, e] = sum_val

```

### Why is $d$ the contraction index conceptually?

In the context of neural networks (specifically attention or linear layers), $d$ usually represents the **feature dimension** (or hidden size).

When you project a vector from one size ($D$) to another size ($E$), you are calculating a dot product between the input features and the weights. A dot product is, by definition, the sum of the products of corresponding entries.

* `query[b, l, :]` is a vector of size $D$.
* `dense[l, :, e]` is a vector of size $D$.
* To get the new value at `output[b, l, e]`, you multiply these two $D$-sized vectors element-wise and **sum** the result. That summation is the "contraction" of dimension $D$.


---

### 3. Operation 2: The `matmul` + `permute` Approach

```python
torch.matmul(query.permute(1, 0, 2), dense).permute(1, 0, 2)

```

This approach manipulates the tensor dimensions to leverage PyTorch's native "Batched Matrix Multiplication" rules to achieve the same result.

**Step-by-Step Execution:**

1. **Permute Query:**
* Input: $(B, L, D)$
* Operation: `.permute(1, 0, 2)` swaps the Batch and Length dimensions.
* New Shape: $(L, B, D)$


2. **Batched Matrix Multiplication (`matmul`):**
* Left Tensor $(A)$: $(L, B, D)$
* Right Tensor $(B)$: $(L, D, E)$ (The weights)
* **Logic:** When `torch.matmul` sees 3D tensors, it treats the **first dimension as the batch dimension**.
* It performs $L$ independent matrix multiplications. For each index $i$ in $L$: <br/>
$\hspace{1cm} \text{Result}[i] = A[i] \times B[i]$

* This effectively multiplies the slice `query[l]` (shape $B \times D$) by the slice `dense[l]` (shape $D \times E$).
* Intermediate Result Shape: $(L, B, E)$


3. **Permute Output:**
* Input: $(L, B, E)$
* Operation: `.permute(1, 0, 2)` swaps Length and Batch back to their original positions.
* Final Shape: $(B, L, E)$



### Summary of Equivalence

Both operations compute the exact same scalar value for every index.

If we look at the math inside the `matmul` for a specific batch  and position :

1. The `matmul` aligns the -th slice of the permuted query with the -th slice of the weight matrix.
2. It calculates the dot product of the query vector and the weight columns.
3. This is mathematically identical to: <br/>
$\hspace{1cm} \sum_{d} \text{Query}_{b,l,d} \times \text{Weight}_{l,d,e}$


**Why does the second version exist?**
While `einsum` is more readable, the `matmul` + `permute` pattern is sometimes used because:

1. **Optimization:** Historically, `torch.matmul` has been more heavily optimized in CUDA libraries (cuBLAS) than `einsum`, though `einsum` performance has improved significantly in recent versions.
2. **Legacy Support:** Older codebases or export formats (like ONNX) sometimes handle standard matrix multiplications better than general Einstein summations.
