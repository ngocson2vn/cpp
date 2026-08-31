# Triton GEMM
## Grid Size
Normally, grid size for a Triton GEMM is computed as follows, correct?
```python
BLOCK_M = 32
BLOCK_N = 64
grid_m = cdiv(M, BLOCK_M)     # number of tiles along M
grid_n = cdiv(N, BLOCK_N)     # number of tiles along N

grid = (grid_m * grid_n, 1, 1)
```

Yes — **for a typical 2D Triton GEMM where each program instance computes one `(BLOCK_M, BLOCK_N)` output tile**, this is correct.

The reasoning is:

* `grid_m = ceil(M / BLOCK_M)` → number of tiles needed to cover the `M` dimension.
* `grid_n = ceil(N / BLOCK_N)` → number of tiles needed to cover the `N` dimension.
* Therefore, the total number of output tiles/program instances is:

$`\text{grid\_m} \times \text{grid\_n}`$.

For example, if:

```text
M = 100
N = 200
BLOCK_M = 32
BLOCK_N = 64
```

then:

```python
grid_m = ceil(100 / 32) = 4
grid_n = ceil(200 / 64) = 4
```

so there are:

```python
4 × 4 = 16
```

program instances, with each program computing one output tile (except boundary tiles, which may be partially masked).

### One important Triton detail

The 1D launch grid is represented simply as:

```python
grid = (grid_m * grid_n,)
```

The kernel then typically converts the 1D `pid` into 2D tile coordinates:

```python
pid = tl.program_id(0)

pid_m = pid // grid_n
pid_n = pid % grid_n
```

So conceptually:

```text
                      N
        ┌──────┬──────┬──────┬──────┐
        │ P0,0 │ P0,1 │ P0,2 │ P0,3 │
        ├──────┼──────┼──────┼──────┤
        │ P1,0 │ P1,1 │ P1,2 │ P1,3 │
   M    ├──────┼──────┼──────┼──────┤
        │ P2,0 │ P2,1 │ P2,2 │ P2,3 │
        ├──────┼──────┼──────┼──────┤
        │ P3,0 │ P3,1 │ P3,2 │ P3,3 │
        └──────┴──────┴──────┴──────┘
```

There are `grid_m * grid_n` Triton **program instances**, each responsible for one output tile.

The main caveat is that this is **not universally the grid calculation for every Triton GEMM**. More sophisticated GEMMs may use **grouped/swizzled program IDs**, persistent kernels, or additional dimensions such as batch size. But for the standard tiled GEMM you're describing, yes, your calculation is correct.


## L2 Swizzle
L2 swizzle is **not** a change to the math. It only changes **which output tile each program (`pid`) computes**, so that programs that share an `A` or `B` tile run close together in time and hit L2 instead of HBM.

In your kernel it is almost a no-op, because there is only one N-tile. The same code is the standard Triton GEMM pattern, so it is worth understanding on a 2-D tile grid first.

---

### What a "tile grid" is

The output `C` is `[M, N]`. Each program writes one block:

```text
BLOCK_M = 32
BLOCK_N = 64

grid_m = cdiv(M, 32)     # how many tiles down
grid_n = cdiv(N, 64)     # how many tiles across
```

So `C` is a `grid_m × grid_n` chessboard of tiles. Program `pid` must pick one square `(pid_m, pid_n)` and compute that block.

Naive mapping (row-major over the chessboard):

```text
pid_m = pid // grid_n
pid_n = pid %  grid_n

                      N
        ┌──────┬──────┬──────┬──────┐
        │ P0,0 │ P0,1 │ P0,2 │ P0,3 │
        ├──────┼──────┼──────┼──────┤
        │ P1,0 │ P1,1 │ P1,2 │ P1,3 │
   M    ├──────┼──────┼──────┼──────┤
        │ P2,0 │ P2,1 │ P2,2 │ P2,3 │
        ├──────┼──────┼──────┼──────┤
        │ P3,0 │ P3,1 │ P3,2 │ P3,3 │
        └──────┴──────┴──────┴──────┘
```

Launch order then walks **left-to-right across a row of tiles, then the next row**.

---

### Why launch order matters for L2

Each tile `(pid_m, pid_n)` loads:

- a `A` slice with size `BLOCK_M × K` at `pid_m` along K dimension
- a `B` slice with size `K × BLOCK_N` at `pid_n` along K dimension

Reuse:

- every tile in the **same row** `pid_m` wants the **same `A` slice**, different `B`
- every tile in the **same column** `pid_n` wants the **same `B` slice**, different `A`

`A`/`B` slices are far larger than shared memory / L1. The only place they can stay hot (alive) across **different CTAs** is **L2**.

The GPU roughly schedules CTAs in increasing `pid` (not a hard guarantee, but good enough that this heuristic works). So the order you assign `(pid_m, pid_n)` to `pid` is the order those slices are pulled through L2.

Row-major launch, `grid_n` large:

```text
(0,0) (0,1) (0,2) ... (0, grid_n-1)  ← A-row 0 reused, B-col 0 used once
(1,0) (1,1) ...                      ← B-col 0 needed again much later
```

By the time you come back to column 0, that `B` slice has usually been evicted from L2. You need to reload it from HBM for every row.

---

### Grouped mapping (the "L2 swizzle")

Triton's usual fix: take `GROUP_M` rows of the tile grid as a **group**, and inside a group walk **down M first, then across N**.

Concrete example, `grid_m = 8`, `grid_n = 4`, `GROUP_M = 4`:

```text
group_size = GROUP_M * grid_n = 4 * 4 = 16
```

`pid` 0..15 (group 0) map to:

```text
pid:    0    1    2    3    4    5    6    7    8 ...
pid_m:  0    1    2    3    0    1    2    3    0
pid_n:  0    0    0    0    1    1    1    1    2
```

On the chessboard, launch order is:

```text
        n=0    n=1    n=2    n=3
m=0     1st    5th    9th    13th
m=1     2nd    6th    10th   14th
m=2     3rd    7th    11th   15th
m=3     4th    8th    12th   16th     ← end of group 0
m=4     17th   ...                    ← group 1 starts
```

Inside the group you finish a **column of 4 tiles** before moving to the next N-tile.

What that does to cache:

1. CTAs 1–4 all need **the same `B` slice** (`n=0`). Four consumers back-to-back → `B` stays in L2.
2. Those four `A` slices (`m=0..3`) stay resident while the group sweeps `n=0,1,2,3`. Each `A` slice is reused `grid_n` times before the group ends.

Working set for one group is roughly:

```text
GROUP_M  A-slices   +   grid_n  B-slices
```

`GROUP_M` is a knob: bigger group → more `B` reuse, but more `A` slices must fit in L2 at once. 8 is a common compromise.

---

### Incomplete last group

If `grid_m` is not a multiple of `GROUP_M`, the last group is shorter:

```python
group_height = min(grid_m - group_id * GROUP_M, GROUP_M)
```

Example: `grid_m = 10`, `GROUP_M = 8` → group 0 has 8 rows, group 1 has 2.

`pid_group % group_height` on that last group is only a **permutation** of the remaining row indices (not lost tiles). Every tile is still computed exactly once. `pid_n` stays in range because `group_size` still uses the full `GROUP_M * grid_n` stride for group boundaries.

---

### Tiny mental model

```text
Without swizzle (row-major tiles):
  reuse A well along a row
  reuse B poorly  (B comes back only after a full row of N-tiles)

With GROUP_M swizzle (column-then-row inside a group):
  reuse B immediately  (GROUP_M CTAs share one B slice)
  reuse those GROUP_M A slices for the whole N-sweep of the group
```

It is a **scheduling permutation** of `pid → (pid_m, pid_n)`, not extra arithmetic on the GEMM itself. After `pid_m` / `pid_n` are chosen, your kernel is ordinary blocked matmul:

```text
rm = pid_m * 32 + arange(32)
rn = pid_n * 64 + arange(64)
```

---

### How to see it
Run [./l2_swizzle/pattern.py](./l2_swizzle/pattern.py)

Watch `(pid_m, pid_n)` walk down a strip of 8 rows, then right by one, rather than racing across a full row first.

That walk is the entire L2 swizzle.
