# Cache dir
```
export TRITON_CACHE_DIR=./tmp
```

# GPU arch
```Bash
export TRITON_OVERRIDE_ARCH=sm86
```

# PTX version
```Python
# /data00/home/son.nguyen/workspace/triton_dev/bytedance/triton/python/triton/backends/nvidia/compiler.py
class CUDABackend(BaseBackend):
    def parse_options(self, opts) -> Any:
        ...
        # PTX version
        try:
            ptx_version = int(os.getenv("TRITON_OVERRIDE_PTX_VERSION", "unset"))
            args["ptx_version"] = ptx_version
        except:
            pass

        return CUDAOptions(**args)

```

# Errors
```Python                                               ^
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1] NameError('_triton_helper_fn_add0 is not defined')
```
There is a bug in TritonComboKernels
<br/>

# GROUP_SIZE_M=8
`GROUP_SIZE_M=8` indicates that each group is designed to span up to 8 program IDs (blocks) along the M dimension (i.e., up to 8 tile-rows in the output grid). This defines the "height" of each grouped strip in the block scheduling.

Let's assume M=1024 (16x64, so num_pid_m=16), N=256 (4x64, so num_pid_n=4), keeping K=128 and block sizes=64 unchanged.<br/>
This creates a 16x4 grid of blocks (total 64 programs).

Here, `num_pid_in_group = 8 * 4 = 32`. There are two full groups:
- Group 0: pids 0–31, covering block-rows (pid_m) 0–7 across all 4 block-columns (pid_n).
- Group 1: pids 32–63, covering pid_m 8–15 across all pid_n.

Within each group (a horizontal "strip" of 8 block-rows by 4 block-columns), the pid assignment follows a column-major order: it fills down each block-column (vertically along pid_m for a fixed pid_n) before moving to the next block-column. This is the key effect of `GROUP_SIZE_M=8` — it prioritizes vertical grouping within the strip for better cache reuse (e.g., sharing columns of B across consecutive blocks).
ASCII diagram of the block grid (each cell shows the assigned pid for that (pid_m, pid_n) position; brackets indicate groups):
```
pid_m \ pid_n |   0     1     2     3
--------------|----------------------------
   0          |   0     8    16    24
   1          |   1     9    17    25    <--- Group 0: pids 0-31
   2          |   2    10    18    26      (strip of 8 block-rows;
   3          |   3    11    19    27       column-major order within)
   4          |   4    12    20    28
   5          |   5    13    21    29
   6          |   6    14    22    30
   7          |   7    15    23    31
--------------|----------------------------
   8          |  32    40    48    56
   9          |  33    41    49    57    <--- Group 1: pids 32-63
  10          |  34    42    50    58      (another strip, same order)
  11          |  35    43    51    59
  12          |  36    44    52    60
  13          |  37    45    53    61
  14          |  38    46    54    62
  15          |  39    47    55    63
```

# Autotune
```Bash
export TRITON_PRINT_AUTOTUNING="1"
```