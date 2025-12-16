# Environment Variables
```Bash
# Cache dir
export TRITON_CACHE_DIR=./tmp

# Autotune
export TRITON_PRINT_AUTOTUNING="1"

# GPU arch
# This is a self-added env var
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

# How Triton creates CUtensorMap objects
Before Triton launches a TMA-based GEMM kernel, it needs to create CUtensorMap objects for matrices A, B, and D.<br/>
Steps are as follows:
### At compile phase
**nvidia/backend/compiler.py**
```Python
pm.run(mod, 'make_ttgir')
metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
```
`"tensordesc_meta"` is retrieved from the MLIR module.<br/>

**python/src/ir.cc**
```C++
py::list getTensorDescMetadata(ModuleOp &mod) {
  // ...
}
```

### At runtime
The `"tensordesc_meta"` is passed into runtime as follows:
```Python
ipdb> where
  /data05/home/son.nguyen/workspace/triton.cpp/gemm_hopper.py(202)<module>()
    200 
    201 if __name__ == "__main__":
--> 202     main()
  /data05/home/son.nguyen/workspace/triton.cpp/gemm_hopper.py(180)main()
    179 
--> 180     c_matmul_tma = matmul_tma(a, b, warp_specialize=True)
    181     print(f"c_matmul_tma: shape={c_matmul_tma.shape} dtype={c_matmul_tma.dtype} {c_matmul_tma}")
  /data05/home/son.nguyen/workspace/triton.cpp/gemm_hopper.py(138)matmul_tma()
    137 
--> 138     kernel = matmul_kernel_tma[grid](
    139         a_desc, b_desc, c_desc,
  /data05/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/triton/runtime/jit.py(419)<lambda>()
    418         
--> 419         return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
    420         # return cast(T, functools.partial(cast(Callable, self.run), grid=grid))
  /data05/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/triton/runtime/autotuner.py(240)run()
    239                 else:
--> 240                     benchmark()
    241 
  /data05/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/triton/runtime/autotuner.py(229)benchmark()
    228                     bench_start = time.time()
--> 229                     timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
    230                     bench_end = time.time()
  /data05/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/triton/runtime/autotuner.py(229)<dictcomp>()
    228                     bench_start = time.time()
--> 229                     timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
    230                     bench_end = time.time()
  /data05/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/triton/runtime/autotuner.py(163)_bench()
    162         try:
--> 163             return self.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))
    164         except (OutOfResources, CompileTimeAssertionFailure, PTXASError) as e:
  /data05/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/triton/testing.py(149)do_bench()
    148 
--> 149     fn()
    150     di.synchronize()
  /data05/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/triton/runtime/autotuner.py(149)kernel_call()
    148                 print(f"args: {args}, current: {current}")
--> 149                 self.fn.run(
    150                     *args,
  /data05/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/triton/runtime/jit.py(756)run()
    755             # launch kernel
--> 756             launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
    757             kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
  /data05/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/triton/compiler/compiler.py(490)launch_metadata()
    489             return None
--> 490         self._init_handles()
    491         ret = LazyDict({"name": self.name, "function": self.function, "stream": stream})
  /data05/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/triton/compiler/compiler.py(460)_init_handles()
    459         # create launcher
--> 460         self._run = driver.active.launcher_cls(self.src, self.metadata)
    461         # not enough shared memory to run the kernel
  /data05/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/triton/backends/nvidia/driver.py(696)__init__()
    695         self.num_ctas = functools.reduce(operator.mul, metadata.cluster_dims, 1)
--> 696         self.launch = wrap_handle_tensordesc(mod.launch, signature, tensordesc_meta)
    697         self.global_scratch_size = metadata.global_scratch_size
> /data05/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/triton/backends/nvidia/driver.py(654)wrap_handle_tensordesc()
    653     ipdb.set_trace()
--> 654     has_tensor_desc_arg = any(isinstance(sig, str) and sig.startswith("tensordesc") for sig in signature.values())
    655     if not has_tensor_desc_arg:

ipdb>
```
<br/>

**/data05/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/triton/backends/nvidia/driver.py**
```Python
def make_tensordesc_arg(arg, metadata):
    swizzle = metadata["swizzle"]
    elem_size = metadata["elem_size"]
    elem_type = metadata["elem_type"]
    block_size = metadata["block_size"]
    fp4_padded = metadata["fp4_padded"]

    cu_tensor_map = triton.runtime.driver.active.utils.fill_tma_descriptor(
        arg.base.data_ptr(),
        swizzle,
        elem_size,
        TMA_DTYPE_DEVICE_TO_HOST[elem_type],
        block_size,
        shape,
        strides,
        padding,
    )

    return [cu_tensor_map, *shape, *strides]
```
Note that `fill_tma_descriptor` comes from a dynamically loaded solib which is built on-the-fly from **nvidia/backend/driver.c**<br/>

**/data05/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/triton/runtime/build.py**
```Python
def compile_module_from_src(src: str, name: str, library_dirs: list[str] | None = None,
                            include_dirs: list[str] | None = None, libraries: list[str] | None = None,
                            ccflags: list[str] | None = None) -> ModuleType:
    key = hashlib.sha256((src + platform_key()).encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    print(f"  ==> Cache key: {cache.key}")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    cache_path = cache.get_file(f"{name}{suffix}")

    if cache_path is not None:
        print(f"cache_path: {cache_path}")
        try:
            return _load_module_from_path(name, cache_path)
        except (RuntimeError, ImportError):
            log = logging.getLogger(__name__)
            log.warning(f"Triton cache error: compiled module {name}.so could not be loaded")

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, name + ".c")
        with open(src_path, "w") as f:
            f.write(src)
        so = _build(name, src_path, tmpdir, library_dirs or [], include_dirs or [], libraries or [], ccflags or [])
        with open(so, "rb") as f:
            cache_path = cache.put(f.read(), f"{name}{suffix}", binary=True)

    print(f"cache_path: {cache_path}")
    return _load_module_from_path(name, cache_path)
```
