<!-- TOC START -->
- [Environment Variables](#environment-variables)
- [PTX version](#ptx-version)
- [Errors](#errors)
- [GROUP_SIZE_M](#group_size_m)
- [How Triton creates CUtensorMap objects](#how-triton-creates-cutensormap-objects)
    - [At compile phase](#at-compile-phase)
    - [At runtime](#at-runtime)
- [Autotuner](#autotuner)
  - [Enable cache](#enable-cache)
  - [self.fn.run()](#selffnrun)
    - [self.launch(...)](#selflaunch)
- [Triton Compile Cache](#triton-compile-cache)
<!-- TOC END -->







# Environment Variables
```Bash
#! Cache dir
export TRITON_CACHE_DIR=./tmp

#! Autotune
export TRITON_PRINT_AUTOTUNING="1"

#! GPU arch
#! This is a self-added env var
export TRITON_OVERRIDE_ARCH=sm86
```

# PTX version
```Python
#! /data00/home/son.nguyen/workspace/triton_dev/bytedance/triton/python/triton/backends/nvidia/compiler.py
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
```Python
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1] NameError('_triton_helper_fn_add0 is not defined')
```
There is a bug in TritonComboKernels
<br/>


# GROUP_SIZE_M
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

# Autotuner
## Enable cache
Set `cache_results=True` and do NOT set `pre_hook` for configs:
```Python
@triton.autotune(
    # configs=matmul_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    configs=matmul_get_configs(),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
    cache_results=True
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma(...):
```

Call Stack:
```Python
ipdb> where
  /data05/home/son.nguyen/workspace/triton.cpp/gemm_hopper.py(157)<module>()
    155 
    156 if __name__ == "__main__":
--> 157     main()
  /data05/home/son.nguyen/workspace/triton.cpp/gemm_hopper.py(135)main()
    134 
--> 135     c_matmul_tma = matmul_tma(a, b, warp_specialize=True)
    136     print(f"c_matmul_tma: shape={c_matmul_tma.shape} dtype={c_matmul_tma.dtype} {c_matmul_tma}")
  /data05/home/son.nguyen/workspace/triton.cpp/gemm_hopper.py(105)matmul_tma()
    104 
--> 105     matmul_kernel_tma[grid](
    106         a_desc, b_desc, c_desc,
  /data05/home/son.nguyen/workspace/triton.cpp/triton/runtime/jit.py(419)<lambda>()
    418         
--> 419         return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
    420         # return cast(T, functools.partial(cast(Callable, self.run), grid=grid))
> /data05/home/son.nguyen/workspace/triton.cpp/triton/runtime/autotuner.py(254)run()
    253         ipdb.set_trace()
--> 254         ret = self.fn.run(
    255             *args,

ipdb>

self.fn is a `JITFunction(__main__:matmul_kernel_tma)`
```

## self.fn.run()
**triton/runtime/jit.py(709)**
```Python
def run(self, *args, grid, warmup, **kwargs):
    # Retrieves kernel cache
    device = driver.active.get_current_device()
    kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
    bound_args, specialization, options = binder(*args, **kwargs)
    key = compute_cache_key(kernel_key_cache, specialization, options)
    kernel = kernel_cache.get(key, None)

    # type(kernel)
    # <class 'triton.compiler.compiler.CompiledKernel'>

    # Launch kernel
    if not warmup:
        launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
        kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                    knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())
```

kernel is a `triton.compiler.compiler.CompiledKernel` object.

**triton/compiler/compiler.py**
```Python
#! This class encapsulates a compiled kernel, including its metadata, cubin, etc.
class CompiledKernel:

self.metadata:
KernelMetadata(allowed_dot_input_precisions=['tf32', 'tf32x3', 'ieee'], arch='sm90', backend_name='cuda', cluster_dims=(1, 1, 1), debug=False, default_dot_input_precision='tf32', deprecated_fp8_dot_operand_dtypes=['fp8e4b15'], enable_fp_fusion=True, extern_libs=[['libdevice', '/data05/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/triton/backends/nvidia/lib/libdevice.10.bc']], global_scratch_align=1, global_scratch_size=0, hash='8cf8fdd416fa69f1a72fd773ecfdb1a0aabadb7f422d8f0df6ceac476222c193', instrumentation_mode='', ir_override=None, launch_cooperative_grid=False, launch_pdl=False, max_num_imprecise_acc_default=1073741824, maxnreg=None, name='matmul_kernel_tma', num_ctas=1, num_stages=4, num_warps=8, profile_scratch_align=1, profile_scratch_size=0, ptx_options=None, ptx_version=None, sanitize_overflow=True, shared=65568, supported_fp8_dtypes=['fp8e4b15', 'fp8e4nv', 'fp8e5'], target=GPUTarget(backend='cuda', arch=90, warp_size=32), tensordesc_meta=[{'swizzle': 3, 'elem_size': 2, 'elem_type': 6, 'block_size': [64, 64], 'fp4_padded': False}, {'swizzle': 3, 'elem_size': 2, 'elem_type': 6, 'block_size': [64, 64], 'fp4_padded': False}, {'swizzle': 3, 'elem_size': 4, 'elem_type': 7, 'block_size': [64, 32], 'fp4_padded': False}], tmem_size=0, triton_version='3.5.1', warp_size=32)


#
#! self.function is the GPU memory address of the kernel function
#! NOTE: driver.active.utils is defined in triton/backends/nvidia/driver.py
#! class CudaDriver(GPUDriver):

#!     def __init__(self):
#!         self.utils = CudaUtils()  # TODO: make static
#!         self.launcher_cls = CudaLauncher
#!         super().__init__()
def _init_handles(self):
    self.module, self.function, self.n_regs, self.n_spills, self.n_max_threads = driver.active.utils.load_binary(
        self.name, self.kernel, self.metadata.shared, device)

@property
def run(self):
    if self._run is None:
        self._init_handles()
    return self._run
```
When `kernel.run(...)` is called, the call is delegated to `self._run` which is a `triton.backends.nvidia.driver.CudaLauncher` object. 

**triton/compiler/compiler.py**
```Python
self._run = driver.active.launcher_cls(self.src, self.metadata)
#! launcher_cls is `triton.backends.nvidia.driver.CudaLauncher`
```

**triton/backends/nvidia/driver.py**:
```Python
class CudaLauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)
        src = make_launcher(constants, signature, tensordesc_meta)
        mod = compile_module_from_src(
            src=src,
            name="__triton_launcher",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )

        self.num_ctas = functools.reduce(operator.mul, metadata.cluster_dims, 1)
        self.launch = wrap_handle_tensordesc(mod.launch, signature, tensordesc_meta)
        self.global_scratch_size = metadata.global_scratch_size
        self.global_scratch_align = metadata.global_scratch_align
        self.profile_scratch_size = metadata.profile_scratch_size
        self.profile_scratch_align = metadata.profile_scratch_align
        self.launch_cooperative_grid = metadata.launch_cooperative_grid
        self.launch_pdl = metadata.launch_pdl

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):

        def allocate_scratch(size, align, allocator):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * self.num_ctas * size
                alloc_fn = allocator.get()
                return alloc_fn(alloc_size, align, stream)
            return None

        global_scratch = allocate_scratch(self.global_scratch_size, self.global_scratch_align, _allocation._allocator)
        profile_scratch = allocate_scratch(self.profile_scratch_size, self.profile_scratch_align,
                                           _allocation._profile_allocator)
        self.launch(gridX, gridY, gridZ, stream, function, self.launch_cooperative_grid, self.launch_pdl,
                    global_scratch, profile_scratch, *args)
```
Pay attention to the following code snippet:
```Python
        src = make_launcher(constants, signature, tensordesc_meta)
        mod = compile_module_from_src(
            src=src,
            name="__triton_launcher",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )
```
This code snippet creates a C source code and then compile it on the fly.<br/>
Then it loads the compiled module dynamically.


# __triton_launcher
## CudaLauncher
**triton/backends/nvidia/driver.py**
```Python
class CudaLauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)

        # Create the contents of __triton_launcher.c
        src = make_launcher(constants, signature, tensordesc_meta)

        # 1. Write `src` to temporary_dir/__triton_launcher.c
        # 2. Compile it to `__triton_launcher.cpython-311-x86_64-linux-gnu.so`
        # 3. Load `__triton_launcher.cpython-311-x86_64-linux-gnu.so` into a Python module
        mod = compile_module_from_src(
            src=src,
            name="__triton_launcher",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )

```
## __triton_launcher.cpython-311-x86_64-linux-gnu.so
This file name is built as follows:
```Python
import sysconfig
name = "__triton_launcher"
suffix = sysconfig.get_config_var('EXT_SUFFIX')
soname = '{name}{suffix}'.format(name=name, suffix=suffix)
print(soname)
```

## self.launch(...)
**triton/backends/nvidia/driver.py**:
```Python
self.launch = wrap_handle_tensordesc(mod.launch, signature, tensordesc_meta)
```

Where, `wrap_handle_tensordesc(...)` is defined here:
```Python
def wrap_handle_tensordesc(launcher, signature, tensordesc_meta):
    print(f"tensordesc_meta: {tensordesc_meta}")
    # ipdb.set_trace()
    has_tensor_desc_arg = any(isinstance(sig, str) and sig.startswith("tensordesc") for sig in signature.values())
    if not has_tensor_desc_arg:
        return launcher

    tensordesc_indices = set(
        [i for i, sig in enumerate(signature.values()) if isinstance(sig, str) and sig.startswith("tensordesc")])
    assert not tensordesc_meta or len(tensordesc_meta) == len(tensordesc_indices)
    if not tensordesc_meta:
        tensordesc_meta = [None] * len(tensordesc_indices)

    def inner(*args):
        final_args = list(args[:_BASE_ARGS_FORMAT_LEN])
        tensordesc_idx = 0
        for i, arg in enumerate(args[_BASE_ARGS_FORMAT_LEN:]):
            if i in tensordesc_indices:
                final_args.extend(make_tensordesc_arg(arg, tensordesc_meta[tensordesc_idx]))
                tensordesc_idx += 1
            else:
                final_args.append(arg)
        return launcher(*final_args)

    return inner


def make_tensordesc_arg(arg, metadata):
    if metadata is None:
        # Currently the host side tensor descriptors get decomposed in
        # the frontend to tensor desc, shape, and strides. We have no
        # way to use these shape and strides when processing tensor
        # descriptors which is why we provide our own decomposition
        # above. Sadly this means we have to pass the shape and strides
        # twice.
        return [arg.base, *arg.shape, *arg.strides, arg.padding == "nan", *arg.shape, *arg.strides]

    swizzle = metadata["swizzle"]
    elem_size = metadata["elem_size"]
    elem_type = metadata["elem_type"]
    block_size = metadata["block_size"]
    fp4_padded = metadata["fp4_padded"]

    shape = arg.shape
    strides = arg.strides
    assert strides[-1] == 1
    padding = 1 if arg.padding == "nan" else 0

    if fp4_padded:
        shape = list(shape)
        shape[-1] *= 2

    # ipdb.set_trace()
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

# Triton Compile Cache
Triton caches compiled kernels as follows:
**triton/compiler/compiler.py**
```Python
def compile(src, target=None, options=None, _env_vars=None):
    # create cache manager
    env_vars = get_cache_invalidating_env_vars() if _env_vars is None else _env_vars
    key = get_cache_key(src, backend, options, env_vars=env_vars)
    hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    fn_cache_manager = get_cache_manager(hash)

    ...

    file_name = src.name[:150]
    metadata_filename = f"{file_name}.json"
    metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
    # `metadata_group` is the contents of `./tmp/XJMYPKDSLEFKYDLVJVOKOESSC4PJCQYW45X7URN7EXPD53LB2LKA/__grp__matmul_kernel_tma.json`
    # where, `XJMYPKDSLEFKYDLVJVOKOESSC4PJCQYW45X7URN7EXPD53LB2LKA` is computed from `hash` above

    metadata_path = metadata_group.get(metadata_filename)
    always_compile = knobs.compilation.always_compile
    if not always_compile and metadata_path is not None:
        # cache hit!
        res = CompiledKernel(src, metadata_group, hash)
        if compilation_listener:
            compilation_listener(
                src=src,
                metadata=res.metadata._asdict(),
                metadata_group=metadata_group,
                times=timer.end(),
                cache_hit=True,
            )
        return res
```

# Triton Compiler Backends
**triton/backends/__init__.py**
```Python
backends: dict[str, Backend] = _discover_backends()
```
Print it out:
```Python
import triton
import json
print(json.dumps(triton.backends.backends, indent=2, default=str))
```
Output:
```Bash
{
  "amd": "Backend(compiler=<class 'triton.backends.amd.compiler.HIPBackend'>, driver=<class 'triton.backends.amd.driver.HIPDriver'>)",
  "nvidia": "Backend(compiler=<class 'triton.backends.nvidia.compiler.CUDABackend'>, driver=<class 'triton.backends.nvidia.driver.CudaDriver'>)"
}
```
Note that the source directory `third_party/nvidia/backend` is copied to `site-packages/triton/backends/nvidia`. <br/>

**refs/triton/compiler/compiler.py**
```Python
def compile(src, target=None, options=None, _env_vars=None):
    assert isinstance(target, GPUTarget), "target must be of GPUTarget type"
    backend = make_backend(target)
    stages = dict()
    backend.add_stages(stages, options, src.language)

    # AST to MLIR module
    try:
        module = src.make_ir(target, options, codegen_fns, module_map, context)
    except Exception as e:
        filter_traceback(e)
        raise

    # Run passes
    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)
        ir_filename = f"{file_name}.{ext}"
```
