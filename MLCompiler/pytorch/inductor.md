# Call Stack
```Python
Traceback (most recent call last):
  File "/data00/home/son.nguyen/workspace/cpp/docs/14MLCompiler/pytorch/test_inductor.py", line 20, in <module>
    res = toy(x, y)
          ^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py", line 749, in compile_wrapper
    raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_dynamo/output_graph.py", line 1871, in _call_user_compiler
    raise BackendCompilerFailed(
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_dynamo/output_graph.py", line 1846, in _call_user_compiler
    compiled_fn = compiler_fn(gm, example_inputs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_dynamo/repro/after_dynamo.py", line 150, in __call__
    compiled_gm = compiler_fn(gm, example_inputs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/__init__.py", line 2380, in __call__
    return compile_fx(model_, inputs_, config_patches=self.config)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/compile_fx.py", line 2418, in compile_fx
    return aot_autograd(
           ^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_dynamo/backends/common.py", line 109, in __call__
    cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_functorch/aot_autograd.py", line 1199, in aot_module_simplified
    compiled_fn = AOTAutogradCache.load(
                  ^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_functorch/_aot_autograd/autograd_cache.py", line 1140, in load
    compiled_fn = dispatch_and_compile()
                  ^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_functorch/aot_autograd.py", line 1184, in dispatch_and_compile
    compiled_fn, _ = create_aot_dispatcher_function(
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_functorch/aot_autograd.py", line 576, in create_aot_dispatcher_function
    return _create_aot_dispatcher_function(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_functorch/aot_autograd.py", line 836, in _create_aot_dispatcher_function
    compiled_fn, fw_metadata = compiler_fn(
                               ^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py", line 245, in aot_dispatch_base
    compiled_fw = compiler(fw_module, updated_flat_args)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_functorch/aot_autograd.py", line 483, in __call__
    return self.compiler_fn(gm, example_inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/compile_fx.py", line 2250, in fw_compiler_base
    return inner_compile(
           ^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/compile_fx.py", line 745, in compile_fx_inner
    return wrap_compiler_debug(_compile_fx_inner, compiler_name="inductor")(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_dynamo/repro/after_aot.py", line 124, in debug_wrapper
    inner_compiled_fn = compiler_fn(gm, example_inputs)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/compile_fx.py", line 860, in _compile_fx_inner
    (key_info, cache_info) = FxGraphCache.prepare_key(
                             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/codecache.py", line 1474, in prepare_key
    key, debug_lines = compiled_fx_graph_hash(
                       ^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/codecache.py", line 960, in compiled_fx_graph_hash
    details = FxGraphHashDetails(gm, example_inputs, fx_kwargs, inputs_to_check)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/codecache.py", line 896, in __init__
    self.system_info = CacheBase.get_system()
                       ^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/codecache.py", line 205, in get_system
    from triton.compiler.compiler import triton_key
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
ImportError: cannot import name 'triton_key' from 'triton.compiler.compiler' (/data00/home/son.nguyen/workspace/triton_dev/triton/python/triton/compiler/compiler.py)

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
```

# Flow
Dynamo -> FX graph -> Inductor 
    │                      │
    │                      └─> Kernel Fusion |-> Codegen Triton Kernels
    │                      │
    │                      └─> Codegen Python Wrapper Code including Triton kernels and calls to Triton kernels 
    │                      │
    │                      └─> Save generated wrapper code to AOT dir as a module
    │                      │
    │                      └─> Load the saved module into memory
    │                      │
    │                      └─> Assign compiled_fn = loaded_module.call
    │                      │
    │                      └─> compile_fx.py returns compiled_graph = CompiledFxGraph(compiled_fn, ...)
    │                      │
    │                      └─> aot_autograd.py returns cg = SerializableCallable(compiled_graph)
    │
    └─> Serialize cg object to AOT dir


# Backend registration
MLCompiler/pytorch/torch/_dynamo/backends/inductor.py

# Debug
Using Python Debugger extension

For setting breakpoints:
```Bash
cd cpp/MLCompiler/pytorch
ln -sf /data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch .
```

# class _TorchCompileInductorWrapper
```Python
# /data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/__init__.py
```

# _operator module
In PyTorch, the module named `_operator` is not part of PyTorch itself. It is a private/internal Python standard library module that comes from CPython (the Python runtime), not from torch. PyTorch sometimes imports it (or you may see it in stack traces) because it provides low-level operator helpers used by the interpreter.

`_operator` is the private counterpart of the public `operator` module in the Python standard library.
```Python
import operator
import _operator
print(operator.__spec__)
print(_operator.__spec__)
```

# class Interpreter
torch/fx/interpreter.py

# class GraphLowering(torch.fx.Interpreter)
torch/_inductor/graph.py

# Inductor IR
MLCompiler/pytorch/torch/_inductor/ir.py

# class CUDACombinedScheduling(BaseScheduling)
MLCompiler/pytorch/torch/_inductor/codegen/cuda_combined_scheduling.py
```Python
    """
    Scheduler for CUDA Kernels, which delegates calls as appropriate
    to the CUDA-C++ and Triton Schedulers, which both work for CUDA devices
    and use a unified-wrapper for codegen.

    If Scheduling code needs to be specialized for the case of mixed Triton / CUDA C++ code,
    this would also be the place to do it.
    """
```

# class TritonScheduling(SIMDScheduling)
/data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/codegen/triton.py

# class TritonKernel(SIMDKernel[TritonCSEVariable])

# How does Inductor perform fusion?
### Overview
PyTorch Inductor performs fusion by taking an FX graph, applying graph-level transformations and scheduling, grouping compatible ops into “fusion groups,” and emitting a single kernel per group (e.g., a Triton kernel on GPU, C++/OpenMP on CPU). The goal is to reduce memory traffic (fewer reads/writes to global memory), launch overhead, and improve locality and vectorization.

### Key steps in Inductor’s fusion pipeline

1. Graph capture and normalization
   - Input: an FX graph produced by torch.compile (Dynamo).
   - Inductor canonicalizes ops into a smaller set of internal IR nodes (pointwise, reduction, memory ops, layout ops, etc.).
   - Broadcasts, shapes, and dtypes are normalized to expose fusion opportunities.

2. Dependency and alias analysis
   - Builds a dependency DAG across tensor-producing/consuming nodes.
   - Tracks reads/writes, aliasing, and in-place semantics to ensure correctness when merging nodes.

3. Partitioning into fusion groups (pattern-agnostic + pattern-specific)
   - Pattern-agnostic fusion:
     - Pointwise chains: Any sequence of elementwise ops with compatible shapes/strides is greedily fused into a single loop nest.
     - Producer-consumer fusion: A producer op (e.g., a small reshape/stride change or a broadcast) can be inlined into its consumer when no conflicting reuse is required.
     - Epilogue fusion: Sinks post-processing ops (e.g., add, bias, activation, clamp, type cast) into the “epilogue” of a kernel that produces the tensor (e.g., GEMM/conv epilogue).
   - Pattern-specific fusion:
     - Reduction + elementwise: Fuses the reduction kernel (sum/max/mean, etc.) with adjacent pointwise ops either in the prologue (pre-op) or epilogue (post-op), respecting parallelization constraints.
     - Memory-bound ops: Softmax-like patterns (scale → exp → reduce → divide) can be fused into a single kernel.
     - Contraction patterns: When using external libraries (e.g., cuBLAS/cuDNN), Inductor tries to fuse surrounding epilogues. When using Triton GEMM, it can inline more math directly into the matmul kernel.
   - Cost/legality checks ensure:
     - No shape- or stride-mismatch that would require materialization.
     - No reuse/in-place conflicts (writes-after-reads).
     - Reasonable register/shared-mem pressure and kernel launch limits.

4. Scheduling and loop/tiling decisions
   - Creates a loop IR for each fusion group with:
     - Iteration space derived from the output shapes (or reduction domains).
     - Indexing that accounts for broadcasting/strides to avoid materializing intermediates.
   - Chooses tiling, block/grid sizes (GPU) or vectorization and parallel chunks (CPU).
   - Places ops into prologue/body/epilogue positions:
     - Prologue: index computations, loads, some pre-transformations.
     - Body: core op (e.g., contraction/reduction).
     - Epilogue: chained pointwise ops, final casts, clamping, stores.
   - May split large groups if resource estimates exceed thresholds (registers, shared memory, occupancy).

5. Code generation per backend
   - GPU:
     - Emits a single Triton kernel per fusion group whenever feasible.
     - For library calls (e.g., cuBLAS matmul), emits a call plus a fused epilogue if possible; otherwise, separate kernels.
   - CPU:
     - Emits C++ with parallel-for (OpenMP or threadpool), SIMD vectorization, and inlined pointwise/reduction ops within the same loop nest.

6. Autotuning and caching
   - For Triton kernels, optionally autotunes tiling/block sizes, num warps/stages.
   - Caches compiled kernels keyed by shapes/dtypes/architecture and reuses them across runs.

### What can and cannot be fused

- Commonly fused
  - Long chains of elementwise ops: add/mul/sub/div, activation, normalization pieces, type casts.
  - Reduction + elementwise: e.g., bias + activation after a reduce; softmax pipelines.
  - Contraction epilogues: matmul/conv outputs with bias, activation, residual adds, dropout masks (where legal).

- Typically not fused (or require special handling)
  - Ops that must materialize: non-trivial reshapes that change memory layout, transposes that would cause excessive non-coalesced access, view->contiguous boundaries.
  - Control-flow or data-dependent indexing that breaks static scheduling.
  - Large library ops where only epilogue fusion is legal (e.g., cuBLAS matmul with limited custom epilogue unless using Triton matmul).
  - Resource-heavy compositions that exceed register/shared-mem budgets; Inductor splits them.

### Example: elementwise chain fusion (GPU)
- FX chain: y = relu(a * b + c).mean(dim=1) + d
- Possible fusion:
  - Kernel 1: reduction with inlined prologue a*b + c, epilogue relu, then compute mean.
  - Kernel 2: pointwise add with d, fused with any following elementwise ops.
- If resource limits allow, Inductor may fuse more ops into Kernel 1 or 2; otherwise, it splits.

### Why fusion improves performance
- Fewer kernel launches and synchronization points.
- Reduced global memory traffic: intermediates remain in registers/shared memory instead of DRAM.
- Better cache/SM utilization by tiling and coalescing loads/stores.
- Opportunity for vectorization and epilogue sinking.

### Practical notes
- Fusion results can vary by hardware, shapes, and Inductor/Triton versions.
- Debugging/inspection:
  - TORCH_LOGS=inductor,fx,output_code or TORCH_COMPILE_DEBUG=1 to dump fusion groups and generated code.
  - dynamo/inductor config flags can alter fusion heuristics (e.g., split reductions, use_triton).

In short, Inductor fuses by building loop-based fusion groups that inline compatible producers/consumers, schedules them into a single kernel with prologue/body/epilogue structure, and generates backend-specific code that keeps intermediates in registers/local memory.

# Device Backend Registration
```Python
# /data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/graph.py
# class GraphLowering(torch.fx.Interpreter)
#  -> init_backend_registration()
#    -> register_backend_for_device(...)

# /data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/codegen/common.py
# The code generated by Inductor consists of two main parts: kernel code and wrapper code.
# For any new backend looking to integrate with Inductor, customization of these two main
# parts are necessary to generate its specific code.
#
# Kernel code generation is determined by different Scheduling. Consequently, a new
# backend needs to provide a custom Scheduling for its unique kernel code generation. Currently,
# CppScheduling and TritonScheduling serve the C++/OpenMP and Triton backends, respectively.
#
# For the Wrapper, Inductor provides a PythonWrapperCodegen class to generate the Python wrapper code
# that bridges kernels. This allows out-of-tree backends to inherit from PythonWrapperCodegen,
# and override specific member functions to create backend-specific Python wrapper code.
#
# Other classes, such as CppKernel and TritonKernel, used for code generation, typically form part
# of the logic for either Scheduling or PythonWrapperCodegen. So the Scheduling and PythonWrapperCodegen interfaces
# provide flexibility to the backend. A backend can choose to implement these classes from scratch,
# or reuse them by extending and overriding as necessary. And Inductor provides the registration API,
# register_backend_for_device, to equip a new backend at runtime.
```

# Scheduler
/data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/scheduler.py
```Python
log.setLevel(logging.DEBUG)
```

# How is a Triton kernel generated?
```Python
# kernel name
# MLCompiler/pytorch/torch/_inductor/codegen/triton.py
    def define_kernel(self, src_code, node_schedule, kernel):
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )
            kernel_category = get_kernel_category_by_source_code(src_code)[:3]
            kernel_name = "_".join(
                ["triton", kernel_category, fused_name, wrapper.next_kernel_suffix()]
            )

# kernel code
# MLCompiler/pytorch/torch/_inductor/codegen/simd.py
    def codegen_node_schedule(self, kernel_features: SIMDKernelFeatures):
```