# Table of Contents
<!-- TOC START -->
- [Call Stack](#call-stack)
- [Flow](#flow)
- [Inductor config](#inductor-config)
- [Backend registration](#backend-registration)
- [Debug](#debug)
- [class _TorchCompileInductorWrapper](#class-_torchcompileinductorwrapper)
- [_operator module](#_operator-module)
- [class Interpreter](#class-interpreter)
- [class GraphLowering(torch.fx.Interpreter)](#class-graphloweringtorchfxinterpreter)
- [Inductor IR](#inductor-ir)
- [class CUDACombinedScheduling(BaseScheduling)](#class-cudacombinedschedulingbasescheduling)
- [class TritonScheduling(SIMDScheduling)](#class-tritonschedulingsimdscheduling)
- [class TritonKernel(SIMDKernel[TritonCSEVariable])](#class-tritonkernelsimdkerneltritoncsevariable)
- [How does Inductor perform fusion?](#how-does-inductor-perform-fusion)
    - [Overview](#overview)
    - [Key steps in Inductor’s fusion pipeline](#key-steps-in-inductors-fusion-pipeline)
    - [What can and cannot be fused](#what-can-and-cannot-be-fused)
    - [Example: elementwise chain fusion (GPU)](#example-elementwise-chain-fusion-gpu)
    - [Why fusion improves performance](#why-fusion-improves-performance)
    - [Practical notes](#practical-notes)
- [Device Backend Registration](#device-backend-registration)
- [Scheduler](#scheduler)
- [How is a Triton kernel generated?](#how-is-a-triton-kernel-generated)
- [How are torch.sum and torch.sigmoid fused?](#how-are-torchsum-and-torchsigmoid-fused)
  - [scheduler.py](#schedulerpy)
    - [Where fusion regions are constructed in scheduler.py](#where-fusion-regions-are-constructed-in-schedulerpy)
    - [Entry point and iteration](#entry-point-and-iteration)
    - [One fusion round: candidates → checks → merges](#one-fusion-round-candidates-checks-merges)
      - [1) Candidate discovery and ordering](#1-candidate-discovery-and-ordering)
      - [2) Legality and direction checks](#2-legality-and-direction-checks)
      - [3) Profitability: benchmarkable speedup or heuristic score](#3-profitability-benchmarkable-speedup-or-heuristic-score)
      - [4) Merge: produce a fused region node](#4-merge-produce-a-fused-region-node)
    - [Supporting structures that define the fusion region’s semantics](#supporting-structures-that-define-the-fusion-regions-semantics)
    - [Summary of fusion region construction](#summary-of-fusion-region-construction)
  - [ir.Buffer](#irbuffer)
    - [What is ir.Buffer?](#what-is-irbuffer)
    - [Relationship to other IR and outputs](#relationship-to-other-ir-and-outputs)
    - [How schedulers and fusion use Buffer](#how-schedulers-and-fusion-use-buffer)
    - [Typical lifecycle](#typical-lifecycle)
    - [What “storage” means here](#what-storage-means-here)
    - [Variants and special cases](#variants-and-special-cases)
  - [/data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/pattern_matcher.py](#data00homesonnguyenworkspacecppmlcompilerpytorchtorch_inductorpattern_matcherpy)
  - [/data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/lowering.py](#data00homesonnguyenworkspacecppmlcompilerpytorchtorch_inductorloweringpy)
  - [What is the PyTorch dispatcher?](#what-is-the-pytorch-dispatcher)
    - [Why a dispatcher?](#why-a-dispatcher)
    - [How it works (conceptually)](#how-it-works-conceptually)
    - [Example flow](#example-flow)
    - [Key pieces of the ecosystem](#key-pieces-of-the-ecosystem)
    - [When you interact with it](#when-you-interact-with-it)
    - [Benefits](#benefits)
  - [log_ir_pre_fusion and log_ir_post_fusion](#log_ir_pre_fusion-and-log_ir_post_fusion)
  - [Custom Pass](#custom-pass)
  - [How does Inductor create the ComputedBuffer?](#how-does-inductor-create-the-computedbuffer)
    - [Step-by-Step Creation Process](#step-by-step-creation-process)
- [How are symbolic variables created?](#how-are-symbolic-variables-created)
- [FX Graph Lowering Process](#fx-graph-lowering-process)
  - [Step 1: Run FX graph to lower aten ops to Inductor IR](#step-1-run-fx-graph-to-lower-aten-ops-to-inductor-ir)
    - [How is torch.ops.aten.add.Tensor lowered?](#how-is-torchopsatenaddtensor-lowered)
  - [Step 2: Fuse ir.Buffer nodes](#step-2-fuse-irbuffer-nodes)
    - [How does Scheduler fuse two SchedulerNode nodes?](#how-does-scheduler-fuse-two-schedulernode-nodes)
      - [Understand the following IR:](#understand-the-following-ir)
      - [Completely understand `class op1461_loop_body` and `def body(self, ops)`](#completely-understand-class-op1461_loop_body-and-def-bodyself-ops)
  - [Step 3: Codegen SchedulerNode nodes](#step-3-codegen-schedulernode-nodes)
    - [How is the following IR lowered to a Triton kernel?](#how-is-the-following-ir-lowered-to-a-triton-kernel)
- [Triton Kernel](#triton-kernel)
- [Codegen Combo Kernels](#codegen-combo-kernels)
- [Triton Config](#triton-config)
- [Get compiled module path](#get-compiled-module-path)
- [Debug Compiled Module](#debug-compiled-module)
<!-- TOC END -->






<br/><br/>
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
    │                      └─> Lower aten ops to Inductor IR ops
    │                      │
    │                      └─> Kernel Fusion -> Codegen Triton Kernels
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
    │
    └─> convert_frame.py: _compile(...) returns guarded_code
    │
    └─> convert_frame.py: class CatchErrorsWrapper returns the guarded_code to the caller in `torch/csrc/dynamo/eval_frame.c`.
    │
    └─> The caller evaluates the guarded code


Summary: FX graph --[lowering]--> Inductor IR (Buffers)--> Scheduler --[fusing]--> FusedSchedulerNode nodes --> CUDACombinedScheduling --[codegening]--> Triton kernels

1. Understand lowering
2. Understand fusing
3. Understand codegening


# Inductor config
/data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/config.py

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

def register_backend_for_device(
    device: str,
    device_scheduling: SchedulingConstructor,
    device_wrapper_codegen: WrapperConstructor,
    device_cpp_wrapper_codegen: Optional[WrapperConstructor] = None,
) -> None:
    device_codegens[device] = DeviceCodegen(
        device_scheduling, device_wrapper_codegen, device_cpp_wrapper_codegen
    )

# For the case device = 'cuda':
# CUDACombinedScheduling combines Triton and CUDA C++ scheduling for CUDA devices via delegation
cuda_backends = {
    "triton": CUDACombinedScheduling,
    "halide": HalideScheduling,
}
register_backend_for_device(
    "cuda",
    lambda scheduling: cuda_backends[config.cuda_backend](scheduling),
    PythonWrapperCodegen,
    CppWrapperGpu,
)
# Where, 
# - config.cuda_backend = 'triton'

# - scheduling is a Scheduler object
# At codegen phase, device_scheduling will be constructed as follows:
# MLCompiler/pytorch/torch/_inductor/scheduler.py
class Scheduler:
    def create_backend(self, device: torch.device) -> BaseScheduling:
        assert not is_gpu(device.type) or device.index is not None, (
            f"{device} should have been normalized in lowering"
        )
        V.graph.add_device_info(device)

        device_scheduling = get_scheduling_for_device(device.type)
        if device_scheduling is None:
            raise RuntimeError(f"Unsupported device type: {device.type}")

        if not has_triton():
            if (
                device.type == "cuda"
                and (device_props := torch.cuda.get_device_properties(device)).major < 7
            ):
                raise GPUTooOldForTriton(device_props, inspect.currentframe())
            elif is_gpu(device.type) and not device.type == "mps":
                raise TritonMissing(inspect.currentframe())

        return device_scheduling(self)

    def get_backend(self, device: Optional[torch.device]) -> BaseScheduling:
        assert device is not None
        if device not in self.backends:
            self.backends[device] = self.create_backend(device)
        return self.backends[device]

    def _codegen(self, nodes: list[BaseSchedulerNode]) -> None:
            if node.is_template():
                prologue, template_node, epilogue = node.get_prologue_template_epilogue(
                    list(node.get_nodes())
                )
                self.get_backend(device).codegen_template(
                    template_node, epilogue, prologue
                )
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

# How are torch.sum and torch.sigmoid fused?
```Python
class ToyModule(nn.Module):
  def __init__(self):
    super().__init__()
    self.forward = torch.compile(self.forward, backend="inductor")

  def forward(self, x, y):
    z = x + y
    d0 = z.shape[0]
    d1 = z.shape[1]
    u = torch.reshape(z, (d1, d0))
    h = torch.matmul(x, u)
    logit = torch.sum(h, 1)
    res = torch.sigmoid(logit)
    return res
```

**SchedulerNode:** wrapper around IR nodes used by the scheduler to track:
- iteration spaces (loop nests, ranges)
- read/write sets (dependencies)
- device, dtype, layout (contiguity/strides)
- grouping state (current fusion group ID)

**Fusion groups:**
- A group holds a set of `SchedulerNode`s that will be emitted as a single kernel.
- Group metadata aggregates domains, tiling/parallelism plans, and in/out buffers.

## scheduler.py
/data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/scheduler.py
### Where fusion regions are constructed in scheduler.py

Inductor’s fusion regions are built in the Scheduler by repeatedly identifying legal/beneficial pairs of nodes and merging them into a synthetic “fusion node” that represents a single future kernel. The code path for constructing these regions is centered around:

- Scheduler.fuse_nodes() → Scheduler.fuse_nodes_once()
- Candidate discovery: Scheduler.get_possible_fusions()
- Legality checks and direction: Scheduler.can_fuse(), Scheduler.can_fuse_vertical(), fusable_read_and_write(), fusable_weak_dep(), will_fusion_create_cycle()
- Profitability ordering: get_possible_fusions_with_highest_priority(), score_fusion_key(), speedup_by_fusion()
- Merge operation: BaseScheduling.fuse() → FusedSchedulerNode.fuse(), refresh_group_node_dependencies()

Below is a focused walkthrough of the key code responsible for constructing fusion groups.

---

### Entry point and iteration

- schedule initialization calls:
  - self.compute_dependencies()
  - topological sorts and DCE
  - self.create_foreach_nodes()
  - self.fuse_nodes(self.nodes)  ← starts fusion region construction

- Scheduler.fuse_nodes(self, nodes): repeatedly calls fuse_nodes_once up to 10 rounds until no change or single node.

```python
def fuse_nodes(self, nodes):
    for i in range(10):
        nodes = self.fuse_nodes_once(nodes)
        if len(nodes) didn't change or is 1: break
    return nodes
```

This greedy, multi-pass loop allows newly formed fused nodes to unlock more fusion opportunities.

---

### One fusion round: candidates → checks → merges

#### 1) Candidate discovery and ordering

- Scheduler.get_possible_fusions(nodes)
  - Skips unfusable nodes (unfusable_node)
  - Groups nodes by shared buffer usage (used_buffer_names); optionally by group signature if aggressive_fusion
  - For each grouping, checks all pairs, running can_fuse(node1, node2) to filter potential fusions
    - Special case: foreach/template fusions can be order-dependent; if can_fuse(node2, node1), it pushes (node2, node1)
  - Prunes to pairs with highest backend priority via get_possible_fusions_with_highest_priority
  - Sorts by profitability score key (score_fusion_key → V.choices.score_fusion)
  - Returns a list of ordered candidate pairs

```python
def get_possible_fusions(self, nodes):
    # build groupings by shared buffers
    check_all_pairs(group)
    # optionally group by loop group signature
    # filter by can_fuse
    pairs = self.get_possible_fusions_with_highest_priority(pairs)
    pairs.sort(key=self.score_fusion_key, reverse=True)
    return pairs
```

#### 2) Legality and direction checks

- Scheduler.can_fuse(node1, node2)
  - Fast outs:
    - Same node, GroupedSchedulerNode involvement, extern/nop unless template, device mismatch
    - Respect no-fuse buffer list and multi-output template special-case
  - Template prologue/epilogue specific gating
  - Compute shared_data_score = score_fusion_memory(node1, node2)
    - If too small and loop_ordering_after_fusion is enabled, try shared_data_after_reordering_loop to reorder loops of one node to expose more matching memory access (and re-evaluate)
  - Ask global choice hooks V.choices.can_fuse(...)
  - Directional:
    - If node2 depends on node1 (node2.ancestors includes node1 output ops): vertical fusion path
      - return can_fuse_vertical(node1, node2) AND V.choices.can_fuse_vertical(...) AND backend.can_fuse_vertical(node1, node2)
    - Else: horizontal fusion path
      - return V.choices.can_fuse_horizontal(...) AND backend.can_fuse_horizontal(node1, node2)

Key subroutines used for vertical checks:
- can_fuse_vertical(node1, node2)
  - Start from node2.unmet_dependencies; drop WeakDeps that are “fusable” (fusable_weak_dep) against node1; for each MemoryDep written by node1, erase matching reads in node2 using fusable_read_and_write
  - If any remaining dep references a buffer that node1 also writes, it’s a conflict (e.g., mismatched indices, StarDep vs MemoryDep)
  - If remaining deps originate from ops that must be scheduled between node1 and node2, refuse (intermediate nodes)
  - Otherwise legal

- fusable_read_and_write(read, write: MemoryDep)
  - For MemoryDep vs MemoryDep:
    - Names match after mutation renames
    - Not a TMP index
    - If loop_ordering_after_fusion and different rank, normalize (merge loops) before comparing
    - Equal index, and read.size is broadcast-compatible with write.size (prefix match)
  - For StarDep vs MemoryDep:
    - Same mode (e.g., inplace mode) and renamed names match
- fusable_weak_dep(weak_dep, node1, node2)
  - Only if node1 reads the real mutating buffer at exactly the same index/size that node2 writes (true inplace), no TMPs

Cross-graph safety:
- will_fusion_create_cycle(node1, node2)
  - DFS over existing fused nodes’ ancestor sets to ensure merging doesn’t create a new path causing a cycle with respect to original ancestor relationships

Memory-safety heuristic to avoid pathological fusions:
- can_fusion_increase_peak_memory(node1, node2)
  - Estimates if losing reuse on single-user inputs could outweigh bandwidth savings (score_fusion_memory). If so, may decline (used elsewhere in V.choices gating).

Loop-ordering assist:
- shared_data_after_reordering_loop(node1, node2)
  - If loop_ordering_after_fusion enabled and on GPU, tries to reorder loops of one participant to match memory layout/access order (by comparing MemoryDep stride-order signatures for common buffers, picking largest common buffer). If successful, increases the shared_data_score used by can_fuse.

#### 3) Profitability: benchmarkable speedup or heuristic score

- After a candidate passes can_fuse and no cycle:
  - speedup = self.speedup_by_fusion(node1, node2)
    - If config.benchmark_fusion is off (and not a multi-template case), returns True
    - Else may return a callable that asynchronously compiles candidate kernels and benchmarks when needed
  - fuse_nodes_once defers fusions with pending async benchmark, records them in pending_fusions
  - Otherwise, if profitable (or small kernels, etc.), proceed to merge

#### 4) Merge: produce a fused region node

- The actual merge happens via backend.fuse(node1, node2) → default BaseScheduling.fuse delegates to:
  - FusedSchedulerNode.fuse(node1, node2) for regular fusion
  - ForeachKernelSchedulerNode.fuse for foreach cases

FusedSchedulerNode.fuse:
- Flattens both inputs’ underlying nodes via get_nodes() and constructs a new FusedSchedulerNode(snodes = node1.nodes + node2.nodes)
- Special handling: template + MultiOutput epilogues rewrite StarDep to MemoryDep for scoring
- init_group_node sets:
  - self.snodes, scheduler, node=None, ancestors = union of snodes’ ancestors
  - read_writes = union of snodes’ read_writes; unmet_dependencies = union minus writes of the fused set, pruned if produced internally
  - min_order/max_order from children
  - outputs_by_name aggregation
- FusedSchedulerNode.group is selected from the reduction node if any, otherwise node1.group

After fusing:
- fuse_nodes_once updates fused_nodes set and name_to_fused_node mapping for all constituent node names to point to the new fused node
- At the end of the round:
  - Sort by min_order, then re-toposort to ensure consistent dependency ordering
  - prune_redundant_deps removes WeakDeps made redundant by fusion (see _prune_redundant_deps)

Pending fusions resolution:
- If a new pair involves a node already in pending_fusions, fuse_nodes_once resolves the pending entries first. For each pending pair:
  - Run the stored is_speedup() (which also waits for compilation)
  - If profitable and no cycle now, fuse them immediately
- After iterating candidates, it drains remaining pending pairs similarly

---

### Supporting structures that define the fusion region’s semantics

- FusedSchedulerNode
  - Represents one fusion region; get_nodes() returns flattened constituent nodes
  - read_writes/unmet_dependencies maintained as unions; enables legality checks relative to the rest of the graph
  - get_outputs() concatenates all child outputs; used to track buffer lifetimes and users
  - has aliasing/mutation aggregation; is_reduction/is_template are unioned

- refresh_group_node_dependencies(group_snode)
  - Recomputes read_writes and unmet_dependencies from snodes after loop reorderings or other updates

- Loop-ordering helpers inside SchedulerNode and FusedSchedulerNode
  - reorder_loops_by_dep_pair influences extracted read_writes and loop sizes
  - merge_loops invoked after fusion rounds (merge_loops pass), and refresh_dependencies keeps fake deps

---

### Summary of fusion region construction

- Fusion regions are FusedSchedulerNode objects created by merging BaseSchedulerNode pairs.
- The core is in:
  - Candidate enumeration and ordering: get_possible_fusions(), get_possible_fusions_with_highest_priority(), score_fusion_key()
  - Legality: can_fuse() → vertical vs horizontal; can_fuse_vertical(); fusable_read_and_write(); fusable_weak_dep(); will_fusion_create_cycle()
  - Profitability: speedup_by_fusion(), with optional async compile/benchmark gating
  - Merge: BaseScheduling.fuse() → FusedSchedulerNode.fuse(); update mappings and dependencies; prune redundant deps
- Loop reordering (shared_data_after_reordering_loop) can increase shareable memory to unlock otherwise marginal fusions.
- The result is a list of nodes where many are FusedSchedulerNode, each representing a fusion region to be codegenerated as a single kernel by the backend.

## ir.Buffer
/data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/ir.py
### What is ir.Buffer?

In Inductor’s IR (file ir.py), Buffer is the core IR node representing a concrete tensor-like storage with a layout. Conceptually:

- It is an IRNode that owns or references storage for a single tensor output.
- It carries a Layout (strides, size, dtype, device, offset), or a more general OutputSpec that may represent special cases (e.g., multi-output, mutation placeholders).
- It is the unit that low-level codegen indexes into (via make_indexer) and loads/stores from (via ops.load/ops.store).

Key points from the class definition:

- Class: Buffer(IRNode)
  - Fields:
    - name: Optional[str]
    - layout: OutputSpec (commonly a Layout subtype like FlexibleLayout or FixedLayout; but could also be MultiOutputLayout, NoneLayout, etc.)
  - Responsibilities:
    - Identity: get_name()
    - Shape/Striding: get_size(), get_stride(), get_offset()
    - Device/Dtype: get_device(), dtype property
    - Indexing and loading: make_indexer() returns a function mapping logical indices to a linear address; make_loader() produces a callable that loads from this buffer at a given index expression
    - Layout control: freeze_layout() and “freeze_with_*” helpers fix FlexibleLayout to a specific FixedLayout or stride ordering
    - Alias/mutation metadata: get_inputs_that_alias_output() and get_mutation_names()
    - Allocation policy: should_allocate() by default False (override in subclasses that represent actual storage-producing ops)
    - Realization: realize() is a no-op (subclasses override if needed)
    - OutputSpec access: get_layout() for Layout-specific behaviors; get_output_spec() for the general case

Simplified snippet (adapted):

```python
@ir_dataclass(frozen=False)
class Buffer(IRNode):
    name: Optional[str]
    layout: OutputSpec

    def get_name(self) -> str: ...
    def get_device(self) -> Optional[torch.device]: return self.get_output_spec().get_device()
    @property
    def dtype(self) -> torch.dtype: return self.get_layout().dtype

    def get_size(self) -> Sequence[Expr]: return [*self.get_layout().size]
    def get_stride(self) -> list[Expr]: return [*self.get_layout().stride]
    def get_offset(self) -> Expr: return self.get_layout().offset

    def get_layout(self) -> Layout:
        if isinstance(self.layout, Layout):
            return self.layout
        raise NotImplementedError(type(self.layout).__name__)

    def get_output_spec(self) -> OutputSpec: return self.layout

    def freeze_layout(self) -> None:
        if isinstance(self.layout, Layout) and not isinstance(self.layout, NonOwningLayout):
            self.layout = self.layout.as_fixed()

    # Variants to fix layout form
    def freeze_layout_with_stride_order(self, order, allow_padding=False): ...
    def freeze_layout_with_fill_order(self, order): ...
    def freeze_layout_with_same_order(self, stride): ...
    def freeze_layout_with_exact_strides(self, exact_strides, allow_padding=False): ...

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        if isinstance(self.layout, NonOwningLayout):
            return [self.layout.view.get_name()]
        return ()

    def get_mutation_names(self) -> Sequence[str]:
        if isinstance(self.layout, MutationLayoutSHOULDREMOVE):
            return [self.layout.target.get_name()]
        return ()

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        return self.get_layout().make_indexer()

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        if self.is_zero_elements():
            return partial(nop_loader_fn, dtype=self.get_dtype())
        def loader(index):
            indexer = self.make_indexer()
            return ops.load(self.name or "unnamed", indexer(index))
        return loader

    def should_allocate(self) -> bool:
        return False
```

### Relationship to other IR and outputs

- Buffer is the base “storage-like” output type. Many IR nodes ultimately produce Buffers:
  - OperationBuffer(Buffer, Operation): A node that is both an operation and a single-output buffer (e.g., ComputedBuffer, ExternKernelOut). Its get_outputs() returns [self].
  - ComputedBuffer(OperationBuffer): A Buffer produced by pointwise/reduction/scan/sort “Loops” computation; it extracts read-writes and participates in fusion.
  - TemplateBuffer(OperationBuffer): Represents a template (e.g., Triton) kernel output; may support epilogue fusion.
  - ExternKernelOut(ExternKernel): An extern op that produces a buffer (out-variant).
  - MutationOutput(Buffer): A special buffer representing mutation of an existing buffer (used to model side-effects).
  - MultiOutput is not a Buffer; it’s an ExternKernel that indexes into a MultiOutputLayout result to materialize per-output Buffers.

- OutputSpec vs Layout:
  - Buffer.layout is an OutputSpec so it can be:
    - Layout (FixedLayout/FlexibleLayout/NonOwningLayout/CommBufferLayout/MutationLayoutSHOULDREMOVE)
    - MultiOutputLayout (multi-result carrier; associated with MultiOutput nodes)
    - NoneLayout (non-tensor outputs or special extern/mutation nodes)
  - get_layout() only works if layout is a concrete Layout; otherwise callers use get_output_spec()

### How schedulers and fusion use Buffer

- Schedulers track Buffers to:
  - Build dependencies (reads/writes) via buffer names (buf.get_name()).
  - Track aliasing and mutations with get_inputs_that_alias_output()/get_mutation_names(), which affect fusion legality.
  - Compute sizes/strides/devices/dtypes to ensure compatible fusion regions.
  - Generate code: Buffers supply indexers and loaders used by codegen.

### Typical lifecycle

- During lowering, high-level ops become IR nodes; when realized, they produce Buffers (often via StorageBox.realize -> ComputedBuffer).
- Buffers are named and registered in the graph (V.graph.register_buffer), allowing dependency linking by name.
- During fusion scheduling, operations that write Buffers may be fused with consumers if dependencies and iteration/indexing are compatible.
- At codegen, Buffers are allocated if should_allocate() and are read/written via their indexers.

In short, ir.Buffer is the foundational data container in Inductor IR representing a tensor’s storage semantics (shape/strides/device/dtype) and providing the primitives that both scheduling (fusion legality) and code generation (indexing and memory ops) rely on.

Yes, but more precisely:

- ir.Buffer is the Inductor IR node that represents a concrete tensor-like buffer with a layout (size/stride/dtype/device/offset). It is the storage abstraction that codegen indexes and reads/writes.
- It can own storage or refer to storage (via special layouts). It is not the runtime tensor itself, but the compiler IR object describing how to access the memory that backs a tensor result.

### What “storage” means here
- Shape/striding: Buffer carries a Layout (FixedLayout/FlexibleLayout/NonOwningLayout/etc.) that defines size, stride, dtype, device, and offset.
- Indexing: Buffer.make_indexer() and make_loader() give codegen the expressions to compute addresses and load values.
- Allocation: Some buffers should_allocate() and get materialized in the wrapper; others are views or produced by ops and may not require separate allocation.

### Variants and special cases
- OperationBuffer: A Buffer that is also an IR operation (single-output op); e.g., ComputedBuffer (from pointwise/reduction IR) and TemplateBuffer (Triton template output).
- NonOwningLayout: The Buffer is a view into someone else’s storage (aliasing).
- MutationLayoutSHOULDREMOVE / MutationOutput: Used to model mutations (in-place effects) rather than independent storage.
- MultiOutputLayout: Not a Layout of a single tensor; it’s a multi-result carrier. A MultiOutput node indexes out actual Buffer(s) from it.
- NoneLayout: Placeholder for non-tensor/special nodes; these don’t represent real tensor storage.

So: ir.Buffer is the compiler’s representation of “a tensor’s backing storage and how to index it,” which scheduling and codegen operate on.


## /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/pattern_matcher.py
fx_to_pattern: Convert an FX graph into a PatternExpr. This is useful for simple patterns that can only match single functions and fixed-length lists.

## /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/lowering.py
```Python
lowerings: dict[Union[Callable[..., Any], str], Callable[..., Any]] = {}
```

## What is the PyTorch dispatcher?

The dispatcher is PyTorch’s central runtime routing system that decides which concrete kernel implementation to run when you call an operator (e.g., `aten::add`). Think of it as a multi-backend, multi-dtype, multi-layout function router with extensibility hooks.

When you call:
- `torch.add(x, y)`, or
- `torch.ops.aten.add.Tensor(x, y)`

the call goes through the dispatcher, which:
1) Identifies the operator schema (name, overload, argument types).
2) Applies the dispatcher’s execution pipeline (a stack of “keys”/dispatch domains).
3) Selects the right kernel implementation based on the inputs’ properties (device, dtype, layout, autograd needs, etc.).
4) Invokes that kernel (possibly after running higher/lower-priority wrappers like Autograd).

### Why a dispatcher?

- Multiple backends: CPU, CUDA, MPS, XPU, Vulkan, Meta, etc.
- Multiple dtypes/layouts: float16/bfloat16/quantized; strided, channels-last; sparse/CSR/COO.
- Cross-cutting concerns: Autograd, functionalization, aliasing/meta kernels, dynamic shapes, tracing.
- Extensibility: custom C++/Python extensions can register kernels for new or existing ops/devices.

### How it works (conceptually)

- Operators are registered with schemas and one or more kernels via `TORCH_LIBRARY` and friends.
- Kernels are tagged by “dispatch keys” (e.g., CPU, CUDA, Sparse, Autograd, Functionalize).
- At call time, the dispatcher builds an ordered “dispatch key set” from the inputs and thread-local state, then picks the highest-priority matching kernel in a two-level scheme:
  - Wrapper keys (e.g., Autograd, Functionalize) can intercept and then redispatch.
  - Backend keys (e.g., CUDA, CPU, SparseCUDA) provide the actual numeric kernel.
- If no kernel matches, you get a runtime error.

### Example flow

Calling `torch.ops.aten.add.Tensor(x, y)` where:
- x, y are CUDA float32, strided tensors
- grad is enabled

Rough path:
- Schema: `aten::add.Tensor(Tensor, Tensor, Scalar alpha=1) -> Tensor`
- Key set derives from inputs and TLS: Autograd, CUDA, (Strided)
- The dispatcher:
  1) Hits Autograd wrapper if gradients are required.
  2) Then redispatches to CUDA backend kernel for add on strided tensors.
  3) Executes the CUDA kernel; Autograd records history.

### Key pieces of the ecosystem

- Operator schema: name, overload, args, returns.
- Dispatch keys: identify concerns/backends (CPU, CUDA, MPS, Sparse, Quantized, Autograd, Functionalize, Python, Meta, etc.).
- Kernel registry: mapping (op, dispatch key) -> function pointer.
- Redispatch: wrapper kernels call back into the dispatcher to continue down the stack.
- Namespaces: `aten` for core ops; others for prims/backends/custom extensions.

### When you interact with it

- `torch.ops.aten.*` calls go straight into the dispatcher (bypassing higher-level Python sugar).
- Custom ops register schemas and kernels; the dispatcher routes to them like built-ins.
- Tools (FX/AOTAutograd/Inductor) reason in terms of dispatcher ops because they are precise and backend-agnostic.

### Benefits

- Unified, extensible operator calling convention.
- Late binding to the most appropriate kernel given runtime types/devices.
- Clean layering for autograd, functionalization, and transformations.
- Enables backend portability and custom kernel injection without changing user code.

## log_ir_pre_fusion and log_ir_post_fusion
/data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/scheduler.py

## Custom Pass
/data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/scheduler.py
```Python
        if config._pre_fusion_custom_pass is not None:
            self.nodes = config._pre_fusion_custom_pass(self.nodes)
```

## How does Inductor create the ComputedBuffer?
Inductor creates a **`ComputedBuffer`** when it lowers a node from the PyTorch FX graph that represents a computation. It does this through a process called **lowering**, which translates high-level PyTorch operators into Inductor's internal, low-level IR. The `ComputedBuffer` is created to represent the output of an operation or a fused sequence of operations. It encapsulates the computational logic needed to produce the final tensor, as opposed to storing the actual tensor data itself. This is a key design choice that enables Inductor's fusion optimizations.

***

### Step-by-Step Creation Process

The creation of a `ComputedBuffer` is part of Inductor's overall lowering pass. It happens in these steps:

1.  **Iterating the FX Graph:** Inductor traverses the PyTorch FX graph, which consists of nodes representing individual operations (e.g., `aten.add`, `aten.relu`).
2.  **Lowering Functions:** For each node, Inductor dispatches to a specific lowering function. There's a lowering function for every supported PyTorch operator. This function is responsible for translating the PyTorch operation into one or more Inductor IR objects.
3.  **Operator-Specific Logic:** The lowering function for a given operator, say `torch.ops.aten.add`, is where the `ComputedBuffer` is created. For simple, element-wise operations, the lowering function returns a **`Pointwise`** object, which is a specialized type of `ComputedBuffer`. This `Pointwise` object contains a **`lambda` function** (the `inner_fn`) that describes the computation for a single element. For example, for an addition, the `inner_fn` would be `lambda index: ops.add(ops.load(x, index), ops.load(y, index))`. This lambda function is the core of the define-by-run IR.
4.  **Tracking Dependencies:** Inductor keeps track of which `ComputedBuffer`s are inputs and outputs for each operation. When it encounters a chain of element-wise operations (like `add` followed by `relu`), it doesn't immediately "realize" or materialize the intermediate `ComputedBuffer` for the addition. Instead, it passes this intermediate `ComputedBuffer` to the lowering function for `relu`.
5.  **Kernel Fusion:** The lowering function for `relu` sees that its input is another `ComputedBuffer` (the `add` operation). This is the key moment for fusion. Instead of creating a new `ComputedBuffer` that loads the output of the previous one, it **merges the two `inner_fn`s**. It creates a new `ComputedBuffer` whose `inner_fn` now contains both operations: `lambda index: ops.relu(ops.add(ops.load(x, index), ops.load(y, index)))`. The intermediate `ComputedBuffer` for the addition is effectively "fused away."

By creating a `ComputedBuffer` with a combined `inner_fn`, Inductor describes a single, fused kernel that can be efficiently generated by backends like Triton. The final result is a simplified IR graph with fewer nodes and no intermediate memory allocations for fused operations.


# How are symbolic variables created?
```Python
# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/fx/experimental/symbolic_shapes.py
@record_shapeenv_event()
def create_symbol(...):
    if val in (0, 1) and specialize_zero_one:
        r = self.val_to_var[val]
    elif not duck or val not in self.val_to_var:
        # If we're not duck shaping, we always create a new symbol
        # Even if we're duck shaping, if we haven't seen this particular
        # value before, we also create a new symbol
        if type(val) is int or is_nested_int(val):
            sympy_expr = make_symbol(
                SymT.SIZE, len(self.var_to_val), positive=positive, integer=True
            )
        else:
            sympy_expr = make_symbol(
                SymT.FLOAT, len(self.var_to_val), positive=positive, real=True
            )
        self.source_to_var[source_name] = sympy_expr
        # We always associate vars to vals
        if isinstance(val, int):
            self.var_to_val[sympy_expr] = sympy.Integer(val)
        elif isinstance(val, float):
            self.var_to_val[sympy_expr] = sympy.Float(val)
        else:
            # Only used for jagged layout nested tensors
            self.var_to_val[sympy_expr] = SingletonInt(
                val.node.nested_int(), coeff=val.node.nested_int_coeff()
            )
```



# FX Graph Lowering Process
## Step 1: Run FX graph to lower aten ops to Inductor IR
```Python
# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/compile_fx.py
with V.set_graph_handler(graph):
    graph.run(*example_inputs)

graph():
  %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
  %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
  %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
  %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add, [1]), kwargs = {})
  %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%sum_1,), kwargs = {})
  return (sigmoid,)
```

### How is torch.ops.aten.add.Tensor lowered?
V0828 04:36:42.229000 3516361 site-packages/torch/_inductor/graph.py:1455] [0/0] lowering %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {}) 
V0828 04:36:42.229000 3516361 site-packages/torch/_inductor/graph.py:1181] [0/0]   via <function make_pointwise.<locals>.inner at 0x7fdf6a70b100>
```Python
# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/graph.py:1189
lowered_fn = lowerings[target]
repr(target)
"<OpOverload(op='aten.add', overload='Tensor')>"

out = lowered_fn(*args, **kwargs)  # type: ignore[index]
repr(lowered_fn)
'<function make_pointwise.<locals>.inner at 0x7fbec91b8c20>'

# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/lowering.py:6353
add = register_pointwise(
    aten.add, allow_alpha=True, override_fn_when_input_bool="logical_or"
)

# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/lowering.py:843
def register_pointwise(
    aten_fn, # "<OpOverloadPacket(op='aten.add')>"
    name=None,
    broadcast=True,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    convert_input_to_bool=False,
    override_return_dtype=None,
    override_fn_when_input_bool=None,
    allow_alpha=False,
    use_libdevice_for_f64=False,
    triton_fallback=None,
):
    """A pointwise function that maps ops.{name} to inputs"""
    name = name or aten_fn.__name__
    fn = ops_wrapper(name)
    if use_libdevice_for_f64:
        fn_libdevice = ops_wrapper("libdevice_" + name)
        register_op_dtype_propagation_rules(
            "libdevice_" + name, type_promotion_kind, override_return_dtype
        )

    register_op_dtype_propagation_rules(
        name, type_promotion_kind, override_return_dtype
    )

    if override_fn_when_input_bool is not None:
        override_fn_when_input_bool = ops_wrapper(override_fn_when_input_bool)

    fn = make_pointwise(
        fn,
        override_return_dtype=override_return_dtype,
        override_fn_when_input_bool=override_fn_when_input_bool,
        override_fn_when_gpu_float64=fn_libdevice if use_libdevice_for_f64 else None,  # type: ignore[possibly-undefined]
        allow_alpha=allow_alpha,
        triton_fallback=triton_fallback,
    )
    fn = register_lowering(
        aten_fn,
        broadcast=broadcast,
        type_promotion_kind=type_promotion_kind,
        convert_input_to_bool=convert_input_to_bool,
    )(fn)

    if hasattr(prims, name):
        register_lowering(
            getattr(prims, name),
            type_promotion_kind=None,
            convert_input_to_bool=convert_input_to_bool,
        )(fn)
    return fn


# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/ir.py
from .virtualized import ops, OpsValue, V

def ops_wrapper(name: str) -> Callable[..., OpsValue]:
    assert isinstance(name, str)

    def fn(*args: object, **kwargs: object) -> OpsValue:
        return getattr(ops, name)(*args, **kwargs)

    return fn


# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/virtualized.py
ops: OpsHandler[Any] = OpsWrapper()

class OpsWrapper(DefaultHandler):
    """This wraps any returned IR values into an `OpsValue` instance, so that we
    can overload the magic methods for writing mathematical expressions fluently.
    """

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        new_args = [OpsWrapper._unwrap(a) for a in args]
        new_kwargs = {k: OpsWrapper._unwrap(v) for k, v in kwargs.items()}
        return OpsWrapper._wrap(getattr(_ops, name)(*new_args, **new_kwargs))

class DefaultHandler(OpsHandler[Any]):
    def __getattr__(self, name: str) -> Any:
        def fallback(*args: Any, **kwargs: Any) -> Any:
            return self._default(name, args, kwargs)

        # would like to remove this function entirely, but it's used in MTIA backend
        warnings.warn(f"undefined OpHandler.{name}, please add missing op schema")
        return fallback

# Register ops
DefaultHandler._init_cls()

# fn = ops_wrapper(name)
# Flow: fn -> make_pointwise -> wrapper1(fn) -> register_lowering -> wrapper2(wrapper1(fn))

#================================================================================================================================================================================
# 1. FX Node Lowering Call Stack
#================================================================================================================================================================================

# FX Node --forwards--> target=aten_fn --forwards--> lowerings[target] --maps--> _register_lowering.<locals>.wrapped --calls--> make_pointwise.<locals>.inner --> does {
#   - define inner_fn --wraps--> ops_wrapper(aten_fn.__name__)
#   - create a Pointwise (which is a Loops(IRNode)) with inner_fn = the inner_fn defined above
#   - return a TensorBox(StorageBox(Pointwise))
# }

#----------------------------------------
# Explain make_pointwise.<locals>.inner
#----------------------------------------
loaders = [x.make_loader() for x in inputs]
# Here, inputs is a tuple of TensorBox objects:
# (
#   TensorBox(StorageBox(InputBuffer(name='arg0_1', layout=FixedLayout('cuda:0', torch.float32, size=[3, 9], stride=[9, 1])))), 
#   TensorBox(StorageBox(InputBuffer(name='arg1_1', layout=FixedLayout('cuda:0', torch.float32, size=[3, 9], stride=[9, 1]))))
# )

# x.make_loader() -> InputBuffer.make_loader --returns--> Buffer.make_loader.<locals>.loader
def loader(index):  # type: ignore[no-untyped-def]
    indexer = self.make_indexer() # Here, self is the InputBuffer object
    return ops.load(self.name or "unnamed", indexer(index)) # self.name is either 'arg0_1' or 'arg1_1'

# InputBuffer.make_indexer() -> ... -> FixedLayout.make_indexer
class FixedLayout(Layout):
    """A Tensor layout we cannot change"""

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        """A closure containing math to read a given element"""

        def indexer(index):  # type: ignore[no-untyped-def]
            assert len(index) == len(self.stride)
            assert len(index) == len(self.size)
            result = self.offset
            for idx, stride, sz in zip(index, self.stride, self.size):
                if sz != 1:
                    result = result + idx * stride
            return result

        return indexer

# indexer returns actual Buffer offset of an element with given index
# So loader(index) returns a loaded value at index in the Buffer

# Next, make_pointwise.<locals>.inner defines a inner_fn function that works on an index as follows
def inner_fn(index):
    assert len(index) == len(ranges), f"wrong ndim {index} {ranges}"
    if dtype == torch.bool and override_fn_when_input_bool is not None:
        return override_fn_when_input_bool(*[load(index) for load in loaders])
    elif (
        override_fn_when_gpu_float64
        and is_gpu_device
        and dtype == torch.float64
    ):
        return override_fn_when_gpu_float64(*[load(index) for load in loaders])
    else:
        inputs_loaded = []
        for inp_index, load in enumerate(loaders):
            out = load(index)
            inp_dtype = inputs[inp_index].get_dtype()
            if emulate_precision_casts and inp_dtype in low_pr_fp:
                downcast = ops.to_dtype(out, inp_dtype, use_compute_types=False)
                out = ops.to_dtype(downcast, inp_dtype)
            inputs_loaded.append(out)

        # inputs_loaded are loaded values from input Buffers for the same index
        out = fn(*inputs_loaded) # Here, fn is ops_wrapper(aten_fn.__name__)
        if emulate_precision_casts:
            # fp16/bf16 kernels are computed in fp32. Casting down to fp16/bf16 here,
            # then upcasting again, to emulate casts that eager would do.
            downcast = ops.to_dtype(out, dtype, use_compute_types=False)
            return ops.to_dtype(downcast, dtype)
        return out # the computed result for the index


# Next, create a Pointwise
pw = Pointwise.create(
    device=device,  # type: ignore[arg-type]
    dtype=dtype,
    inner_fn=inner_fn,
    ranges=ranges,
)

# Pointwise.create
@classmethod
def create(cls, *args: Any, **kwargs: Any) -> TensorBox:
    origin_node = kwargs.pop("origin_node", None)
    tb = kwargs.pop("traceback", None)
    # if "origin_node" in kwargs:
    #     breakpoint()
    r = cls(*args, **kwargs) # cls is Pointwise
    # Need to explicitly set origin_node here to propagate it down.
    # todo(chilli): I think it would be better for IRNode to directly set
    # origin_node
    r._post_init_setattr("origin_node", origin_node)
    r._post_init_setattr("traceback", tb or r.traceback)
    return TensorBox.create(r)

# TensorBox.create
class TensorBox(MutableBox):
    @staticmethod
    def create(data):  # type: ignore[no-untyped-def]
        if isinstance(data, ShapeAsConstantBuffer):
            return data
        return TensorBox(StorageBox(data)) # Here, data is the Pointwise object

# Return the TensorBox(StorageBox(Pointwise)) which captures
#   1. All the input Buffers
#   2. Computation of the aten op

# TensorBox objects are realized to Buffer nodes

#
# Output
#
# A list of ir.Buffer nodes
```

## Step 2: Fuse ir.Buffer nodes
```Python
# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/graph.py
class GraphLowering(torch.fx.Interpreter):
    def _update_scheduler(self) -> None:
        """
        (Re)initializes the scheduler member.  When initializing the scheduler, no CUBIN
        files should be generated (to avoid biasing any benchmarks and pessimizing
        fusion decisions).
        """
        from .scheduler import Scheduler

        with config.patch("triton.store_cubin", False):
            self.scheduler = Scheduler(self.operations)

    def codegen(self) -> tuple[ValueWithLineMap, ValueWithLineMap]:
        with dynamo_timed("GraphLowering.codegen", log_pt2_compile_event=True):
            self.init_wrapper_code()

            self._update_scheduler() # Fusion happens here
            V.debug.draw_orig_fx_graph(self.orig_gm, self.scheduler.nodes)

#================================================================================================================================================================================
# 2. Scheduler
#================================================================================================================================================================================
# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/scheduler.py
# class Scheduler:
# Input are a list of Buffer nodes:
nodes = [
  0: ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[3], stride=[1]), data=Reduction(
    'cuda',
    torch.float32,
    def inner_fn(index, rindex):
        i0 = index
        r0_0 = rindex
        tmp0 = ops.load(arg0_1, r0_0 + 9 * i0)
        tmp1 = ops.load(arg1_1, r0_0 + 9 * i0)
        tmp2 = tmp0 + tmp1
        return tmp2
    ,
    ranges=[3],
    reduction_ranges=[9],
    reduction_type=sum,
    origin_node=sum_1,
    origins=OrderedSet([sum_1, add])
  )),
  1: ComputedBuffer(name='buf1', layout=FixedLayout('cuda:0', torch.float32, size=[3], stride=[1]), data=Pointwise(device=device(type='cuda', index=0), dtype=torch.float32, inner_fn=<function make_pointwise.<locals>.inner.<locals>.inner_fn at 0x7fdcff2b34c0>, ranges=[3]))
]

# Create BaseSchedulerNode nodes from the Buffer nodes
self.nodes = [self.create_scheduler_node(n) for n in nodes]

def create_scheduler_node(self, node: ir.Operation) -> BaseSchedulerNode:
    assert node.get_origins() is not None, (
        "All nodes passed to scheduling must have an origin"
    )
    if node.is_no_op():
        return NopKernelSchedulerNode(self, node)
    elif isinstance(node, (ir.ComputedBuffer, ir.TemplateBuffer)):
        return SchedulerNode(self, node)
    elif isinstance(node, ir.ExternKernel):
        return ExternKernelSchedulerNode(self, node)
    else:
        raise NotImplementedError(node)

class SchedulerNode(BaseSchedulerNode):
    _sizes: tuple[Sequence[sympy.Expr], ...]
    _body: LoopBody

    def __init__(
        self,
        scheduler: Scheduler,
        node: Union[ir.ComputedBuffer, ir.TemplateBuffer],
    ) -> None:
        super().__init__(scheduler)
        self._init_from_node(node)
        self._compute_attrs()

# _compute_attrs(...)
self._sizes, self._body = self.node.simplify_and_reorder(
    extra_indexing_constraints=extra_indexing_constraints,
    recompute_sizes_body_func=recompute_sizes_body_func,
)

# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/ir.py
@ir_dataclass(frozen=False)
class ComputedBuffer(OperationBuffer):
    data: Loops

    def simplify_and_reorder(
        self,
        extra_indexing_constraints: Optional[tuple[dict[Any, Any], list[Any]]] = None,
        recompute_sizes_body_func: Optional[Callable[..., Any]] = None,
    ) -> tuple[tuple[list[sympy.Expr], list[sympy.Expr]], LoopBody]:
        """
        This is a main place where we do loop transformations in a
        backend-agnostic way.

        Here we:
            1) Remove any 1 dimensions
            2) Fuse contiguous dimensions together
            3) Reorder dimensions based on stride orders

        Optional argument extra_indexing_constraints can be used to append additional
        indexing expressions to existing ones derived from buffer's body. This can be useful
        to fuse scheduler nodes with compatible ranges, e.g. (s0*s1*...,) and (s0, s1, s2, ...)
        on CPU by preventing indexing simplifications and obtaining index/reduce ranges for
        the scheduler node compatible with other nodes.
        Optional argument recompute_sizes_body_func can be used to recompute sizes and body
        on the default body. This can be useful to append additional loop transformations.
        """
        (
            (index_size, reduce_size),
            body,
            (index_vars, reduce_vars),
        ) = self.get_default_sizes_body()

# get_default_sizes_body()
args = [(q0,), (q1,)]
var_ranges = {q0: 3, q1: 9}
```

### How does Scheduler fuse two SchedulerNode nodes?
Must really understand Scheduler IR
```Python
class Scheduler: 
  fuse_nodes(nodes):
    fuse_nodes_once(nodes):
      get_possible_fusions(nodes):
```

#### Understand the following IR:
```
op1461: SchedulerNode(ComputedBuffer)
op1461.writes = [MemoryDep('buf1461', c0, {c0: 7936*s0})]
op1461.unmet_dependencies = [   MemoryDep('buf1460', 7936*c0 + 64*c1 + 1984*c2 + c3, {c0: s0, c1: 31, c2: 4, c3: 64})]
op1461.met_dependencies = []
op1461.outputs = [
    buf1461: ComputedBuffer
    buf1461.layout = FixedLayout('cuda:0', torch.float16, size=[s0, 31, 4, 64], stride=[7936, 256, 64, 1])
    buf1461.users = [NodeUser(node=ExternKernelSchedulerNode(name='op1462'), can_inplace=False, is_weak=False)]
]
op1461.group.device = cuda:0
op1461.group.iteration = (7936*s0, 1)
op1461.sizes = ([s0, 31, 4, 64], [])
buf1460_layout = FixedLayout('cuda:0', torch.float16, size=[4*s0, 31, 64], stride=[1984, 64, 1])
buf1461_layout = FixedLayout('cuda:0', torch.float16, size=[s0, 31, 4, 64], stride=[7936, 256, 64, 1])
class op1461_loop_body:
    var_ranges = {p0: s0, p1: 31, p2: 4, p3: 64}
    index0 = 7936*p0 + 64*p1 + 1984*p2 + p3
    index1 = 7936*p0 + 256*p1 + 64*p2 + p3
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf1460', get_index)
        get_index_1 = self.get_index('index1')
        store = ops.store('buf1461', get_index_1, load, None)
        return store
```

#### Completely understand `class op1461_loop_body` and `def body(self, ops)`
Perform operations on 1 element


## Step 3: Codegen SchedulerNode nodes
```Python
# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/graph.py
class GraphLowering(torch.fx.Interpreter):
    def _update_scheduler(self) -> None:
        """
        (Re)initializes the scheduler member.  When initializing the scheduler, no CUBIN
        files should be generated (to avoid biasing any benchmarks and pessimizing
        fusion decisions).
        """
        from .scheduler import Scheduler

        with config.patch("triton.store_cubin", False):
            self.scheduler = Scheduler(self.operations)

    def codegen(self) -> tuple[ValueWithLineMap, ValueWithLineMap]:
        with dynamo_timed("GraphLowering.codegen", log_pt2_compile_event=True):
            self.init_wrapper_code()

            # Fusion happens here
            self._update_scheduler()
            V.debug.draw_orig_fx_graph(self.orig_gm, self.scheduler.nodes)

            # Codegen
            self.wrapper_code.push_codegened_graph(self)
            self.scheduler.codegen()

# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/codegen/common.py
    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        store_cache = self.kernel.cse.store_cache
        if name in store_cache:
            return store_cache[name]
        out = self.kernel.load(name, index)

# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/codegen/triton.py
    def load(self, name: str, index: sympy.Expr):
        indexing = self.indexing(index, block_ptr=True)

# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/codegen/simd.py
    def prepare_indexing(...):
        return self.codegen_indexing(simp_index)

    def codegen_indexing(self, expr: sympy.Expr) -> sympy.Expr:
        expr = V.graph.sizevars.simplify_with_ranges(expr, self.var_ranges())
        for sym in sorted(expr.free_symbols, key=str):
            if sym in self.range_tree_nodes:
                # if indexing expression is complicated, we precompute it on the host side
                # and send the result as a kernel argument
                replacements = {}
                for ps in self.range_tree_nodes[sym].precomputed_args():  # type: ignore[index]
                    replacements[ps] = V.graph.sizevars.lookup_precomputed_size(ps)
                if len(replacements) > 0:
                    self.range_tree_nodes[sym].expr = sympy_subs(  # type: ignore[index]
                        self.range_tree_nodes[sym].expr,
                        replacements,  # type: ignore[index]
                    )
                self.range_tree_nodes[sym].codegen()  # type: ignore[index]
        return expr
```

### How is the following IR lowered to a Triton kernel?
```
op0_op1: FusedSchedulerNode(SchedulerNode,SchedulerNode)
op0_op1.writes = [MemoryDep('buf0', c0, {c0: s0*s1}), MemoryDep('buf1', c0, {c0: s0*s1})]
op0_op1.unmet_dependencies = []
op0_op1.met_dependencies = [MemoryDep('arg1_1', c0, {c0: 9*s0}), MemoryDep('arg3_1', c0, {c0: 9*s1})]
op0_op1.outputs = [
    buf0: MultiTemplateBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[s1, s0], stride=[s0, 1])
    buf0.users = [NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False)]
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[s1, s0], stride=[s0, 1])
    buf1.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0_op1.snodes[0] =
op0: SchedulerNode(MultiTemplateBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: s0*s1})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg1_1', c0, {c0: 9*s0}), MemoryDep('arg3_1', c0, {c0: 9*s1})]
op0.outputs = [
    buf0: MultiTemplateBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[s1, s0], stride=[s0, 1])
    buf0.users = [NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (s0*s1, 1)
op0.sizes = ([s1, s0], ())
arg3_1_layout = FixedLayout('cuda:0', torch.float32, size=[s1, 9], stride=[9, 1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[9, s0], stride=[s0, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[s1, s0], stride=[s0, 1])
op0_op1.snodes[1] =
op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', c0, {c0: s0*s1})]
op1.unmet_dependencies = [MemoryDep('buf0', c0, {c0: s0*s1})]
op1.met_dependencies = []
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[s1, s0], stride=[s0, 1])
    buf1.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (s0*s1, 1)
op1.sizes = ([s0*s1], [])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[s1, s0], stride=[s0, 1])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[s1, s0], stride=[s0, 1])
class op1_loop_body:
    var_ranges = {p0: s0*s1}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf0', get_index)
        sigmoid = ops.sigmoid(load)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf1', get_index_1, sigmoid, None)
        return store

```

# Triton Kernel
```Python
# Python code:
#   /path/to/model/aot_compile_cache/inductor/bj/cbj4ocbh5avlulkuyxty3hyad5gk4xghgpisjet45smce6m4yrdr.py
# IR path:
#   /path/to/model/aot_compile_cache/inductor/triton/0/5JCCLTSPXZL37Z5XFEMM7ZYJRUTWKNXOAEGR23673G6PXO5OWF3A/triton_mm.ttir
```
<br/>

How to find the corresponding Triton kernel:
```Bash
cd /path/to/model/aot_compile_cache/inductor/
grep -R 'N = 1781'
```

From Python code filename, we can look up IR as follows:
```Bash
cd /path/to/model/aot_compile_cache/inductor/triton/
grep -R cbj4ocbh5avlulkuyxty3hyad5gk4xghgpisjet45smce6m4yrdr.py
```
<br/>


# Codegen Combo Kernels
```Python
# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/codegen/simd.py
    def codegen_combo_kernel(self, combo_kernel_node):
        subkernel_nodes = combo_kernel_node.get_subkernel_nodes()
        custom_part_algorithm = combo_kernel_node.use_custom_partition_algo
        enable_autotune = combo_kernel_node.enable_autotune
        mixed_sizes = config.combo_kernel_allow_mixed_sizes > 1 or (
            config.combo_kernel_allow_mixed_sizes == 1 and custom_part_algorithm
        )

        kernel_code_list = self.generate_combo_kernel_code(
            subkernel_nodes, custom_part_algorithm, enable_autotune, mixed_sizes
        )

        for src_code, kernel, _ in kernel_code_list:
            kernel_name = self.define_kernel(src_code, [combo_kernel_node], kernel)
            self.codegen_comment([combo_kernel_node])
            log.debug("ComboKernels: generated kernel %s.", kernel_name)
            kernel.call_kernel(V.graph.wrapper_code, kernel_name)

        self.free_buffers_in_scheduler()


# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/codegen/triton_combo_kernel.py
```

# Triton Config
```Python
# /data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/runtime/triton_heuristics.py
def triton_config(
    size_hints,
    x,
    y=None,
    z=None,
    num_stages=1,
    num_elements_per_warp=256,
    min_elem_per_thread=0,
) -> Config:
...
    x, _num_blocks = _check_max_grid_x(size_hints, x, num_warps)
    x = min(x, size_hints["x"])
    x = min(x, config.triton.max_xblock) # This may be needed

    cfg = {"XBLOCK": x}
    if y:
        cfg["YBLOCK"] = y
    if z:
        cfg["ZBLOCK"] = z
    check_max_block(cfg)
    check_config(cfg, xnumel=xnumel, ynumel=ynumel, znumel=znumel)
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)
```

# Get compiled module path
At runtime, the compiled module will be loaded by class SerializableCallable(torch.nn.Module)
```Python
# site-packages/torch/_functorch/aot_autograd.py
class SerializableCallable(torch.nn.Module):
    @staticmethod
    def deserialize(compiled_fn, runtime_metadata):
        path = get_path(compiled_fn.cache_key, "py")[2]
        compiled_fn.current_callable = PyCodeCache.load_by_key_path(
            compiled_fn.cache_key,
            path,
            compiled_fn.cache_linemap,
            compiled_fn.constants,
        ).call
        return SerializableCallable(compiled_fn, runtime_metadata)
```
Because this class is serialized, so if we modify it, the change won't take effect.<br/>
In order to get the path to the compiled module, we need to modify **PyCodeCache.load_by_key_path()** as follows:

```Python
# site-packages/torch/_inductor/codecache.py
class PyCodeCache:
    @classmethod
    def load_by_key_path(
        cls,
        key: str,
        path: str,
        linemap: Optional[list[tuple[int, str]]] = None,
        attrs: Optional[dict[str, Any]] = None,
    ) -> ModuleType:
        if linemap is None:
            linemap = []

        # print(f"Module path: {path}")
        mod = _reload_python_module(key, path)
        if hasattr(mod, "benchmark_compiled_module"):
            print(f"\n=> Compiled module: {path}\n")
```

# Debug Compiled Module
After we get the path to the compiled module, we can add a **print()** statement for debugging:
```Python
def call(args):
        triton_poi_fused_4.run(_tensor_constant0, arg1_1, buf81, buf147, triton_poi_fused_4_xnumel_1, stream=stream0)
        print(f"buf81: {buf81}")
```
When we run the compiled model with export SKIP_TRACE=1, Dynamo will load the compiled module and call the function **call(args)**.

# Debug Triton Kernels
Directly modify Triton kernels inside the compiled module and then rerun the compiled module. <br/>
For example,
```Python
@triton.jit
def triton_poi_fused_4(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel_1, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(2305, XBLOCK)
    tl.device_print("[SONY] num_xblocks_0", num_xblocks_0) # <-- Add tl.device_print
```
Output:
```Bash
pid (2, 0, 0) idx () [SONY] num_xblocks_0: 10
pid (2, 0, 0) idx () [SONY] num_xblocks_0: 10
pid (2, 0, 0) idx () [SONY] num_xblocks_0: 10
...
```

# Triton GEMM Autotune
```Python
# /data00/home/son.nguyen/workspace/torch_dev/optimizer/site-packages/torch/_inductor/select_algorithm.py
# `extra_args` including `grid` are passed into `bmreq_cls`
class TritonTemplate(KernelTemplate):
    def generate(...):  # type: ignore[override]
        bmreq_cls: type[TritonBenchmarkRequest]
        if layout.device.type == "cpu":
            bmreq_cls = TritonCPUBenchmarkRequest
        else:
            bmreq_cls = TritonGPUBenchmarkRequest

        bmreq = bmreq_cls(
            module_path=result.mod.__file__,
            module_cache_key=result.mod.key,
            kernel_name=f"triton_{self.name}",
            extra_args=[*extra_args, *workspace_args, *grid],
            num_stages=num_stages,
            num_warps=num_warps,
            num_consumer_groups=num_consumer_groups,
            num_buffers_warp_spec=num_buffers_warp_spec,
            matrix_instr_nonkdim=kwargs.get("matrix_instr_nonkdim", 0),
            waves_per_eu=kwargs.get("waves_per_eu", 0),
            kpack=kwargs.get("kpack", 2),
            input_tensor_meta=TensorMeta.from_irnodes(full_input_nodes),  # type: ignore[arg-type]
            output_tensor_meta=TensorMeta.from_irnodes(layout),
        )

# /data00/home/son.nguyen/workspace/torch_dev/optimizer/site-packages/torch/_inductor/autotune_process.py
class TritonGPUBenchmarkRequest(GPUDeviceBenchmarkMixin, TritonBenchmarkRequest):
    pass

class TritonBenchmarkRequest(BenchmarkRequest):
    def benchmark(
        self,
        *input_tensors: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
    ) -> float:

# /data00/home/son.nguyen/workspace/torch_dev/optimizer/site-packages/torch/_inductor/select_algorithm.py
class TritonTemplateCaller(ir.TritonTemplateCallerBase):
    def benchmark(self, *args, out):
        assert self.bmreq is not None
        return self.bmreq.benchmark(*args, output_tensor=out) # self.bmreq is an instance of TritonGPUBenchmarkRequest

# /data00/home/son.nguyen/workspace/torch_dev/optimizer/site-packages/torch/_inductor/autotune_process.py
# `self.make_run_fn` builds extra arguments
class TritonBenchmarkRequest(BenchmarkRequest):
    def benchmark(
        self,
        *input_tensors: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
    ) -> float:
        try:
            fn = self.make_run_fn(*input_tensors, out=out)
        except NonzeroWorkspaceNotSupportedError:
            # Skipping all ops with nonzero workspace requirements
            autotuning_log.info("Skipping op due to nonzero workspace requirement")
            return float("inf")

        try:
            if hasattr(self, "module_cache_key") and self.module_cache_key == "ctxjmjdz4ko4zzxcyqompjgo67d47coay75po32swkil5cuhp4kz":
                ipdb.set_trace()
                print(f"Module path: {self.module_path}")

            out = self.do_bench(fn, *input_tensors, output_tensor)
        except Exception as ex:
            msg = str(ex)
            if "misaligned address" in msg:
                ipdb.set_trace()
                print("[SONY_DEBUG]: {ex}")
            raise ex

# /data00/home/son.nguyen/workspace/torch_dev/optimizer/site-packages/torch/_inductor/autotune_process.py
class GPUDeviceBenchmarkMixin:
    def do_bench(
        self,
        fn,
        *input_tensors: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
    ) -> float:
        with device_interface.device(device_idx):  # type: ignore[attr-defined]
            out = benchmarker.benchmark_gpu(fn)
            device_interface.synchronize()  # shake out any CUDA errors

# /data00/home/son.nguyen/workspace/torch_dev/optimizer/site-packages/torch/_inductor/runtime/benchmarking.py
benchmarker = (
    InductorBenchmarker() if use_experimental_benchmarker else TritonBenchmarker()
)
# By default, use_experimental_benchmarker is True
class InductorBenchmarker(TritonBenchmarker):
    @time_and_count
    def benchmark_gpu(
        self: Self,
        _callable: Callable[[], Any],
        estimation_iters: int = 5,
        memory_warmup_iters: int = 100,
        benchmark_iters: int = 100,
        max_benchmark_duration: int = 25,
        **kwargs: Any,
    ) -> float:
        """Benchmark a GPU callable using a custom benchmarking implementation.

        Arguments:
        - _callable: The callable to benchmark.

        Keyword Arguments:
        - estimation_iters: Optionally, the number of iterations to run `_callable`
        during runtime estimation.
        - memory_warmup_iters: Optionally, the number of iterations to flush the L2
        cache before starting benchmarking.
        - benchmark_iters: Optionally, the number of iterations to run `_callable`
        during the benchmarking.
        - max_benchmark_duration: Optionally, the maximum duration of the benchmarking,
        in milliseconds. An estimated duration is calculated based on the values
        of `memory_warmup_iters` and `benchmark_iters`, along with the estimated
        runtime of `_callable` and various other factors, and we then shrink
        `benchmark_iters` to fit in the alloted maximum duration.
        - **kwargs: Additional kwargs that may be passed to the fallback.

        Returns:
        - The minimum runtime of `_callable`, in milliseconds.
        """
```

# Triton Kernel run method
Given [cases/triton_mm_ok.py](./cases/triton_mm_ok.py):
```Python
triton_path = "./cases/triton_mm_ok.py"
key = os.path.basename(triton_path).split(".")[0]
mod = PyCodeCache.load_by_key_path(key, triton_path)
mod.triton_mm.run(
    A, B, Out,
    _grid_0=grid_0,
    _grid_1=1,
    _grid_2=1,
    stream=stream.cuda_stream
)

# The `run` method is defined here
# /data00/home/son.nguyen/workspace/torch_dev/optimizer/site-packages/torch/_inductor/runtime/triton_heuristics.py
class CachingAutotuner(KernelInterface):
    def run(
        self,
        *args,
        stream,
        benchmark_run=False,
        **kwargs,
    ):  # type:ignore[override]
        if len(self.launchers) != 1:
            if len(self.launchers) == 0:
                start_time = time.time_ns()
                self.precompile() # Precompile the kernel
                self.precompile_time_taken_ns = time.time_ns() - start_time
            if len(self.launchers) > 1:
                self.autotune_to_one_config(*args, **kwargs)

        if autograd_profiler._is_profiler_enabled:
            # omit for brevity
        else:
            return launcher(
                *args,
                **kwargs,
                stream=stream,
            )
```


# Run Triton Kernel
```Bash
# Run a Triton DSL kernel
python3.11 run_triton_mm.py

# Launch a kernel from cubin
python3.11 launch_triton_cubin.py
```

# GEMM Template Rendering
Call Stack:
```Python
ipdb> where
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/test_inductor_mm_hopper.py(63)<module>()
     62   y1 = (torch.rand(64, 1024, dtype=torch.bfloat16).cuda() - 0.5)
---> 63   res1 = toy(x1, y1.T.contiguous())
     64   print()
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/nn/modules/module.py(1775)_wrapped_call_impl()
   1774         else:
-> 1775             return self._call_impl(*args, **kwargs)
   1776 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/nn/modules/module.py(1786)_call_impl()
   1785                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1786             return forward_call(*args, **kwargs)
   1787 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/eval_frame.py(832)compile_wrapper()
    831                 try:
--> 832                     return fn(*args, **kwargs)
    833                 except Unsupported as e:
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/convert_frame.py(1875)__call__()
   1874             # skip=1: skip this frame
-> 1875             result = self._torchdynamo_orig_backend(
   1876                 frame, cache_entry, self.hooks, frame_state, skip=1
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/convert_frame.py(1625)__call__()
   1624         try:
-> 1625             result = self._inner_convert(
   1626                 frame, cache_entry, hooks, frame_state, skip=skip + 1
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/convert_frame.py(688)__call__()
    687         with compile_context(CompileContext(compile_id)):
--> 688             result = _compile(
    689                 frame.f_code,
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/convert_frame.py(1434)_compile()
   1433         try:
-> 1434             guarded_code, tracer_output = compile_inner(code, one_graph, hooks)
   1435 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_utils_internal.py(92)wrapper_function()
     91             if not StrobelightCompileTimeProfiler.enabled:
---> 92                 return function(*args, **kwargs)
     93 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/convert_frame.py(1117)compile_inner()
   1116             stack.enter_context(CompileTimeInstructionCounter.record())
-> 1117             return _compile_inner(code, one_graph, hooks)
   1118 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/convert_frame.py(1151)_compile_inner()
   1150         try:
-> 1151             dynamo_output = compile_frame(
   1152                 code,
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/convert_frame.py(1032)compile_frame()
   1031             with dynamo_timed(f"compile_attempt_{attempt}", log_pt2_compile_event=True):
-> 1032                 bytecode, tracer_output = transform_code_object(code, transform)
   1033                 assert tracer_output is not None
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/bytecode_transformation.py(1592)transform_code_object()
   1591 
-> 1592     tracer_output = transformations(instructions, code_options)
   1593     _, bytecode = clean_and_assemble_instructions(instructions, keys, code_options)
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/convert_frame.py(1004)transform()
   1003         )
-> 1004         tracer_output = trace_frame(
   1005             code,
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/convert_frame.py(312)_fn()
    311             try:
--> 312                 return fn(*args, **kwargs)
    313             finally:
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/convert_frame.py(815)trace_frame()
    814     try:
--> 815         run_tracer()
    816         tracer_output = DynamoTracerOutput(tracer)
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/convert_frame.py(797)run_tracer()
    796             with tracing(tracer.output.tracing_context), tracer.set_current_tx():
--> 797                 tracer.run()
    798         except exc.UnspecializeRestartAnalysis:
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/symbolic_convert.py(1500)run()
   1499                 try:
-> 1500                     while self.step():
   1501                         pass
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/symbolic_convert.py(1348)step()
   1347         try:
-> 1348             self.dispatch_table[inst.opcode](self, inst)
   1349             return not self.output.should_exit
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/symbolic_convert.py(4103)RETURN_VALUE()
   4102     def RETURN_VALUE(self, inst: Instruction) -> None:
-> 4103         self._return(inst)
   4104 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/symbolic_convert.py(4081)_return()
   4080         log.debug("%s triggered compile", inst.opname)
-> 4081         all_stack_locals_metadata = self.output.compile_subgraph(
   4082             self,
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/output_graph.py(1568)compile_subgraph()
   1567                 output.extend(
-> 1568                     self.compile_and_call_fx_graph(tx, pass2.graph_output_vars(), root)
   1569                 )
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/output_graph.py(2013)compile_and_call_fx_graph()
   2012             with self.restore_global_state():
-> 2013                 compiled_fn = self.call_user_compiler(gm, self.example_inputs())
   2014 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/output_graph.py(2136)call_user_compiler()
   2135         ):
-> 2136             return self._call_user_compiler(gm, example_inputs)
   2137 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/output_graph.py(2171)_call_user_compiler()
   2170                 compiler_fn = WrapperBackend(compiler_fn)
-> 2171             compiled_fn = compiler_fn(gm, example_inputs)
   2172             _step_logger()(logging.INFO, f"done compiler function {name}")
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/repro/after_dynamo.py(156)__call__()
    155         else:
--> 156             compiled_gm = compiler_fn(gm, example_inputs)
    157 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/__init__.py(2392)__call__()
   2391 
-> 2392         return compile_fx(model_, inputs_, config_patches=self.config)
   2393 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/compile_fx.py(2681)compile_fx()
   2680             try:
-> 2681                 return aot_autograd(
   2682                     fw_compiler=fw_compiler,
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/backends/common.py(117)__call__()
    116             with enable_aot_logging(), patch_config:
--> 117                 cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
    118                 counters["aot_autograd"]["ok"] += 1
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_functorch/aot_autograd.py(1106)aot_module_simplified()
   1105             aot_graph_capture = aot_stage1_graph_capture(aot_state, functional_call)
-> 1106             compiled_fn, _ = aot_stage2_compile(aot_state, aot_graph_capture)
   1107 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_functorch/_aot_autograd/graph_compile.py(242)aot_stage2_compile()
    241     else:
--> 242         return aot_stage2_inference(aot_state, aot_graph_capture)
    243 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_functorch/_aot_autograd/graph_compile.py(315)aot_stage2_inference()
    314                 tensorify_python_scalars(fw_module, fake_mode.shape_env, fake_mode)
--> 315             compiled_fw = compiler(fw_module, updated_flat_args)
    316 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_functorch/_aot_autograd/schemas.py(1251)__call__()
   1250     ) -> OutputCode:
-> 1251         return self.compiler_fn(gm, example_inputs)
   1252 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/compile_fx.py(2558)fw_compiler_base()
   2557                     num_orig_model_outputs = get_num_model_outputs(gm)
-> 2558                 return compile_fx_forward(
   2559                     gm,
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/compile_fx.py(2275)compile_fx_forward()
   2274 
-> 2275     return inner_compile(
   2276         gm,
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/compile_fx.py(782)compile_fx_inner()
    781         )
--> 782         return wrap_compiler_debug(_compile_fx_inner, compiler_name="inductor")(
    783             gm,
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_dynamo/repro/after_aot.py(144)debug_wrapper()
    143             # with fake inputs
--> 144             inner_compiled_fn = compiler_fn(gm, example_inputs)
    145         except Exception:
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/compile_fx.py(974)_compile_fx_inner()
    973             try:
--> 974                 mb_compiled_graph = fx_codegen_and_compile(
    975                     gm, example_inputs, inputs_to_check, **graph_kwargs
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/compile_fx.py(1695)fx_codegen_and_compile()
   1694 
-> 1695     return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
   1696 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/compile_fx.py(1420)codegen_and_compile()
   1419                 with V.set_graph_handler(graph), V.set_extern_kernel_nodes([]):
-> 1420                     graph.run(*example_inputs)
   1421                     output_strides: list[Optional[tuple[_StrideExprStr, ...]]] = []
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/graph.py(937)run()
    936         with dynamo_timed("GraphLowering.run"):
--> 937             return super().run(*args)
    938 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/fx/interpreter.py(174)run()
    173             try:
--> 174                 self.env[node] = self.run_node(node)
    175             except Exception as e:
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/graph.py(1624)run_node()
   1623                 debug("")
-> 1624                 result = super().run_node(n)
   1625 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/fx/interpreter.py(256)run_node()
    255             assert isinstance(kwargs, dict)
--> 256             return getattr(self, n.op)(n.target, args, kwargs)
    257 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/graph.py(1279)call_function()
   1278 
-> 1279             out = lowerings[target](*args, **kwargs)  # type: ignore[index]
   1280 
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/lowering.py(488)wrapped()
    487 
--> 488         out = decomp_fn(*args, **kwargs)
    489         validate_ir(out)
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/kernel/mm.py(795)tuned_mm()
    794         # Get template choices using the new unified function
--> 795         choices.extend(
    796             V.choices.get_mm_configs(kernel_inputs, layout, [mm_template], "mm")
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/choices.py(166)get_mm_configs()
    165             for c in cs:
--> 166                 choice = template.choice_or_none(**{**c, **overrides}, **extra_kwargs)
    167                 if choice is not None:
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/codegen/common.py(2431)choice_or_none()
   2430         temp_choices: list[Any] = []
-> 2431         result = self.maybe_append_choice(temp_choices, **kwargs)
   2432         if result is None and len(temp_choices) == 1:
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/select_algorithm.py(1467)maybe_append_choice()
   1466         try:
-> 1467             choice = self.generate(generate_with_caching=True, **kwargs)
   1468             if choice is not None:
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/select_algorithm.py(1724)generate()
   1723 
-> 1724         result = self.generate_and_load(
   1725             input_nodes,
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/select_algorithm.py(1650)generate_and_load()
   1649             else:
-> 1650                 result = generate_code(kernel)
   1651                 if result is None:  # happens at ZeroDivisionError:
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/select_algorithm.py(1602)generate_code()
   1601             try:
-> 1602                 template = kernel.render(self.template, kwargs, caching_enabled)
   1603                 with kernel.set_subgraph_body("<STORE_OUTPUT>"):
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/select_algorithm.py(1171)render()
   1170         return PartialRender(
-> 1171             template.render(**template_env, **kwargs),
   1172             self.render_hooks,
  /data05/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/jinja2/environment.py(1293)render()
   1292         try:
-> 1293             return self.environment.concat(self.root_render_func(ctx))  # type: ignore
   1294         except Exception:
  <template>(19)root()
  /data05/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/jinja2/runtime.py(303)call()
    302         try:
--> 303             return __obj(*args, **kwargs)
    304         except StopIteration:
  /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/select_algorithm.py(489)wrapper()
    488                 pre_state = self.input_dependent_preserved_state()
--> 489                 result = fn(*args, **kwargs)
    490                 post_state = self.input_dependent_preserved_state()
> /data05/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/select_algorithm.py(658)def_kernel()
    657 
--> 658         assert all(isinstance(x, str) for x in argnames)
    659         renames = IndentedBuffer(initial_indent=1)

DEBUG:asyncio:Using selector: EpollSelector
ipdb>
```

**MLCompiler/pytorch/torch/_inductor/kernel/mm.py**
```Python
@register_lowering(aten.mm, type_promotion_kind=None)
def tuned_mm(mat1, mat2, *, layout=None):
    ...
    if is_nonzero and use_triton_template(layout, check_max_autotune=False):
        # Get template choices using the new unified function
        choices.extend(
            V.choices.get_mm_configs(kernel_inputs, layout, [mm_template], "mm")
        )
```

**MLCompiler/pytorch/torch/_inductor/choices.py**
```Python
class InductorChoices:
    def get_mm_configs(
        self,
        kernel_inputs: KernelInputs,
        layout: Any,
        templates: list[Union[KernelTemplate, ExternKernelChoice]],
        op_name: str,
        kwarg_overrides: Optional[dict[str, dict[str, Any]]] = None,
    ) -> Generator[ChoiceCaller, None, None]:
        ...
        for template in templates:
            # Extract template_name from the template object
            template_name = template.uid

            # Get the appropriate template-specific heuristic
            heuristic = get_template_heuristic(template_name, device_type, op_name)

            # 
            # kernel_inputs encapsulates 2 input nodes:
            # 
            # kernel_inputs._input_nodes
            # [StorageBox(
            #   InputBuffer(name='arg0_1', layout=FixedLayout('cuda:0', torch.bfloat16, size=[256, 1024], stride=[1024, 1]))
            # ), StorageBox(
            #   InputBuffer(name='arg1_1', layout=FixedLayout('cuda:0', torch.bfloat16, size=[1024, 64], stride=[64, 1]))
            # )]

            # 
            # layout
            # 
            # layout is the layout of matrix D
            # FixedLayout('cuda:0', torch.bfloat16, size=[256, 64], stride=[64, 1])

            cs = heuristic.get_template_configs(
                kernel_inputs,
                layout,
                op_name,
            )
            extra_kwargs = heuristic.get_extra_kwargs(kernel_inputs, layout, op_name)

            # Extract layout and input_nodes from extra_kwargs to pass them explicitly
            layout_val = layout
            # adjust the kernel inputs to the template-specific heuristic, if needed
            # default here is to just return the kernel_inputs as is
            input_nodes_val = heuristic.adjust_kernel_inputs(
                kernel_inputs, op_name
            ).nodes()

            # Get overrides for this specific template
            overrides = kwarg_overrides.get(template.uid, {})

            extra_kwargs["layout"] = layout_val
            extra_kwargs["input_nodes"] = input_nodes_val
            for c in cs:
                choice = template.choice_or_none(**{**c, **overrides}, **extra_kwargs)
                if choice is not None:
                    yield choice

```