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
    │
    └─> convert_frame.py: _compile(...) returns guarded_code
    │
    └─> convert_frame.py: class CatchErrorsWrapper returns the guarded_code to the caller in `torch/csrc/dynamo/eval_frame.c`.
    │
    └─> The caller evaluates the guarded code

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


# FX Graph Lowering Process
```Python
# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/compile_fx.py
with V.set_graph_handler(graph):
    graph.run(*example_inputs)

graph():
    %arg0_1 : [num_users=2] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
    %view : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add, [4, 3]), kwargs = {})
    %mm : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%arg0_1, %view), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mm, [1]), kwargs = {})
    %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%sum_1,), kwargs = {})
    return (sigmoid,)
```

## Lower torch.ops.aten.add.Tensor
V0828 02:13:45.831000 3131540 site-packages/torch/_inductor/graph.py:1455] [0/0] lowering %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {}) 
V0828 02:13:45.832000 3131540 site-packages/torch/_inductor/graph.py:1181] [0/0]   via <function make_pointwise.<locals>.inner at 0x7fd451003100>
```Python
# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/graph.py:1189
lowered_fn = lowerings[target]
out = lowered_fn(*args, **kwargs)  # type: ignore[index]
print(lowered_fn)
<function make_pointwise.<locals>.inner at 0x7f098a7f7100>

# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/lowering.py
add = register_pointwise(
    aten.add, allow_alpha=True, override_fn_when_input_bool="logical_or"
)

# /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/torch/_inductor/lowering.py
def _register_lowering(
    aten_fn,
    decomp_fn,
    broadcast,
    type_promotion_kind: Optional[ELEMENTWISE_TYPE_PROMOTION_KIND],
    convert_input_to_bool,
):
    """
    Add a lowering to lowerings dict

    Arguments:
        aten_fn: torch.ops.aten.* fn we are lowering
        decomp_fn: alternate implementation on our IR
        broadcast: True to apply broadcasting to tensor inputs
        type_promotion_kind: kind of type promotion applied to tensor inputs, `None` means no type promotion
        convert_input_to_bool: some logical ops require inputs are converted to bool
    """

    @functools.wraps(decomp_fn) # Capture metadata of decomp_fn
    def wrapped(*args, **kwargs): # Wrap decomp_fn
        args: list[Any] = list(args)
        kwargs: dict[str, Any] = dict(kwargs)
        unpacked = False
        # TODO maybe we need to use pytrees here
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            unpacked = True
            args = list(args[0])

        if not all(
            (fn in fallbacks or in_namespace(fn, "_c10d_functional")) for fn in aten_fn
        ):
            # explicitly assert for "out=" ops for better error messages
            assert not any(x == "out" for x in kwargs.keys()), (
                "out= ops aren't yet supported"
            )

        args, kwargs = transform_args(
            args, kwargs, broadcast, type_promotion_kind, convert_input_to_bool
        )

        if unpacked:
            args = [args]

        out = decomp_fn(*args, **kwargs)
        validate_ir(out)

        return out

    aten_fn = get_overloads(aten_fn)

    lowerings.update(dict.fromkeys(aten_fn, wrapped))
    return wrapped

print(decomp_fn)
<function make_pointwise.<locals>.inner at 0x7fd451003060>
```