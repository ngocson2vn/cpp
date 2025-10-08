import sys
import os
import shutil

debug_dir = "./scheduler_debug2"

if os.path.exists(debug_dir):
    shutil.rmtree(debug_dir)
    print(f"Cleaned {debug_dir}")
else:
    os.makedirs(debug_dir, exist_ok=True)
    print(f"Created {debug_dir}")

# """
os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

# Enable IR dumping for Inductor
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCH_COMPILE_DEBUG_DIR"] = debug_dir
os.environ["DYNAMO_CKPT_PATH"] = f"{debug_dir}/aot"
os.environ['TORCHINDUCTOR_CACHE_DIR'] = f"{debug_dir}/aot/inductor"
os.environ["INDUCTOR_POST_FUSION_SVG"] = "1"
os.environ["INDUCTOR_ORIG_FX_SVG"] = "1"
os.environ["TORCHINDUCTOR_PROLOGUE_FUSION"] = "1"
os.environ["TORCHINDUCTOR_DEBUG_FUSION"] = "1"
os.environ["TORCHDYNAMO_COMPILE_DYNAMIC_SHAPE"] = "1"

os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "TRITON"

# Extended vertical fusion
os.environ["TORCHINDUCTOR_EXTENDED_VERTICAL_FUSION"] = "1"
# """

import torch
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("torch._inductor.scheduler").setLevel(logging.DEBUG)
from torch._inductor.scheduler import fusion_log, loop_ordering_log
fusion_log.propagate = True
loop_ordering_log.propagate = True

import contextlib

from torch.fx.graph_module import GraphModule

from torch._inductor.debug import DebugContext
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
  Layout,
  FixedLayout,
  FlexibleLayout,
  TensorBox,
  StorageBox,
  View,
  Pointwise,
  InputBuffer,
  ConstantBuffer,
  ShapeAsConstantBuffer,
  ExternKernelOut,
  ComputedBuffer
)

from torch.utils._sympy.value_ranges import ValueRanges

aten = torch.ops.aten
from torch._inductor.virtualized import V

from torch._inductor.lowering import lowerings
import sympy

from typing import TypeVar

T = TypeVar("T")
def identity(x: T) -> T:
    return x

def make_dummy_gm():
    return torch.fx.symbolic_trace(identity)

gm = make_dummy_gm()
graph_id = 0

graph = GraphLowering(
    gm,
    # example_inputs will be used by AOTInductor to dry-run the generated code for Triton kernel tuning.
    # For the forward pass, we have the real inputs to be used as example_inputs. For the backward pass,
    # we currently use fake tensors and defake them later.
    example_inputs=None,
    shape_env=None,
    graph_id=graph_id,
    cpp_wrapper=False,
    aot_mode=False,
    extern_node_serializer=None,
    is_inference=True,
    is_backward=False,
    const_output_index=None,
    const_wrapper_code=None,
    const_kernel_code=None,
    const_module=None,
    inputs_to_check=None,
)

DEVICE = torch.device("cuda", 0)
V.set_graph_handler(graph)

stack = contextlib.ExitStack()
dbg_ctx = DebugContext()
stack.enter_context(dbg_ctx)

"""
op0: ExternKernelSchedulerNode(ExternKernelOut)
op0.writes = [StarDep(name='buf0', mode=None)]
op0.unmet_dependencies = [StarDep(name='buf1454', mode=None), StarDep(name='buf1459', mode=None)]
op0.met_dependencies = []
op0.outputs = [
    buf0: ExternKernelOut
    buf0.layout = FixedLayout('cuda:0', torch.float16, size=[4*s0, 31, 64], stride=[1984, 64, 1])
    buf0.users = [NodeUser(node=SchedulerNode(name='op1'), can_inplace=False, is_weak=False)]
]
op0.node.kernel = extern_kernels.bmm
"""

s0 = sympy.Symbol("s0", positive=True, is_number = True)
graph.graph_inputs['arg0_1'] = s0

V.graph.sizevars.shape_env.size_like.add(s0)
V.graph.sizevars.shape_env.var_to_range[s0] = ValueRanges(lower=8, upper=512)
V.graph.sizevars.shape_env.var_to_val[s0] = 256
# V.graph.sizevars.shape_env.replacements[7936*s0] = 7936*256


input0 = ConstantBuffer(name='arg1_1', layout=FixedLayout(DEVICE, torch.float16, size=[4*s0, 31, 64], stride=[1984, 64, 1]))
graph.graph_inputs['arg1_1'] = TensorBox(StorageBox(input0))

input1 = ConstantBuffer(name='arg2_1', layout=FixedLayout(DEVICE, torch.float16, size=[4*s0, 64, 64], stride=[4096, 64, 1]))
graph.graph_inputs['arg2_1'] = TensorBox(StorageBox(input1))

V.graph.graph_input_names.extend(graph.graph_inputs.keys())

# layout0 = FixedLayout(DEVICE, torch.float16, size=[4*s0, 31, 64], stride=[1984, 64, 1])
# buf0 = ExternKernelOut(layout0, [input0, input1], python_kernel_name="extern_kernels.bmm")
# x = TensorBox(StorageBox(buf0))

# Triton bmm template
target0 = aten.bmm
x = lowerings[target0](input0, input1)

"""
op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', c0, {c0: 7936*s0})]
op1.unmet_dependencies = [   MemoryDep('buf0', 7936*c0 + 64*c1 + 1984*c2 + c3, {c0: s0, c1: 31, c2: 4, c3: 64})]
op1.met_dependencies = []
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float16, size=[s0, 31, 4, 64], stride=[7936, 256, 64, 1])
    buf1.users = [NodeUser(node=ExternKernelSchedulerNode(name='op1462'), can_inplace=False, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (7936*s0, 1)
op1.sizes = ([s0, 31, 4, 64], [])
buf0_layout = FixedLayout('cuda:0', torch.float16, size=[4*s0, 31, 64], stride=[1984, 64, 1])
buf1_layout = FixedLayout('cuda:0', torch.float16, size=[s0, 31, 4, 64], stride=[7936, 256, 64, 1])
class op1_loop_body:
    var_ranges = {p0: s0, p1: 31, p2: 4, p3: 64}
    index0 = 7936*p0 + 64*p1 + 1984*p2 + p3
    index1 = 7936*p0 + 256*p1 + 64*p2 + p3
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf0', get_index)
        get_index_1 = self.get_index('index1')
        store = ops.store('buf1', get_index_1, load, None)
        return store
"""

# 1. RESHAPE
# size=[4*s0, 31, 64] -> [s0, 4, 31, 64]
size1 = [s0, 4, 31, 64]
target1 = aten.reshape
tmp1 = lowerings[target1](x, size1)

# 2. PERMUTE
# size=[s0, 4, 31, 64], stride=[7936, 1984, 64, 1] -> size=[s0, 31, 4, 64], stride=[7936, 256, 64, 1]
# To materialize this permuation, we need to access the original tensor with a permuted stride=[7936, 64, 1984, 1]
# Given a new coordinate {i0: range(s0), i1: range(31), i2: range(4), i3: range(64)},
# Old:
#   - index0 = 7936*i0 + 64*i1 + 1984*i2 + i3
#   - value0 = tensor0[index0]
# New:
#   - index1 = 7936*i0 + 256*i1 + 64*i2 + i3
#   - tensor1[index1] = value0
target2 = aten.permute
tmp2 = lowerings[target2](tmp1, dims=[0, 2, 1, 3])

# 3. REALIZE
tmp3 = Pointwise.create(device=DEVICE, dtype=torch.float16, inner_fn=tmp2.make_loader(), ranges=tmp2.get_size())
buf1 = tmp3.realize()

graph.graph_outputs = [graph.operations[-1]]

m = graph.compile_to_module()

stack.close()

print(m)


def get_args():
    inputs_file = "./inputs.pt"
    if os.path.exists(inputs_file):
        loaded_inputs = torch.load(inputs_file, map_location=torch.device("cuda", 0))
        print(f"Loaded input tensors from {inputs_file}")
        return loaded_inputs["arg0_1"], loaded_inputs["arg1_1"], loaded_inputs["arg2_1"]

    from torch._dynamo.testing import rand_strided
    arg0_1 = 2
    arg1_1 = rand_strided((4*arg0_1, 31, 64), (1984, 64, 1), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((4*arg0_1, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.float16)
    inputs = {"arg0_1": arg0_1, "arg1_1": arg1_1, "arg2_1": arg2_1}
    torch.save(inputs, inputs_file)
    print(f"Saved input tensors to {inputs_file}")
    return arg0_1, arg1_1, arg2_1

arg0_1, arg1_1, arg2_1 = get_args()

"""
# Incorrect epilogue fusion:
    # inductor generates a suffix
    xindex = idx_n + 64*idx_m + 1984*idx_q
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)

# Correct epilogue fusion:
    # inductor generates a suffix
    xindex = idx_n + 64*idx_m + 1984*idx_q

    # xindex = 7936*p0 + 64*p1 + 1984*p2 + p3
    p3 = xindex % 64
    p2 = (xindex // 1984) % 4
    p1 = (xindex // 64) % 31
    p0 = xindex // 7936
    index1 = 7936*p0 + 256*p1 + 64*p2 + p3
    tl.store(out_ptr1 + (tl.broadcast_to(index1, acc.shape)), acc, mask)
"""