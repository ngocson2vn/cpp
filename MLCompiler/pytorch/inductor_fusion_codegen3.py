import sys
import os
import shutil

debug_dir = "./scheduler_debug3"

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
  NonOwningLayout,
  TensorBox,
  StorageBox,
  Pointwise,
  InputBuffer,
  ConstantBuffer,
  ExternKernelOut,
  ComputedBuffer,
  ConcatKernel,
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
op1513: ExternKernelSchedulerNode(ExternKernelOut)
op1513.writes = [StarDep(name='buf1513', mode=None)]
op1513.unmet_dependencies = [StarDep(name='buf232', mode=None)]
op1513.met_dependencies = [StarDep(name='arg766_1', mode=None)]
op1513.outputs = [
    buf1513: ExternKernelOut
    buf1513.layout = FixedLayout('cuda:0', torch.float16, size=[50, 256], stride=[256, 1])
    buf1513.users = [NodeUser(node=SchedulerNode(name='op1519'), can_inplace=False, is_weak=False)]
]
op1513.node.kernel = extern_kernels.mm
"""

s0 = sympy.Symbol("s0", positive=True, is_number = True)
graph.graph_inputs['arg0_1'] = s0

V.graph.sizevars.shape_env.size_like.add(s0)
V.graph.sizevars.shape_env.var_to_range[s0] = ValueRanges(lower=8, upper=512)
V.graph.sizevars.shape_env.var_to_val[s0] = 256


input0 = ConstantBuffer(name="arg1_1", layout=FixedLayout(DEVICE, torch.float16, size=[s0*50, 64], stride=[64, 1]))
graph.graph_inputs["arg1_1"] = TensorBox(StorageBox(input0))
graph.constants["arg1_1"] = torch.rand(size=[50, 64], dtype=torch.float16)

input1 = ConstantBuffer(name="arg2_1", layout=FixedLayout(DEVICE, torch.float16, size=[64, 256], stride=[256, 1]))
graph.graph_inputs["arg2_1"] = TensorBox(StorageBox(input1))
graph.constants["arg2_1"] = torch.rand(size=[64, 256], dtype=torch.float16)

input2 = ConstantBuffer(name="arg3_1", layout=FixedLayout(DEVICE, torch.float16, size=[s0, 64, 256], stride=[12800, 256, 1]))
graph.graph_inputs["arg3_1"] = TensorBox(StorageBox(input1))
graph.constants["arg3_1"] = torch.rand(size=[10, 50, 256], dtype=torch.float16)

V.graph.graph_input_names.extend(graph.graph_inputs.keys())

# layout0 = FixedLayout(DEVICE, torch.float16, size=[50, 256], stride=[256, 1])
# buf0 = ExternKernelOut(layout0, [input0, input1], python_kernel_name="extern_kernels.mm")
# x = TensorBox(StorageBox(buf0))


"""
op1519: SchedulerNode(ComputedBuffer)
op1519.writes = [MemoryDep('buf1519', c0, {c0: 12800})]
op1519.unmet_dependencies = [MemoryDep('buf1513', c0, {c0: 12800})]
op1519.met_dependencies = []
op1519.outputs = [
    buf1519: ComputedBuffer
    buf1519.layout = NonOwningLayout('cuda:0', torch.float16, size=[1, 50, 256], stride=[80640, 256, 1])
    buf1519.aliases = ['buf1528']
    buf1519.users = []
]
op1519.group.device = cuda:0
op1519.group.iteration = (12800, 1)
op1519.sizes = ([12800], [])
buf1513_layout = FixedLayout('cuda:0', torch.float16, size=[50, 256], stride=[256, 1])
buf1519_layout = NonOwningLayout('cuda:0', torch.float16, size=[1, 50, 256], stride=[80640, 256, 1])
class op1519_loop_body:
    var_ranges = {p0: 12800}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf1513', get_index)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf1519', get_index_1, load, None)
        return store
"""

target_mm = aten.mm
tmp0 = lowerings[target_mm](input0, input1)

target_sigmoid = aten.sigmoid
tmp2 = lowerings[target_sigmoid](tmp0)
size2 = tmp2.get_size()
target_reshape = aten.reshape
tmp3 = lowerings[target_reshape](tmp2, [s0, size2[0]/s0, size2[1]])

tmp4 = lowerings[target_sigmoid](input2)


# target3 = aten.cat
# tmp4 = lowerings[target3]([tmp1, tmp3], dim=0)

class NodeArg:
    def __init__(self, name):
        self.name = name
        self.meta = {}

current_node = torch.fx.Node(
    graph=graph,
    name="cat1",
    op="call_function",
    target=aten.cat,
    args=([NodeArg("fake_arg_1"), NodeArg("fake_arg_2")], 0),
    kwargs={}
)
V.graph.current_node = current_node
tmp4 = ConcatKernel.create([tmp3, tmp4], dim=1)
tmp4.realize()

graph.graph_outputs = [graph.operations[-1]]

m = graph.compile_to_module()

stack.close()

print(m)


"""
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

