import sys
sys.path.insert(0, "/data00/home/son.nguyen/workspace/triton_dev/bytedance/triton/python")

import os
os.environ["TORCH_LOGS"] = "+dynamo,guards,graph_breaks"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL"] = "1"

import logging, torch
from torch import nn

logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
for name in ["torch._dynamo.convert_frame",
             "torch._dynamo.eval_frame"]:
    logging.getLogger(name).setLevel(logging.WARNING)

logging.getLogger("torch._dynamo.symbolic_shapes").setLevel(logging.DEBUG)
logging.getLogger("torch._dynamo.shape_env").setLevel(logging.DEBUG)
logging.getLogger("torch._dynamo.guards").setLevel(logging.DEBUG)
logging.getLogger("torch._dynamo.graph_breaks").setLevel(logging.DEBUG)

class Context:
    _data = {}

    def __setattr__(self, name, value):
        Context._data[name] = value

    def __getattr__(self, name):
        if name in Context._data:
            return Context._data[name]
        return None
    
C = Context()

class CandSize(nn.Module):
    def forward(self):
        return C.group_candidate_size

@torch.compile
def f(x, y):
    # Intentional graph break (Python-side operation)
    repeats = CandSize()
    n = int(repeats().item())
    x = x.repeat_interleave(n, dim=0)
    y = y.transpose(-1, -2)
    res = torch.matmul(x, y)
    return res


C.group_candidate_size = torch.tensor(5, dtype=torch.int32)
x = torch.rand([1, 3], dtype=torch.float32)
y = torch.rand([3, 5], dtype=torch.float32)

out = f(x, y)
print(out)
# Expect: guard or graph-break messages (WARNING), but no trace/DEBUG spam.
