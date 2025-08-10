#!/usr/bin/env python3.11

#==================================================================
# Set necessary env vars
#==================================================================
import os

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["RUN_STAGE"] = "runtime"
os.environ["COMPILATION"] = "serving"

os.environ["DYNAMO_CKPT_PATH"] = "./blade_debug/aot_compile_cache"
os.environ["TORCH_COMPILE_USE_LAZY_GRAPH_MODULE"] = "0"
os.environ["PYPILOT_DEPS"] = "/opt/tiger/pypilot"
os.environ["MODEL_NAME"] = "toy_model_v0"

# For debugging
os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCH_BLADE_DUMP_FXGRAPH"] = "1"

# For torch_blade
os.environ["TORCH_BLADE_DEBUG_LOG"] = "1"
os.environ["TORCH_DISC_DUMP_PREFIX"] = "./blade_debug/dump_dir"
#==================================================================

import sys

import torch
from torch import nn

# Register disc_infer backend
import torch_blade


compile_backend = "disc_infer"

class ToyModule(nn.Module):
    def __init__(self):
        super(ToyModule, self).__init__()
        self.forward = torch._dynamo.optimize(backend=compile_backend, dynamic=False)(self.forward)

    def forward(self, x, y):
        res = torch.multiply(x, y)
        self.relu = nn.ReLU()
        res = self.relu(res) / 2
        pred = torch.sum(res)
        return pred

toy = ToyModule()
x = torch.rand(3, 4)
y = torch.rand(3, 4)
pred = toy(x, y)
print(f"prediction: {pred}")
