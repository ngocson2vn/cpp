"""
Python 3.11.10
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/extras/CUPTI/lib64:$LD_LIBRARY_PATH
"""

import os
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.4/extras/CUPTI/lib64:" + os.environ["LD_LIBRARY_PATH"]

import torch
from torch import nn
import torch.fx as fx

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()

    def forward(self, x, y):
        res = torch.multiply(x, y)
        pred = torch.sum(res)
        return pred

# Instantiate the model
model = ToyModel()

# Symbolically trace the model to create a GraphModule
traced_model = fx.symbolic_trace(model)
print(f"type(traced_model): {type(traced_model)}")

# Print the GraphModule
print(f"type(traced_model.graph): {type(traced_model.graph)}")
print(traced_model.graph)

print()
print("======================================")
print("Convert GraphModule to TorchScript")
print("======================================")
scripted_model = torch.jit.script(traced_model)

print()
print(f"type(scripted_model): {type(scripted_model)}")                 # <class 'torch.jit._script.RecursiveScriptModule'>
print(f"type(scripted_model.graph): {type(scripted_model.graph)}")     # <class 'torch.Graph'>
print(scripted_model.graph)
print()

print()
print(f"type(scripted_model._c): {type(scripted_model._c)}")
print(scripted_model._c)
print()