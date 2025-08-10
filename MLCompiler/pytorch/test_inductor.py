import os

os.environ["TORCHDYNAMO_VERBOSE"] = "1"

# Enable IR dumping for Inductor
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCH_COMPILE_DEBUG_DIR"] = "./debug"
os.environ["INDUCTOR_POST_FUSION_SVG"] = "1"
os.environ["INDUCTOR_ORIG_FX_SVG"] = "1"
os.environ["TORCHINDUCTOR_PROLOGUE_FUSION"] = "1"
os.environ["TORCHINDUCTOR_DEBUG_FUSION"] = "1"

import torch
from torch import nn

torch.set_float32_matmul_precision('high')

class ToyModule(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x, y):
    z = x + y
    d0 = z.shape[0]
    d1 = z.shape[1]
    u = torch.reshape(z, (d1, d0))
    h = torch.matmul(x, u)
    logit = torch.sum(h, 1)
    res = torch.sigmoid(logit)
    return res

toy = ToyModule()
toy.forward = torch.compile(toy.forward, backend="inductor")
x = torch.rand(3, 4).cuda()
y = torch.rand(3, 4).cuda()
res = toy(x, y)
print(f"Result: {res}")
