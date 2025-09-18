import os
import shutil

debug_dir = "./debug"

if os.getenv("VSCODE_MODE", "false") == "true" and os.path.exists(debug_dir):
  shutil.rmtree(debug_dir)
  print(f"Cleaned {debug_dir}")

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
# """

import logging
logging.basicConfig(level=logging.DEBUG)

import torch
from torch import nn

from custom_pass import CustomFusion
torch._inductor.config._pre_fusion_custom_pass = CustomFusion.fuse

import ipdb

torch.set_float32_matmul_precision('high')

class ToyModule(nn.Module):
  def __init__(self):
    super().__init__()
    self.forward = torch.compile(self.forward, backend="inductor")

  # def forward(self, x, y):
  #   z = x + y
  #   d0 = z.shape[0]
  #   d1 = z.shape[1]
  #   u = torch.reshape(z, (d1, d0))
  #   h = torch.matmul(x, u)
  #   logit = torch.sum(h, 1)
  #   res = torch.sigmoid(logit)
  #   return res

  def forward(self, x, y):
    tmp0 = torch.add(x, y)
    tmp1 = torch.sum(tmp0, 1)

    tmp2 = torch.sigmoid(tmp1)
    tmp3 = torch.sum(tmp2, 0)

    tmp4 = torch.sigmoid(tmp3)
    return tmp4

  # def forward(self, x: torch.Tensor, y: torch.Tensor):
  #   tmp0 = y.transpose(-2, -1)
  #   tmp1 = torch.matmul(x, tmp0)
  #   res = torch.sigmoid(tmp1)
  #   return res

toy = ToyModule()

if __name__ == "__main__":
  x1 = torch.rand(9, 9).cuda()
  y1 = torch.rand(9, 9).cuda()
  res1 = toy(x1, y1)
  print(f"Result: {res1}")

  x2 = torch.rand(128, 9).cuda()
  y2 = torch.rand(128, 9).cuda()
  res2 = toy(x2, y2)
  print(f"Result: {res2}")
