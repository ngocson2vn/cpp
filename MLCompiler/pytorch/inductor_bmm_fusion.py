import os
import shutil

debug_dir = "./scheduler_debug"

if os.path.exists(debug_dir):
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

os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "TRITON"
# """

import torch
from torch import nn

import logging
logging.basicConfig(level=logging.DEBUG)

torch.set_float32_matmul_precision('high')
logging.getLogger("torch._inductor.scheduler").setLevel(logging.DEBUG)
# import torch._inductor.scheduler
# torch._inductor.scheduler.fusion_log.setLevel(logging.DEBUG)

class ToyModule(nn.Module):
  def __init__(self):
    super().__init__()
    self.forward = torch.compile(self.forward, backend="inductor")

  # Epilogue fusion pattern
  def forward(self, x: torch.Tensor, y: torch.Tensor):
    tmp1 = torch.bmm(x, y)
    res = torch.sigmoid(tmp1)
    # res = torch.sum(tmp1, dim=0)
    return res

toy = ToyModule()

if __name__ == "__main__":
  x1 = torch.rand(3, 3, 9).cuda()
  y1 = torch.rand(3, 9, 4).cuda()
  res1 = toy(x1, y1)
  print(f"Result: {res1}")

  x2 = torch.rand(32, 3, 9).cuda()
  y2 = torch.rand(32, 9, 4).cuda()
  res2 = toy(x2, y2)
  print(f"Result: {res2}")
