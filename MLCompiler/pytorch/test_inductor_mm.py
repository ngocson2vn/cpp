import os
import shutil

debug_dir = "./debug_dir"

if os.getenv("VSCODE_MODE", "false") == "true" and os.path.exists(debug_dir):
  shutil.rmtree(debug_dir)
  print(f"Cleaned {debug_dir}")


os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

# Enable IR dumping for Inductor
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCH_COMPILE_DEBUG_DIR"] = debug_dir
os.environ["DYNAMO_CKPT_PATH"] = f"{debug_dir}/aot"
os.environ['TORCHINDUCTOR_CACHE_DIR'] = f"{debug_dir}/aot/inductor"
os.environ["INDUCTOR_POST_FUSION_SVG"] = "1"
os.environ["INDUCTOR_ORIG_FX_SVG"] = "1"
os.environ["TORCHINDUCTOR_PROLOGUE_FUSION"] = "0"
os.environ["TORCHINDUCTOR_DEBUG_FUSION"] = "1"
os.environ["TORCHDYNAMO_COMPILE_DYNAMIC_SHAPE"] = "1"

# os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "ATEN"
os.environ["EXPOSE_SERIALIZABLE_BACKEND_CALLABLE"] = "1"


import torch
from torch import nn
torch._inductor.config.epilogue_fusion = False

import logging
logging.basicConfig(level=logging.DEBUG)

torch.set_float32_matmul_precision('high')
logging.getLogger("torch._inductor.scheduler").setLevel(logging.DEBUG)

class ToyModule(nn.Module):
  def __init__(self):
    super().__init__()
    self.forward = torch.compile(self.forward, backend="inductor")

  # Epilogue fusion pattern
  def forward(self, x: torch.Tensor, y: torch.Tensor):
    tmp1 = torch.matmul(x, y)
    res = torch.sigmoid(tmp1)
    return res

toy = ToyModule()

if __name__ == "__main__":
  x1 = (torch.rand(8, 256).cuda() - 0.5)
  y1 = (torch.rand(256, 256).cuda() - 0.5).to(torch.float64)
  res1 = toy(x1, y1)
  print(f"x1: {x1}")
  print(f"y1: {y1}")
  print(f"Result: {res1}\n")

  x2 = (torch.rand(16, 256).cuda() - 0.5)
  y2 = (torch.rand(256, 256).cuda() - 0.5).to(torch.float64)
  res2 = toy(x2, y2)
  print(f"x2: {x2}")
  print(f"y2: {y2}")
  print(f"Result: {res2}")
