import sys
import os
base_dir = os.path.basename(__file__)
sys.path.insert(0, base_dir)

import argparse
import torch
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("--module-name", "-m", required=True, help="The module name to be imported")
args = parser.parse_args()
module_name = args.module_name

try:
  mod = importlib.import_module(module_name)
except:
  raise

x = torch.rand(1024, 9).cuda()
y = torch.rand(1024, 9).cuda()
is_perf_mode = os.environ.get("PERF_MODE") == "true"
if not is_perf_mode:
  res = mod.toy(x, y)
  print(f"Result:\n{res}")
else:
  torch.cuda.profiler.start() # start profiling
  for i in range(50):
    if i < 3:
      torch.cuda.nvtx.range_push(f"warmup{i}")
    else:
      torch.cuda.nvtx.range_push(f"infer{i-3}")
    res = mod.toy(x, y)
    torch.cuda.nvtx.range_pop()
  torch.cuda.profiler.stop() # end profiling
  print(f"Result: {res}")
