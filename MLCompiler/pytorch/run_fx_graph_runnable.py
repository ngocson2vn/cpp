import os
import importlib.util, torch

path = "/data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/debug_dir/torch_compile_debug/run_2025_11_02_00_55_13_258858-pid_1379719/torchinductor/model__1_inference_1.1/fx_graph_runnable.py"
base_dir = os.path.dirname(path)
spec = importlib.util.spec_from_file_location("runner", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument("--save-args", "-s", action="store_true")
# parser.add_argument("--load-args", "-l", action="store_true")
# args = parser.parse_args()
# save_args = args.save_args
# load_args = args.load_args

save_args = True
load_args = False

from torch._dynamo.repro.after_aot import run_repro

with torch.no_grad():
    if save_args:
        _mod, args = run_repro(
            m.mod, m.load_args, accuracy=False, command='get_args',
            save_dir=None, tracing_mode="symbolic", check_str=None
        )
        torch.save((args, {}), f"{base_dir}/inputs.pt")
        print("Saved inputs to inputs.pt")
    elif load_args:
        args, kwargs = torch.load(f"{base_dir}/inputs.pt", map_location="cuda:0")
        m.mod.eval()
        # Optional: compiled = torch.compile(m.mod)
        out = m.mod(*args, **kwargs)
        print("Ran replay with saved inputs.")
