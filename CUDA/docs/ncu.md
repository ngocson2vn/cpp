# Nsight Compute

## Start ncu
worker.py:
```Python
import torch

start_profiler = False
if not start_profiler
  print("=== Start cuda profiler ===")
  torch.cuda.cudart().cudaProfilerStart()

torch.cuda.nvtx.range_push(f"sony_inference")
outputs = model(inputs)
torch.cuda.nvtx.range_pop()

torch.cuda.cudart().cudaProfilerStop()
print("=== Stop cuda profiler ===")
sys.exit(0)
```

Startup script:
```Python
script_path = "/path/to/worker.py"

# ncu
begin_cmd = [
  "/usr/local/cuda-13.3/bin/ncu",
  "--nvtx",
  "--profile-from-start",
  "off",
  "--metrics",
  "launch__registers_per_thread,sm__maximum_warps_per_active_cycle_pct,sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_avg_per_active_cycle",
  "-f",
  "-o",
  "/data02/home/son.nguyen/workspace/torch_dev/pypilot_gpu/manhattan_worker_profile",
]

cmd = begin_cmd + [
  "python3",
  script_path
]

stdout_file = open(stdout_filepath, "a")
stderr_file = open(stderr_filepath, "a")
process = subprocess.Popen(
    cmd,
    env=env,
    stdout=stdout_file,
    stderr=stderr_file,
    start_new_session=True,
)
```


## Some useful metrics
| NCU Metric Name | What It Tells You |
| --- | --- |
| `launch__registers_per_thread` | The exact number of registers allocated to a single thread by the compiler. If this is near 255, register pressure is severe. |
| `sm__maximum_warps_per_active_cycle_pct` | The "Theoretical Occupancy." If this is low (e.g., 12%), it means block size, shared memory, or registers are acting as a hard limit. |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | The "Achieved Occupancy." This shows the actual average percentage of warps active during the kernel's execution. |
| `sm__maximum_warps_avg_per_active_cycle` | The raw number of warps the SM was able to keep resident based on the limiting factor (registers vs. shared memory). |
