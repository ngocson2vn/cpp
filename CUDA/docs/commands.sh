# Get Compute Capability
nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader

# ==============================================
# For profiling
# ==============================================
# Read current value
sudo sysctl kernel.perf_event_paranoid

# Set a value
sudo sysctl -w kernel.perf_event_paranoid=2

# ncu
/usr/local/cuda-13.1/bin/ncu --set full --nvtx --nvtx-include "sony_inference/" -o manhattan_worker_profile python3 script.py