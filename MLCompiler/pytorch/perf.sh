#!/bin/bash

export PERF_MODE=true
export SKIP_TRACE=1

# Load compiled code that can handle dynamic shape
export DYNAMO_CACHE_START_INDEX=1

export PYTHONPATH=/data00/home/son.nguyen/workspace/triton_dev/bytedance/triton/python

rm -fv toy_model_profile.nsys-rep

/usr/local/cuda-12.4/bin/nsys profile -o toy_model_profile \
  --force-overwrite true --trace=cuda,nvtx,osrt \
  --capture-range-end=stop \
  --sample process-tree \
  --capture-range=cudaProfilerApi \
  --backtrace lbr \
  --sampling-period 250000 \
  --cudabacktrace all \
  python3.11 model_runner.py -m test_inductor

find toy_model_profile.nsys-rep