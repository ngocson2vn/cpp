#!/bin/bash

export PERF_MODE=true
export SKIP_TRACE=true
export PYTHONPATH=/data00/home/son.nguyen/workspace/triton_dev/bytedance/triton/python

/usr/local/cuda-12.4/bin/nsys profile -o toy_model_profile \
  --force-overwrite true --trace=cuda,nvtx,osrt \
  --capture-range-end=stop \
  --sample process-tree \
  --capture-range=cudaProfilerApi \
  --backtrace lbr \
  --sampling-period 250000 \
  --cudabacktrace all \
  python3.11 ./test_inductor.py