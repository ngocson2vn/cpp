#!/bin/bash

/usr/local/cuda-12.4/bin/nsys profile -o toy_model_profile \
  --force-overwrite true --trace=cuda,nvtx,osrt \
  --sample process-tree \
  --capture-range=cudaProfilerApi \
  --backtrace lbr \
  --sampling-period 250000 \
  --cudabacktrace all \
  python3.11 ./test_inductor.py