#!/bin/bash

export TORCHINDUCTOR_ENABLE_INFERENCE_FX_PASS=1

export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/extras/CUPTI/lib64
export PYTHONPATH=/data00/home/son.nguyen/workspace/triton_dev/bytedance/triton/python

python3.11 test_inductor.py
