#!/bin/bash

export SKIP_TRACE=1
export TORCHINDUCTOR_ENABLE_INFERENCE_FX_PASS=1

export PYTHONPATH=/data00/home/son.nguyen/workspace/triton_dev/bytedance/triton/python

python3.11 test_inductor.py
