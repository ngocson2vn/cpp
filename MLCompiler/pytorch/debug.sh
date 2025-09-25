#!/bin/bash

set -e

export TORCHINDUCTOR_MAX_AUTOTUNE="1"
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="TRITON"

export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/extras/CUPTI/lib64
export PYTHONPATH=/data00/home/son.nguyen/workspace/triton_dev/bytedance/triton/python



# python3.11 inductor_fusion_codegen1.py
python3.11 inductor_bmm_fusion.py
echo

find scheduler_debug/torch_compile_debug
