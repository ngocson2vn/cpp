#!/bin/bash

set -e

export TORCH_LOGS="+dynamo"
export TORCHDYNAMO_VERBOSE="1"
export TORCH_COMPILE_DEBUG="1"
export TORCH_COMPILE_DEBUG_DIR="./debug_dir"
export DYNAMO_CKPT_PATH="./debug_dir/aot"
export TORCHDYNAMO_COMPILE_DYNAMIC_SHAPE="1"
export TORCHINDUCTOR_CACHE_DIR="./debug_dir/aot/inductor"
export INDUCTOR_POST_FUSION_SVG="1"
export INDUCTOR_ORIG_FX_SVG="1"
export INDUCTOR_WRITE_SCHEDULER_GRAPH=1
export TORCHINDUCTOR_PROLOGUE_FUSION="1"
export TORCHINDUCTOR_DEBUG_FUSION="1"
export TORCHINDUCTOR_ENABLE_INFERENCE_FX_PASS=1

export TORCHINDUCTOR_MAX_AUTOTUNE="1"
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="TRITON"
# export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="CUTLASS"

export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/extras/CUPTI/lib64
export PYTHONPATH=/data00/home/son.nguyen/workspace/triton_dev/bytedance/triton/python

export TRITON_OVERRIDE_ARCH=sm86
export TRITON_OVERRIDE_PTX_VERSION=74

./clean.sh
echo

python3.11 test_inductor.py
echo

find debug/aot/ -maxdepth 1
find debug/torch_compile_debug
