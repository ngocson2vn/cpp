#!/bin/bash

export TORCH_LOGS="+dynamo,inductor,cache"
export TORCHDYNAMO_VERBOSE="1"

# export SKIP_TRACE=1
# export DYNAMO_CACHE_START_INDEX=1

# export PYTHONPATH=/data00/home/son.nguyen/workspace/triton_dev/bytedance/triton/python

export TORCH_COMPILE_DEBUG_DIR="./scheduler_debug_mm"
export DYNAMO_CKPT_PATH="${TORCH_COMPILE_DEBUG_DIR}/aot"
export TORCHDYNAMO_COMPILE_DYNAMIC_SHAPE="1"
export TORCHINDUCTOR_CACHE_DIR="${TORCH_COMPILE_DEBUG_DIR}/aot/inductor"

export CLEAN_DEBUG_DIR=0
python3.11 model_runner.py -m test_inductor_mm_hopper
