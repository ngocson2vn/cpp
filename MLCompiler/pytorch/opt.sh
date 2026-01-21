#!/bin/bash

set -e

# export TORCHDYNAMO_REPRO_LEVEL=4
# export TORCHDYNAMO_REPRO_AFTER="dynamo"
# export TORCHDYNAMO_REPRO_FORWARD_ONLY=1
# export TORCHDYNAMO_REPRO_IGNORE_GUARD_PRINT_FAILURE=1

export TORCH_LOGS="+dynamo"
export TORCHDYNAMO_VERBOSE="1"
export TORCH_COMPILE_DEBUG="1"
export TORCH_COMPILE_DEBUG_DIR="./scheduler_debug_mm"
export DYNAMO_CKPT_PATH="${TORCH_COMPILE_DEBUG_DIR}/aot"
export TORCHDYNAMO_COMPILE_DYNAMIC_SHAPE="1"
export TORCHINDUCTOR_CACHE_DIR="${TORCH_COMPILE_DEBUG_DIR}/aot/inductor"
export INDUCTOR_POST_FUSION_SVG="1"
export INDUCTOR_ORIG_FX_SVG="1"
export INDUCTOR_WRITE_SCHEDULER_GRAPH=1
export TORCHINDUCTOR_PROLOGUE_FUSION="1"
export TORCHINDUCTOR_DEBUG_FUSION="1"
export TORCHINDUCTOR_ENABLE_INFERENCE_FX_PASS=1

export TORCHINDUCTOR_MAX_AUTOTUNE="1"
# export ENABLE_PERSISTENT_TMA_MATMUL="1"
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER="0"
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="TRITON"
# export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="CUTLASS"

export EXPOSE_SERIALIZABLE_BACKEND_CALLABLE=1


export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:/usr/local/cuda-13.1/extras/CUPTI/lib64:/usr/lib/aarch64-linux-gnu/nvshmem/13
export LD_LIBRARY_PATH=/data03/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
# export PYTHONPATH=/data00/home/son.nguyen/workspace/triton_dev/bytedance/triton/python

# export TRITON_OVERRIDE_ARCH=sm86
# export TRITON_OVERRIDE_PTX_VERSION=74

# python3.11 test_inductor.py
# python3.11 test_inductor_mm.py
# python3.11 test_inductor_combo.py
export CLEAN_DEBUG_DIR=1
python3.11 test_inductor_mm_hopper.py
echo

find ${TORCH_COMPILE_DEBUG_DIR}/aot/ -maxdepth 1
find ${TORCH_COMPILE_DEBUG_DIR}/torch_compile_debug
