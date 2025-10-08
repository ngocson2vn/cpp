#!/bin/bash


export TORCHINDUCTOR_MAX_AUTOTUNE="1"
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="TRITON"

export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/extras/CUPTI/lib64
export PYTHONPATH=/data00/home/son.nguyen/workspace/triton_dev/bytedance/triton/python


python3.11 inductor_fusion_codegen3.py
echo

ir_post_fusion=$(find scheduler_debug3/aot | grep ir_post_fusion.txt)
dbg_dir=$(dirname ${ir_post_fusion})
base_dir=$(dirname ${dbg_dir})
find ${base_dir}
