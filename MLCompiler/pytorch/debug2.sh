#!/bin/bash

set -e

export TORCHINDUCTOR_MAX_AUTOTUNE="1"
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="TRITON"

export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/extras/CUPTI/lib64
export PYTHONPATH=/data00/home/son.nguyen/workspace/triton_dev/bytedance/triton/python


# For debugging how Triton bmm template works
# python3.11 inductor_bmm_fusion.py

# Using extern_kernels.bmm
python3.11 inductor_fusion_codegen2.py
echo

ir_post_fusion=$(find scheduler_debug2/aot | grep ir_post_fusion.txt)
dbg_dir=$(dirname ${ir_post_fusion})
base_dir=$(dirname ${dbg_dir})
find ${base_dir}
