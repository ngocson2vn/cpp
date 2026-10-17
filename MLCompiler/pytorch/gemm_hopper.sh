#!/bin/bash

# Cache dir
export TRITON_CACHE_DIR=./tmp

# Autotune
export TRITON_PRINT_AUTOTUNING="1"

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:$LD_LIBRARY_PATH

export CUDNN_LIB_DIR=${HOME}/.pyenv/versions/3.11.2/lib/python3.11/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH=${CUDNN_LIB_DIR}:$LD_LIBRARY_PATH

python3.11 gemm_hopper.py
