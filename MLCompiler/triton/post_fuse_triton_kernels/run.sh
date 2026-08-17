#!/bin/bash

#========================================================================
# Prerequisites
#========================================================================
## Install cudnn
# pip3.11 install nvidia-cudnn-cu12
# 
## Install libomp
# sudo apt install -y libomp-15-dev
# sudo ln -sf /usr/lib/llvm-15/lib/libomp.so.5 /lib/x86_64-linux-gnu/libomp.so
# ls -l /lib/x86_64-linux-gnu/libomp.so
#
## PyTorch
# A custom PyTorch version
# 
## Triton
# Version 3.3.1
#========================================================================

export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:$LD_LIBRARY_PATH

export CUDNN_LIB_DIR=${HOME}/.pyenv/versions/3.11.2/lib/python3.11/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH=${CUDNN_LIB_DIR}:$LD_LIBRARY_PATH

export TORCHINDUCTOR_CACHE_DIR="$(pwd)/inductor"
mkdir -p ${TORCHINDUCTOR_CACHE_DIR}

python3.11 run_model.py
