#!/bin/bash

set -e

ROOT_DIR=$(pwd)
echo "ROOT_DIR=${ROOT_DIR}"
mkdir -p ${ROOT_DIR}/build

echo
echo "==================================================="
echo "Generate ninja build file"
echo "==================================================="
cd ${ROOT_DIR}/build

# -DCMAKE_BUILD_TYPE=Debug | Release \

cmake -G Ninja .. \
  -DCMAKE_PREFIX_PATH="$(python3.11 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCUDA_ROOT=/usr/local/cuda-12.4

cmake --build . -v

echo
echo "==================================================="
echo "Run ninja build"
echo "==================================================="
cd ${ROOT_DIR}/build
cmake --build . -v
