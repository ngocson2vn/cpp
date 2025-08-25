set -x


pip3 install -r requirements.txt


CC=/usr/bin/clang \
CXX=/usr/bin/clang++ \
CMAKE_CUDA_ARCHITECTURES_DEFAULT="75;80;86;89;90" \
CMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
USE_CUDA=1 \
USE_CUDNN=1 \
CUDNN_INCLUDE_DIR=/usr/local/cuda-12.4/include \
CUDNN_LIBRARY=/usr/local/cuda-12.4/lib64 \
USE_GFLAGS=0 \
USE_GLOG=1 \
USE_NUMPY=1 \
USE_SYSTEM_NCCL=0 \
USE_NUMA=0 \
USE_NCCL=1 \
USE_TENSORRT=0 \
USE_MKL=0 \
USE_MKLDNN=0 \
USE_MPI=0 \
USE_EXCEPTION_PTR=1 \
TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
NCCL_ROOT_DIR=/usr/local/cuda-12.4 \
TH_BINARY_BUILD=1 \
USE_STATIC_CUDNN=0 \
USE_STATIC_NCCL=1 \
ATEN_STATIC_CUDA=0 \
USE_CUDA_STATIC_LINK=0 \
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" \
_GLIBCXX_USE_CXX11_ABI=1 \
CMAKE_CXX_FLAGS="-g -Wno-error=uninitialized -Wno-extra-semi -I/usr/local/cuda-12.4/include" \
CMAKE_C_FLAGS="-g -Wno-error=uninitialized -I/usr/local/cuda-12.4/include" \
CMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda-12.4/lib64 -lcusparseLt" \
LIBRARY_PATH=/usr/local/cuda-12.4/lib64 \
BUILD_TEST=0 \
PYTORCH_EXTRA_INSTALL_REQUIREMENTS=${PYTORCH_EXTRA_INSTALL_REQUIREMENTS} \
VERBOSE=1 \
pip install -e .

