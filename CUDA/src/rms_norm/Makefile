NVCC := /usr/local/cuda-13.1/bin/nvcc
GENCODE_FLAGS := -gencode arch=compute_100a,code=sm_100a
CUDA_INCLUDE_DIR := /usr/local/cuda-13.1/include
CUDA_CCCL_DIR := /usr/local/cuda-13.1/include/cccl

CXX := /usr/bin/clang++

CUDA_SRC_DIR := $(shell dirname $(CURDIR))
TORCH_DIR := $(shell python3.11 -c "import torch; print(torch.__path__[0])")
INCLUDES := -I$(CUDA_INCLUDE_DIR) -I$(CUDA_CCCL_DIR) -I$(CUDA_SRC_DIR) -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include
LIBS := -L$(TORCH_DIR)/lib -L$(CUDA_SRC_DIR)/common
$(info PyTorch lib dir: $(TORCH_DIR)/lib)

default: main torch

main: main.cu
	$(NVCC) $(GENCODE_FLAGS) $(INCLUDES) $(LIBS) -o $@ main.cu -lcommon
	/usr/bin/patchelf --set-rpath $(TORCH_DIR)/lib main

main.ptx: main.cu
	$(NVCC) $(GENCODE_FLAGS) $(INCLUDES) $(LIBS) -ptx -o $@ main.cu

torch: torch.cu
	$(NVCC) $(GENCODE_FLAGS) $(INCLUDES) $(LIBS) -o $@ torch.cu -lcommon -ltorch -ltorch_cpu -ltorch_cuda -lc10
	/usr/bin/patchelf --set-rpath $(TORCH_DIR)/lib torch

ptx: main.ptx

run:
	./run.sh

clean:
	rm -fv main torch *.ptx
