NVCC := /usr/local/cuda-13.1/bin/nvcc
GENCODE_FLAGS := -gencode arch=compute_100a,code=sm_100a
CUDA_INCLUDE_DIR := /usr/local/cuda-13.1/include
CUDA_CCCL_DIR := /usr/local/cuda-13.1/include/cccl

CXX := /usr/bin/clang++

CUDA_SRC_DIR := $(shell dirname $(CURDIR))
TORCH_DIR := $(shell python3.11 -c "import torch; print(torch.__path__[0])")
INCLUDES := -I$(CUDA_INCLUDE_DIR) -I$(CUDA_CCCL_DIR) -I$(CUDA_SRC_DIR) -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include
LIBS := -L$(TORCH_DIR)/lib -L$(CUDA_SRC_DIR)/common -L$(CUDA_SRC_DIR)/torch_api
$(info PyTorch lib dir: $(TORCH_DIR)/lib)

default: main

main: main.cu *.h *.cuh
	$(NVCC) $(GENCODE_FLAGS) $(INCLUDES) $(LIBS) -o $@ main.cu -lcommon -ltorch_api -ltorch -ltorch_cpu -ltorch_cuda -lc10
	/usr/bin/patchelf --set-rpath $(TORCH_DIR)/lib main

main.ptx: main.cu *.h *.cuh
	$(NVCC) $(GENCODE_FLAGS) $(INCLUDES) $(LIBS) -ptx -o $@ main.cu

torch: torch.cu *.h *.cuh
	$(NVCC) $(GENCODE_FLAGS) $(INCLUDES) $(LIBS) -o $@ torch.cu -lcommon -ltorch_api -ltorch -ltorch_cpu -ltorch_cuda -lc10
	/usr/bin/patchelf --set-rpath $(TORCH_DIR)/lib torch

test: test.cu *.h *.cuh
	$(NVCC) $(GENCODE_FLAGS) -o $@ test.cu

ptx: main.ptx

run:
	./run.sh

clean:
	rm -fv main torch test *.ptx *.o
