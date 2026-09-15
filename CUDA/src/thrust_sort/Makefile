CUDA_VERSION := $(or $(CUDA_VERSION), 13.1)
$(info CUDA_VERSION: $(CUDA_VERSION))

NVCC := /usr/local/cuda-$(CUDA_VERSION)/bin/nvcc
NVCC_FLAGS := -gencode arch=compute_120a,code=sm_120a

default: main

ptx: main.ptx

main: main.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ main.cu

main.ptx: main.cu
	$(NVCC) $(NVCC_FLAGS) -ptx -o $@ main.cu

clean:
	rm -rf main main.ptx
