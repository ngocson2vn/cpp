CUDA_VERSION := $(or $(CUDA_VERSION), 13.1)
$(info CUDA_VERSION = $(CUDA_VERSION))

NVCC := /usr/local/cuda-$(CUDA_VERSION)/bin/nvcc

main: main.cu
	$(NVCC) -gencode arch=compute_120a,code=sm_120a -o $@ main.cu

clean:
	@rm -rfv main *.ptx
