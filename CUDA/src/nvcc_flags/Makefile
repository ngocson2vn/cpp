CUDA_VERSION := $(or $(CUDA_VERSION), "13.1")
$(info "CUDA_VERSION=$(CUDA_VERSION)")

CUDA_HOME := /usr/local/cuda-$(CUDA_VERSION)
NVCC := $(CUDA_HOME)/bin/nvcc

main: main.cu
	$(NVCC) --dryrun -o $@ main.cu > nvcc.sh 2>&1
	sed -i 's/#\$$/$$/g' nvcc.sh
	code nvcc.sh
