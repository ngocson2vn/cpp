CUDA_VERSION := $(or $(CUDA_VERSION), 12.8)
NVCC = /usr/local/cuda-$(CUDA_VERSION)/bin/nvcc
CUDA_INCLUDE = /usr/local/cuda-$(CUDA_VERSION)/include


NVCC_FLAGS = -std=c++17 -g -O3
NVCC_FLAGS += --compiler-options=-fno-inline --expt-relaxed-constexpr
NVCC_FLAGS += --disable-warnings
NVCC_FLAGS += -gencode arch=compute_75,code=sm_75
NVCC_FLAGS += -gencode arch=compute_120a,code=sm_120a
cuda_version := $(shell /usr/local/cuda-$(CUDA_VERSION)/bin/nvcc --version | grep cuda)
$(info NVCC: $(cuda_version))
cuda12_str := $(shell /usr/local/cuda-$(CUDA_VERSION)/bin/nvcc --version | grep cuda_12)
ifneq ("$(cuda12_str)", "")
	NVCC_FLAGS += -gencode arch=compute_90a,code=sm_90a
endif

main: main.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -o main main.cu

run: main
	./main verbose

clean:
	rm -rfv *.o main
