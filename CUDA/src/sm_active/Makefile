CUDA_VERSION := $(or $(CUDA_VERSION), 13.1)
NVCC = /usr/local/cuda-$(CUDA_VERSION)/bin/nvcc
INCLUDE_DIRS = -I/usr/local/cuda-$(CUDA_VERSION)/include
LIB_DIRS = -L/usr/local/cuda-$(CUDA_VERSION)/lib64

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

main: main.cu gpu_monitor.cc
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) -o main main.cu gpu_monitor.cc -lcudart -lnvidia-ml
	patchelf --add-rpath ./lib main

install:
	mkdir -p ./lib
	cp -v /usr/local/cuda-$(CUDA_VERSION)/lib64/libcudart.so.13 ./lib/
	cp -v /lib/x86_64-linux-gnu/libnvidia-ml.so.1 ./lib/

run: main
	./main verbose

clean:
	rm -rfv *.o main
