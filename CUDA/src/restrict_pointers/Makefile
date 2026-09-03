CUDA_VERSION := $(or $(CUDA_VERSION), 13.1)
$(info CUDA_VERSION=$(CUDA_VERSION))

NVCC := /usr/local/cuda-$(CUDA_VERSION)/bin/nvcc
INCLUDES := -I /usr/local/cuda-$(CUDA_VERSION)/include

cuda12 := $(shell echo $(CUDA_VERSION) | grep "12")
ifneq ("$(cuda12)", "")
	NVCC_GENCODE_FLAGS = -gencode arch=compute_90a,code=sm_90a
else
	NVCC_GENCODE_FLAGS = -gencode arch=compute_120a,code=sm_120a
endif

NVCC_FLAGS := $(NVCC_GENCODE_FLAGS)

v1: main.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -DKERNEL_VERSION=1 -o main1 main.cu

v2: main.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -DKERNEL_VERSION=2 -o main2 main.cu

ptx1: main.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -DKERNEL_VERSION=1 -ptx -o main1.ptx main.cu

ptx2: main.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -DKERNEL_VERSION=2 -ptx -o main2.ptx main.cu

clean:
	rm -fv main main1 main2 *.o *.ptx
