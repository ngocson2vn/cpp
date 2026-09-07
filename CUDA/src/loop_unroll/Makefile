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

main: main1 main2
ptx: main1.ptx main2.ptx

main1: main1.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ main1.cu

main2: main2.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ main2.cu

main1.ptx: main1.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -ptx -o $@ main1.cu

main2.ptx: main2.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -ptx -o $@ main2.cu

run: main
	@echo
	./main1
	@echo
	./main2

clean:
	rm -fv main main1 main2 main3 *.o *.ptx
