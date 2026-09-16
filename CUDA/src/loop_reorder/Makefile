NVCC := $(shell which nvcc)
$(info NVCC = $(NVCC))

main: main.cu
	$(NVCC) -gencode arch=compute_120a,code=sm_120a -o $@ main.cu
