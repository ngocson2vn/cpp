NVCC := /usr/local/cuda-13.1/bin/nvcc
GENCODE_FLAGS = -gencode arch=compute_100a,code=sm_100a

main: main.cu
	$(NVCC) $(GENCODE_FLAGS) -o $@ main.cu

main.ptx: main.cu
	$(NVCC) $(GENCODE_FLAGS) -ptx -o $@ main.cu

ptx: main.ptx

test: test.cu
	$(NVCC) $(GENCODE_FLAGS) -o $@ test.cu
