NVCC = /usr/local/cuda-13.1/bin/nvcc

main: main.cu
	$(NVCC) -gencode arch=compute_120a,code=sm_120a -o $@ main.cu
