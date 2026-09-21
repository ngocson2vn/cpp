NVCC := $(shell which nvcc)

main: main.cu
	$(NVCC) -gencode arch=compute_120a,code=sm_120a -o $@ main.cu

clean:
	@rm -rfv main *.ptx
