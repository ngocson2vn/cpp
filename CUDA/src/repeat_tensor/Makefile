NVCC := $(shell which nvcc)

main: main.cu
	$(NVCC) -gencode arch=compute_120a,code=sm_120a -o $@ main.cu

ptx: main.ptx
main.ptx: main.cu
	$(NVCC) -gencode arch=compute_120a,code=sm_120a -ptx -o $@ main.cu

clean:
	@rm -rfv main *.ptx
