NVCC := /usr/local/cuda-13.1/bin/nvcc
$(info $(shell $(NVCC) --version))

GENCODE_FLAGS := -gencode arch=compute_100a,code=sm_100a
LIB_DIRS := -L/lib/x86_64-linux-gnu

main: main.cu
	$(NVCC) $(GENCODE_FLAGS) $(LIB_DIRS) -G -o $@ main.cu -lcuda

main.ptx: main.cu
	$(NVCC) $(GENCODE_FLAGS) $(LIB_DIRS) -ptx -o $@ main.cu

test: test.cpp
	$(NVCC) -o $@ test.cpp

ptx: main.ptx

run: main
	./main

clean:
	rm -fv ./main *.ptx
