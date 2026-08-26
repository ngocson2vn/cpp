CUDA_VERSION := $(or $(CUDA_VERSION), 13.1)
NVCC = /usr/local/cuda-$(CUDA_VERSION)/bin/nvcc
CUDA_INCLUDE = /usr/local/cuda-$(CUDA_VERSION)/include


NVCC_FLAGS = -std=c++17 -g -O3
NVCC_FLAGS += --compiler-options=-fno-inline --expt-relaxed-constexpr
NVCC_FLAGS += --disable-warnings
cuda_version := $(shell /usr/local/cuda-$(CUDA_VERSION)/bin/nvcc --version | grep cuda)
$(info NVCC: $(cuda_version))
cuda12_str := $(shell /usr/local/cuda-$(CUDA_VERSION)/bin/nvcc --version | grep cuda_12)
ifneq ("$(cuda12_str)", "")
	NVCC_GENCODE_FLAGS += -gencode arch=compute_75,code=sm_75
	NVCC_GENCODE_FLAGS += -gencode arch=compute_86,code=sm_86
	NVCC_GENCODE_FLAGS += -gencode arch=compute_89,code=sm_89
else
	NVCC_GENCODE_FLAGS += -gencode arch=compute_90a,code=sm_90a
	NVCC_GENCODE_FLAGS += -gencode arch=compute_100a,code=sm_100a
	NVCC_GENCODE_FLAGS += -gencode arch=compute_120a,code=sm_120a
endif
NVCC_FLAGS += $(NVCC_GENCODE_FLAGS)

main: main.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -o main main.cu

run: main
	./main verbose

dist:
	rm -rf device_query
	mkdir -p device_query/lib/
	cp main device_query/
	patchelf --add-rpath ./lib device_query/main
	patchelf --set-interpreter ./lib/ld-linux-x86-64.so.2 device_query/main
	cp /lib/x86_64-linux-gnu/libstdc++.so.6 device_query/lib/
	cp /lib/x86_64-linux-gnu/libgcc_s.so.1 device_query/lib/
	cp /lib/x86_64-linux-gnu/libc.so.6 device_query/lib/
	cp /lib/x86_64-linux-gnu/libm.so.6 device_query/lib/
	cp /lib64/ld-linux-x86-64.so.2 device_query/lib/
	tar -cvf device_query.tar device_query
	echo
	ls -l device_query.tar

clean:
	rm -rfv *.o main device_query.tar device_query/
