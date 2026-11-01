NVCC := /usr/local/cuda-13.1/bin/nvcc
GENCODE_FLAGS := -gencode arch=compute_100a,code=sm_100a

CUDA_SRC_DIR := $(dir $(CURDIR))
TORCH_DIR := $(shell python3.11 -c "import torch; print(torch.__path__[0])")
INCLUDES := -I$(CUDA_SRC_DIR) -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include

default: libcommon.a

common.o: common.cu
	$(NVCC) $(GENCODE_FLAGS) $(INCLUDES) -c -o $@ common.cu

libcommon.a: common.o
	ar rcs $@ common.o

clean:
	rm -fv *.o *.a
