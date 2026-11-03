NVCC := /usr/local/cuda-13.1/bin/nvcc
GENCODE_FLAGS := -gencode arch=compute_100a,code=sm_100a

CUDA_SRC_DIR := $(dir $(CURDIR))
TORCH_DIR := $(shell python3.11 -c "import torch; print(torch.__path__[0])")
INCLUDES := -I$(CUDA_SRC_DIR) -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include

default: libtorch_api.a

torch_api.o: torch_api.cu
	$(NVCC) $(GENCODE_FLAGS) $(INCLUDES) -c -o $@ torch_api.cu

libtorch_api.a: torch_api.o
	ar rcs $@ torch_api.o

clean:
	rm -fv *.o *.a
