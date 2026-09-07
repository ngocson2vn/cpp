# Load Kernel Parameters
Given the following CUDA kernel in C++:
```cpp
extern "C" {

__global__ void add_vectors(const float *a, const float *b, float *c,
                            int totalElems, int maxRound) {
  auto numThreads = blockDim.x;
  auto numElems = (totalElems + numThreads - 1) / numThreads;
  auto tid = threadIdx.x;

  // numElems = 4
  //  tid=0  |  tid=1  |  tid=2
  // 0 1 2 3 | 4 5 6 7 | 8 9 10 11

  // Ensure that `maxIdx` is not greater than `totalElems`
  auto maxIdx = min((tid + 1) * numElems, totalElems);

  for (int i = 0; i < maxRound; i++) {
    for (int idx = tid * numElems; idx < maxIdx; idx++) {
      c[idx] = a[idx] + b[idx];
    }
  }
}

}
```

`nvcc` generates the following PTX code:
```mlir
.visible .entry add_vectors(
	.param .u64 .ptr .align 1 add_vectors_param_0,
	.param .u64 .ptr .align 1 add_vectors_param_1,
	.param .u64 .ptr .align 1 add_vectors_param_2,
	.param .u32 add_vectors_param_3,
	.param .u32 add_vectors_param_4
)
{
	.reg .pred 	%p<9>;
	.reg .b32 	%r<51>;
	.reg .b64 	%rd<18>;

	ld.param.b64 	%rd7, [add_vectors_param_0];
	ld.param.b64 	%rd8, [add_vectors_param_1];
	ld.param.b64 	%rd9, [add_vectors_param_2];
	ld.param.b32 	%r19, [add_vectors_param_3];
	ld.param.b32 	%r18, [add_vectors_param_4];

  // Omit for brevity
}
```

`ld.param.b64 	%rd7, [add_vectors_param_0]`

This PTX instruction is responsible for loading the first argument passed to your CUDA kernel into a register so the GPU can use it.

Here is a breakdown of exactly what each part of the instruction does:

* **`ld`**: This is the basic **load** instruction. It tells the GPU to read data from memory into a register.
* **`.param`**: This specifies the **state space** being accessed. In PTX, kernel arguments are passed via a special, read-only memory space called the **parameter space**.
* **`.b64`**: This indicates the **data type and size**. It stands for "64-bit binary." Because modern GPUs use 64-bit memory addressing, the pointer being passed to the kernel is 64 bits (8 bytes) long.
* **`%rd7`**: This is the **destination register**. The loaded 64-bit value will be stored in the 64-bit register named `%rd7` (which was declared earlier in the code as `.reg .b64 %rd<18>;`).
* **`[add_vectors_param_0]`**: This is the **source address**. It refers to the memory location of the very first parameter defined in the kernel's signature (`.param .u64 .ptr .align 1 add_vectors_param_0`).

**In plain English:**
"Read the 64-bit pointer passed as the first argument to the kernel from the parameter memory space, and store it in register `%rd7`."

Visualization:
```
                      param0   param1   param2   param3   param4
Parameter Space --> |--------|--------|--------|--------|--------|
                    |  arg0  |  arg1  |  arg2  |  arg3  |  arg4  |
                    |--------|--------|--------|--------|--------|
```
Where, `arg0`, `arg1`, ..., `argN` are exactly what you passed into the kernel. For example,
```C++
int main(int argc, char **argv) {
  using DataType = float;

  // Omit for brevity

  DataType *dev_a_ptr = nullptr;
  DataType *dev_b_ptr = nullptr;
  DataType *dev_c_ptr = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&dev_a_ptr, kNumBytes));
  CHECK_CUDA_ERROR(cudaMalloc(&dev_b_ptr, kNumBytes));
  CHECK_CUDA_ERROR(cudaMalloc(&dev_c_ptr, kNumBytes));

  // Omit for brevity

  add_vectors<<<gridSize, blockSize>>>(dev_a_ptr, dev_b_ptr, dev_c_ptr,
                                       kTotalElems, kMaxRound);

  // Omit for brevity
}
```

Then,
```
arg0 = dev_a_ptr, which is a pointer
arg1 = dev_b_ptr, which is a pointer
arg2 = dev_c_ptr, which is a pointer
arg3 = kTotalElems, which is an integer
arg4 = kMaxRound, which is an integer
```

After the `ld.param.b64` instructions are executed, 
```mlir
	ld.param.b64 	%rd7, [add_vectors_param_0];
	ld.param.b64 	%rd8, [add_vectors_param_1];
	ld.param.b64 	%rd9, [add_vectors_param_2];
	ld.param.b32 	%r19, [add_vectors_param_3];
	ld.param.b32 	%r18, [add_vectors_param_4];
```

- register `%rd7` contains `dev_a_ptr`, which is a pointer
- register `%rd8` contains `dev_b_ptr`, which is a pointer
- register `%rd9` contains `dev_c_ptr`, which is a pointer
- register `%r19` contains `kTotalElems`, which is an integer
- register `%r18` contains `kMaxRound`, which is an integer