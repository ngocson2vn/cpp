# TensorFlow Compiler

## 1. Convert GraphDef to TensorFlow Dialect
TensorFlow Dialect
https://github.com/tensorflow/tensorflow/tree/v2.9.0/tensorflow/compiler/mlir/tensorflow

defines
- TensorFlow Dialect
- TensorFlow Ops

## 2. Fuse Ops
Steps
- Fuse ops into a MLIR function - callee
- Replace fused ops with a custom op that accepts all necessary input tensors extracted from fused ops.

Fusion Strategy
- Vertical Fusion:
  - Fuse point-wise ops such as Add, Multiply, Div, Pow, StridedSlice and Boolean ops vertically
  - LayerNorm fusion
  - GELU fusion
  - ReLU(GEMM + Bias) -> call cublasLt API
  - GELU(GEMM + Bias) -> call cublasLt API
  - Fuse multiple related custom ops
- Horizontal Fusion:
  - Fuse multiple unrelated custom ops horizontally

## 3. Lower callee functions
We lower each calle function to a CUDA kernel and a host function that launches the CUDA kernel.

Lowering Pipeline:
```txt
- TF dialect -> MLIR-HLO -> linalg.generic
  -> scf.parallel ops -> Merge scf.parallel ops -> Kernel functions (rely on MLIR SCFToGPU pass)
  -> Host function (uses gpu.launch_func to launch a kernel function)

- Kernel function -> LLVM dialect + NVVM dialect -> LLVM IR 
  -> NVPTX Target Backend -> PTX -> CUBIN -> Global blob symbol

- Host function -> LLVM dialect -> LLVM IR + External function calls -> Object file
  External function calls are calls to runtime APIs such as APIs for calling cublasLt APIs, launching CUDA kernel in the Global blob symbol

- Object file -> g++ -> Shared object library
```

All CUDA kernels and host functions are compiled into a shared object library. <br/>
Next, we serialize the `.so` library to a byte string and store it into an attribute of a `Const` node in the model's GraphDef.

## 4. Runtime
We develop a custom runtime library to parse and load the model's GraphDef.

The runtime performs the following steps:

**Loading phase**: <br/>
- Loops over the GraphDef's nodes to extract CUDA kernel names from custom nodes
- Extracts the byte string from GraphDef and creates a temporary `.so` file from it.
- Calls `dlopen()` to load the `.so` file and dynamically link runtime APIs with it.
- Calls `dlsym()` to get the address of the host function for each kernel name.
- Calls `dlsym()` to get the address of the global blob symbol for each CUDA kernel name.
- Calls `cuModuleLoadData()` to load the blob onto GPU memory
- Calls `cuModuleGetFunction()` to get the GPU address of the CUDA kernel residing inside the CUDA module.

**Execution phase**: <br/>
- TensorFlow executes a custom op
- Custom op calls the corresponding host function with 4 arguments: (1) kernel pointer, (2) OpKernelContext pointer, (3) array of input tensor descriptors, (4) array of output tensor descriptors
- Host function calls a runtime API for launching the CUDA kernel
- The runtime API calls `cuLaunchKernel()` API
