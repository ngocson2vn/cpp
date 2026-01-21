# Errors
## 1. undefined hidden symbol
Even though the function template `void lego_ops::softmax_top1_kernel<float>(float const*, int*, float*, int, int)` was manually instantiated, <br/>
the following error still occurs when compiling with CUDA 13.x:
```bash
ld.lld: error: undefined hidden symbol: void lego_ops::softmax_top1_kernel<float>(float const*, int*, float*, int, int)
>>> referenced by tmpxft_0028743a_00000000-6_moe.compute_100.cudafe1.cpp
```
This error is caused by a **breaking change introduced in CUDA 13.0** (and carried over to 13.1.0) regarding the default symbol visibility of device kernels.

### The Cause

In previous versions of CUDA, `__global__` functions (kernels) had default visibility, meaning they were exported and visible across shared library boundaries.

**Starting with CUDA 13.0**, the compiler (NVCC) defaults `__global__` functions to **hidden visibility** to prevent symbol collisions and improve safety in complex builds. This means that even if you manually instantiate your template:

1. The compiler generates the code for `softmax_top1_kernel<float>`.
2. It marks the symbol as `hidden` (private to that translation unit or shared library).
3. When another object file (like the generated `tmpxft_...` intermediate file) tries to link against it, the linker (`ld.lld`) rejects it because it cannot "see" the hidden symbol, resulting in `undefined hidden symbol`.

### Solutions

You can fix this in one of two ways:

#### Option 1: Revert to Legacy Behavior (Recommended for Quick Fix)

Pass the following flag to `nvcc` to disable the new hidden visibility behavior. This restores the behavior found in CUDA 12.x and earlier.

```bash
nvcc ... --device-entity-has-hidden-visibility=false ...

```

If you are using CMake, add this to your `CMAKE_CUDA_FLAGS`:

```cmake
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --device-entity-has-hidden-visibility=false")

```

#### Option 2: Explicitly Mark the Template as Visible

If you prefer to keep the new stricter defaults, you must explicitly mark your specific kernel template as visible in its definition/declaration.

Modify your code for `lego_ops::softmax_top1_kernel`:

```cpp
// Add __attribute__((visibility("default")))
template <typename T>
__global__ __attribute__((visibility("default")))
void softmax_top1_kernel(const T* input, int* output, float* val, int rows, int cols) {
    // ... kernel implementation ...
}

```

#### Option 3: Check `rdc` (Relocatable Device Code)

If you are compiling with Relocatable Device Code enabled (`-rdc=true`), you may also need to ensure that the template stub behavior matches your linkage expectations by adding:

```bash
-static-global-template-stub=false

```

### Summary of Flags

| Flag | Purpose |
| --- | --- |
| `--device-entity-has-hidden-visibility=false` | **Fixes your error.** Forces kernels to be visible (public) like in CUDA 12. |
| `-static-global-template-stub=false` | Prevents the compiler from making template stubs internal (static). |
