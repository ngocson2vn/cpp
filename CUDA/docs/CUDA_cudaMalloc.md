# cudaMalloc
By default, memory allocated using `cudaMalloc()` in the CUDA Runtime API is guaranteed to be aligned to at least **256 bytes**.

This generous default alignment is intentional and serves a few critical purposes for GPU architecture:

* **Universal Compatibility:** A 256-byte alignment guarantees that the starting address is suitable for *any* built-in data type or vector type (such as `float4`, `double2`, or `int4`). Since 256 is a multiple of the strict 16-byte alignment required by these large types, you can safely cast a `cudaMalloc` pointer to any type without triggering a misalignment exception or performance penalty.
* **Memory Coalescing:** The GPU fetches memory from global memory in chunks (typically 32, 64, or 128 bytes). Because 256 is a multiple of these cache line sizes, starting an array at a 256-byte aligned address ensures that the memory segments map perfectly to the GPU's memory transactions, optimizing for coalesced access on the first read.
* **Texture and Surface Memory:** Certain specialized GPU memory operations, such as binding global memory to texture or surface references, have strict alignment constraints. The 256-byte guarantee ensures the base pointer is already compliant with these hardware requirements.

### When 256-Byte Alignment Isn't Enough (2D/3D Arrays)

While `cudaMalloc()` guarantees the *base* pointer is 256-byte aligned, it does not guarantee that subsequent rows in a 2D or 3D array will be properly aligned for optimal coalesced access.

If you are allocating a 2D or 3D array where the width (in bytes) is not a multiple of 256, you should use **`cudaMallocPitch()`** or **`cudaMalloc3D()`** instead. These functions automatically pad the end of each row (the "pitch") to ensure that every single row starts at a properly aligned memory address, preserving optimal memory bandwidth across the entire multidimensional structure.
