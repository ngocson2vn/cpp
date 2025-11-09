import pycuda.driver as cuda
import pycuda.autoinit  # initializes a context
# Prepare device arrays (example using PyCUDA)
import pycuda.gpuarray as gpuarray
import numpy as np
import torch

cubin_path = "/data00/home/son.nguyen/workspace/models/eu_torch_gpu_remove_uesless_heads_r1125020_0/snapshot/v3_mod_123456_eu_ttp2@1762311748/eu_torch_gpu_remove_uesless_heads_r1125020_0/gpu/aot_compile_cache/inductor/triton/0/5JCCLTSPXZL37Z5XFEMM7ZYJRUTWKNXOAEGR23673G6PXO5OWF3A/triton_mm.cubin"

# Load cubin from file (or pass bytes directly)
with open(cubin_path, "rb") as f:
    cubin_bytes = f.read()

module = cuda.module_from_buffer(cubin_bytes)

# The entrypoint name is the Triton-generated kernel symbol.
# Often it matches the Python function name: "triton_mm".
triton_mm = module.get_function("triton_mm")

M, N, K = 8, 1781, 256
a = gpuarray.to_gpu(np.random.rand(M, K).astype(np.float32))
b = gpuarray.to_gpu(np.random.rand(K, N).astype(np.float32))
c = gpuarray.to_gpu(np.zeros((M, N), dtype=np.float32))

# Kernel launch configuration
BLOCK_M, BLOCK_N = 16, 32
gm = (M + BLOCK_M - 1) // BLOCK_M
gn = (N + BLOCK_N - 1) // BLOCK_N
grid = ( gm*gn, 1, 1 )
block = ( 64, 1, 1 )  # Triton uses warp scheduling internally; block size here is not used the same way as CUDA C kernels, but driver API requires a triple.

shared_bytes = 3072
triton_mm.set_attribute(cuda.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes)

# Triton kernels expect raw device pointers and scalars
triton_mm(
    a.gpudata, b.gpudata, c.gpudata,
    block=block, grid=grid, shared=shared_bytes
)

cuda.Context.synchronize()

# Retrieve result
c_host = c.get()
print(c_host)
