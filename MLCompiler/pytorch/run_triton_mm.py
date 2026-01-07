import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from torch._inductor.codecache import PyCodeCache

import ipdb

# triton_path = "./cases/triton_mm_faulty.py"
triton_path = "./cases/triton_mm_ok.py"
key = os.path.basename(triton_path).split(".")[0]
mod = PyCodeCache.load_by_key_path(key, triton_path)

# ---- Host code to allocate inputs and launch ----
def main():
    device = 'cuda'
    torch.cuda.set_device(0)
    stream = torch.cuda.current_stream(0)
    # Shapes and strides
    M, N, K = 8, 1781, 256
    # A: shape [M, K], stride_am=256, stride_ak=1
    A = torch.randn((M, K), device=device, dtype=torch.float32)
    # Ensure row-major [M, K] => stride (K, 1) which is (256, 1)
    assert A.stride() == (K, 1), f"A strides {A.stride()} != (256, 1)"
    # B: shape [K, N], logical stride_bk=1781, stride_bn=1
    # Allocate exactly [K, N] without padding to reproduce misaligned access.
    B = torch.randn((K, N), device=device, dtype=torch.float32)
    assert B.stride() == (N, 1), f"B strides {B.stride()} != (1781, 1)"
    # Output: [M, N]
    Out = torch.empty((M, N), device=device, dtype=torch.float32)
    # Compute launch grid like Triton matmul
    BLOCK_M, BLOCK_N = 16, 32
    grid_m = (M + BLOCK_M - 1) // BLOCK_M  # 1
    grid_n = (N + BLOCK_N - 1) // BLOCK_N  # ceil(1781/32)
    # The original template uses FixedGrid with a single dim; emulate a 1D grid size
    # Triton kernel uses tl.program_id(0) only.
    grid_0 = grid_m * grid_n  # number of program ids
    print(f"Launching grid size: {grid_0} (grid_m={grid_m}, grid_n={grid_n})")
    # Launch

    # ipdb.set_trace()

    mod.triton_mm.run(
        A, B, Out,
        _grid_0=grid_0,
        _grid_1=1,
        _grid_2=1,
        stream=stream.cuda_stream
    )
    # If it didn't crash, print a sanity message
    torch.cuda.synchronize(torch.cuda.current_device())
    # print("Kernel finished (no crash). Out[0,0] =", Out[0, 0].item())
    print(f"Out: {Out}")

if __name__ == "__main__":
    main()
