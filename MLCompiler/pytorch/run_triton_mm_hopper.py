import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER"] = "0"

import torch
from torch._inductor.codecache import PyCodeCache

import ipdb

triton_path = "./triton_mm_hopper.py"
key = os.path.basename(triton_path).split(".")[0]
mod = PyCodeCache.load_by_key_path(key, triton_path)

# ---- Host code to allocate inputs and launch ----
def main():
    torch.cuda.set_device(0)
    stream = torch.cuda.current_stream(0)
    
    # Shapes and strides
    M, N, K = 512, 256, 1024
    A = (torch.randn(M, K, dtype=torch.float16).cuda() - 0.5)
    B = (torch.randn(N, K, dtype=torch.float16).cuda() - 0.5)
    print(f"A: {A.shape} {A}")
    print(f"B: {B.shape} {B}")
    
    # Output: [M, N]
    Out = torch.empty(M, N, dtype=torch.float32).cuda()

    # Compute launch grid like Triton matmul
    BLOCK_M, BLOCK_N = 64, 64
    grid_m = (M + BLOCK_M - 1) // BLOCK_M  # 1
    grid_n = (N + BLOCK_N - 1) // BLOCK_N  # ceil(1781/32)
    
    # The original template uses FixedGrid with a single dim; emulate a 1D grid size
    # Triton kernel uses tl.program_id(0) only.
    grid_0 = grid_m * grid_n  # number of program ids
    print(f"Launching grid size: {grid_0} (grid_m={grid_m}, grid_n={grid_n})")
    
    # Launch
    torchOut = torch.matmul(A.to(torch.float32), B.T.to(torch.float32))

    # ipdb.set_trace()
    mod.matmul_kernel_tma.run(
        A, B, Out,
        M, N, K,
        _grid_0=grid_0,
        _grid_1=1,
        _grid_2=1,
        stream=stream.cuda_stream
    )
    torch.cuda.synchronize(torch.cuda.current_device())
    print(f"Out: {Out}")

    print("Verify results")
    EPSILON = 1e-3
    matched = True
    mismatch_count = 0
    for i in range(torchOut.shape[0]):
        for j in range(torchOut.shape[1]):
            diff = abs((torchOut[i, j] - Out[i, j]))
            if diff > EPSILON:
                matched = False
                print(f"{torchOut[i, j]} != {Out[i, j]}")
                mismatch_count += 1
    if matched:
        print(f"OK: matmul_tma matches torch.matmul")
    else:
        print(f"NG: There are {mismatch_count} mismatches between matmul_tma and torch.matmul")

if __name__ == "__main__":
    main()
