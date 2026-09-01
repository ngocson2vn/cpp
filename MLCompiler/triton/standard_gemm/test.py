# ===============================================================================================
# Prerequisites
# ===============================================================================================
"""
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:$LD_LIBRARY_PATH

export CUDNN_LIB_DIR=${HOME}/.pyenv/versions/3.11.2/lib/python3.11/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH=${CUDNN_LIB_DIR}:$LD_LIBRARY_PATH

export TRITON_CACHE_DIR=./tmp_triton_cache
mkdir -p ${TRITON_CACHE_DIR}
"""

# ===============================================================================================

import sys

import triton
import triton.language as tl

import torch


@triton.jit
def triton_gemm(
    A, B, C,
    M, N, K
):
    """
    A: MxK
    B: KxN
    C: MxN
    """
    GROUP_M: tl.constexpr = 8
    ALLOW_TF32: tl.constexpr = True
    ACC_TYPE: tl.constexpr = tl.float32
    BLOCK_M: tl.constexpr = 16
    BLOCK_N: tl.constexpr = 16
    BLOCK_K: tl.constexpr = 16

    if ((M * N == 0) or (M % BLOCK_M != 0)) or ((N % BLOCK_N != 0) or (K % BLOCK_K != 0)):
        return

    # launch with grid = (cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N), 1, 1)
    pid = tl.program_id(0)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    #================================================================================
    # Original L2 swizzle, applied per-problem
    # Read more GEMM.md
    #================================================================================
    swizzle_size = GROUP_M * grid_n
    swizzle_id = pid // swizzle_size
    swizzle_height = min(grid_m - swizzle_id * GROUP_M, GROUP_M)
    pid_m = swizzle_id * GROUP_M + (pid % swizzle_height)
    pid_n = (pid % swizzle_size) // swizzle_height
    #================================================================================

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # NOTE: You must ensure that M is multiple of BLOCK_M
    offs_a_m = tl.max_contiguous(rm, BLOCK_M)
    idx_m = offs_a_m[:, None]

    # NOTE: You must ensure that N is multiple of BLOCK_N
    offs_b_n = tl.max_contiguous(rn, BLOCK_N)
    idx_n = offs_b_n[None, :]

    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    # Loop through K blocks
    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        # Load A block
        idx_k = offs_k[None, :] + (k_idx * BLOCK_K) # shape = (1, BLOCK_K)
        a = tl.load(A + (idx_k + K * idx_m))

        # Load B block
        idx_k = offs_k[:, None] + (k_idx * BLOCK_K) # shape = (BLOCK_K, 1)
        b = tl.load(B + (idx_n + N * idx_k))

        # Compute dot product
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    tl.store(
        C + tl.broadcast_to(idx_n + N * idx_m, acc.shape),
        acc,
        mask,
    )

from util import diff
def main():
    M = 32
    N = 32
    K = 32
    BLOCK_M = 16
    BLOCK_N = 16
    DEVICE = torch.device("cuda", 0)

    # =================================================================
    # A, B, C
    # =================================================================
    A = torch.rand(M, K, dtype=torch.float16, device=DEVICE) - 0.5
    B = torch.rand(K, N, dtype=torch.float16, device=DEVICE) - 0.5

    print(f"A.shape: {list(A.shape)}")
    print(f"A.stride: {list(A.stride())}")
    print(f"A.dtype: {A.dtype}")
    print(f"A.tensor: {A}")
    print()

    print(f"B.shape: {list(B.shape)}")
    print(f"B.stride: {list(B.stride())}")
    print(f"B.dtype: {B.dtype}")
    print(f"B.tensor: {B}")

    C = torch.zeros([M, N], dtype=torch.float16, device=DEVICE)
    print()


    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    triton_gemm.run(
        A, B, C,
        M, N, K,
        grid=(grid_m * grid_n, 1, 1),
        warmup=None
    )

    torch.cuda.synchronize()
    print(f"C.shape: {list(C.shape)}")
    print(f"C.stride: {list(C.stride())}")
    print(f"C.dtype: {C.dtype}")
    print(f"C.tensor: {C}")
    print()

    torch_C = torch.matmul(A.float(), B.float())
    torch.cuda.synchronize()

    print("Verify results")
    matched, mismatch_count = diff(C, torch_C)
    if matched:
        print(f"OK: C matches torch_C")
    else:
        print(f"NG: There are {mismatch_count} mismatches between C and torch_C")
        sys.exit(1)
    print()


main()
