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
from torch._inductor.runtime import triton_helpers

import torch


@triton.jit
def triton_tem_fused_4x4(
    A0, B0, bias0, C0,
    A1, B1, bias1, C1,
    A2, B2, bias2, C2,
    A3, B3, bias3, C3,
    ks0,
):
    """
    This is a modified version of the original Inductor-generated kernel, 
    triton_tem_fused_4(arg_A, arg_B, in_ptr2, out_ptr1, ks0), which can compute only one problem,
    C = ReLU(A @ B + bias).

    This kernel can compute 4 independent problems concurrently: C_i = ReLU(A_i @ B_i + b_i), i = 0, 1, 2, 3
    4 problems are computed in 4 different groups. Each group consists of the same number of CTAs.
    """

    GROUP_M: tl.constexpr = 8
    ALLOW_TF32: tl.constexpr = True
    ACC_TYPE: tl.constexpr = tl.float32
    BLOCK_M: tl.constexpr = 32
    BLOCK_N: tl.constexpr = 64
    BLOCK_K: tl.constexpr = 32
    NUM_GEMMS: tl.constexpr = 4

    M = ks0
    N = 64
    K = 64
    if M * N == 0:
        return

    stride_am = 64
    stride_ak = 1
    stride_bk = 64
    stride_bn = 1

    # packed 1D grid: pid = gemm_group * grid_m + pid_in_gemm
    # For example, grid_m = cdiv(M, BLOCK_M) = 4 then,
    # pid = 0, 1, ..., (4 * 4 - 1)
    #|------------|------------|------------|--------------|----------------|
    #| gemm_group |      0     |      1     |      2       |       3        |
    #|------------|------------|------------|--------------|----------------|
    #| pid        | 0, 1, 2, 3 | 4, 5, 6, 7 | 8, 9, 10, 11 | 12, 13, 14, 15 |
    #|------------|------------|------------|--------------|----------------|

    # launch with grid = (NUM_GEMMS * cdiv(M, BLOCK_M), 1, 1)
    pid = tl.program_id(0)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N  # == 1 for N=BLOCK_N=64

    # which of the 4 GEMM groups (0..3)
    # Each gemm_group computes one problem C_i = ReLU(A_i @ B_i + b_i)
    gemm_group = pid // grid_m

    # pid inside a GEMM group
    # One GEMM computes grid_m tiles
    pid_in_gemm = pid % grid_m # 0, ..., (grid_m - 1)

    A = A0
    B = B0
    in_ptr2 = bias0
    out_ptr1 = C0
    if gemm_group == 1:
        A = A1
        B = B1
        in_ptr2 = bias1
        out_ptr1 = C1
    elif gemm_group == 2:
        A = A2
        B = B2
        in_ptr2 = bias2
        out_ptr1 = C2
    elif gemm_group == 3:
        A = A3
        B = B3
        in_ptr2 = bias3
        out_ptr1 = C3

    #================================================================================
    # original L2 swizzle, applied per-problem
    #================================================================================
    swizzle_size = GROUP_M * grid_n
    swizzle_id = pid_in_gemm // swizzle_size
    swizzle_height = min(grid_m - swizzle_id * GROUP_M, GROUP_M)
    pid_m = swizzle_id * GROUP_M + (pid_in_gemm % swizzle_height)
    pid_n = (pid_in_gemm % swizzle_size) // swizzle_height
    #================================================================================

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (
        (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)
    ) and M >= BLOCK_M:
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if (
        (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)
    ) and N >= BLOCK_N:
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N

    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        a = tl.load(A + (idx_n + 64 * idx_m))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        b = tl.load(B + (idx_n + 64 * idx_m))
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

    # rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    tmp0 = tl.load(
        in_ptr2 + tl.broadcast_to(idx_n, acc.shape),
        mask,
        eviction_policy="evict_last",
    ).to(tl.float32)
    tmp1 = acc + tmp0
    tmp2 = tl.full([1], 0, tmp1.dtype)
    tmp3 = triton_helpers.maximum(tmp2, tmp1)
    tl.store(
        out_ptr1 + tl.broadcast_to(idx_n + 64 * idx_m, acc.shape),
        tmp3,
        mask,
    )


def launch_fused_4x4(
    A0, B0, bias0, C0,
    A1, B1, bias1, C1,
    A2, B2, bias2, C2,
    A3, B3, bias3, C3,
    s1,
    num_warps=4,  # match whatever inductor used for the original
    num_stages=2,
):
    BLOCK_M = 32

    grid_m = (31 + s1) // BLOCK_M  # == cdiv(s1, BLOCK_M)
    # one CTA per (M-tile, problem)
    # Why tiling is done via M dimension?
    #                  N=64
    #        |------------------------|
    #        |                        |
    #        |         32x64          |
    #        |                        |
    #        |------------------------|
    #        |                        |
    #        |         32x64          |
    #        |                        |
    #  M=128 |------------------------|
    #        |                        |
    #        |         32x64          |
    #        |                        |
    #        |------------------------|
    #        |                        |
    #        |         32x64          |
    #        |                        |
    #        |------------------------|
    # 
    # In this case, BLOCK_N = N => grid_n = 1
    # Therefore, grid_n is ignored.
    # Normally, grid = grid_m * grid_n

    triton_tem_fused_4x4.run(
        A0, B0, bias0, C0,
        A1, B1, bias1, C1,
        A2, B2, bias2, C2,
        A3, B3, bias3, C3,
        s1,
        grid=(grid_m * 4, 1, 1),
        warmup=None,
        num_warps=num_warps,
        num_stages=num_stages,
    )


from util import diff


def main():
    M = 128
    N = 64
    K = 64
    DEVICE = torch.device("cuda", 0)

    # =================================================================
    # A0, B0, bias0
    # =================================================================
    A0 = torch.rand(M, K, dtype=torch.float16, device=DEVICE) - 0.5
    B0 = torch.rand(K, N, dtype=torch.float16, device=DEVICE) - 0.5

    print(f"A0: dtype={A0.dtype} shape={A0.shape} tensor={B0}")
    print(f"B0: dtype={B0.dtype} shape={B0.shape} tensor={B0}")
    print(f"A0.stride: {A0.stride()}")
    print(f"B0.stride: {B0.stride()}")

    bias0 = torch.zeros(N, dtype=torch.float16, device=DEVICE)
    C0 = torch.zeros([M, N], dtype=torch.float16, device=DEVICE)
    print()

    # =================================================================
    # A1, B1, bias1
    # =================================================================
    A1 = torch.rand(M, K, dtype=torch.float16, device=DEVICE) - 0.5
    B1 = torch.rand(K, N, dtype=torch.float16, device=DEVICE) - 0.5

    print(f"A1: dtype={A1.dtype} shape={A1.shape} tensor={B1}")
    print(f"B1: dtype={B1.dtype} shape={B1.shape} tensor={B1}")
    print(f"A1.stride: {A1.stride()}")
    print(f"B1.stride: {B1.stride()}")

    bias1 = torch.zeros(N, dtype=torch.float16, device=DEVICE)
    C1 = torch.zeros([M, N], dtype=torch.float16, device=DEVICE)
    print()

    # =================================================================
    # A2, B2, bias2
    # =================================================================
    A2 = torch.rand(M, K, dtype=torch.float16, device=DEVICE) - 0.5
    B2 = torch.rand(K, N, dtype=torch.float16, device=DEVICE) - 0.5

    print(f"A2: dtype={A2.dtype} shape={A2.shape} tensor={B2}")
    print(f"B2: dtype={B2.dtype} shape={B2.shape} tensor={B2}")
    print(f"A2.stride: {A2.stride()}")
    print(f"B2.stride: {B2.stride()}")

    bias2 = torch.zeros(N, dtype=torch.float16, device=DEVICE)
    C2 = torch.zeros([M, N], dtype=torch.float16, device=DEVICE)
    print()

    # =================================================================
    # A3, B3, bias2
    # =================================================================
    A3 = torch.rand(M, K, dtype=torch.float16, device=DEVICE) - 0.5
    B3 = torch.rand(K, N, dtype=torch.float16, device=DEVICE) - 0.5

    print(f"A3: dtype={A3.dtype} shape={A3.shape} tensor={B3}")
    print(f"B3: dtype={B3.dtype} shape={B3.shape} tensor={B3}")
    print(f"A3.stride: {A3.stride()}")
    print(f"B3.stride: {B3.stride()}")

    bias3 = torch.zeros(N, dtype=torch.float16, device=DEVICE)
    C3 = torch.zeros([M, N], dtype=torch.float16, device=DEVICE)
    print()

    launch_fused_4x4(
        A0, B0, bias0, C0,
        A1, B1, bias1, C1,
        A2, B2, bias2, C2,
        A3, B3, bias3, C3,
        M
    )

    torch.cuda.synchronize()

    print(f"C0: dtype={C0.dtype} shape={C0.shape} tensor={C0}")
    print()

    print(f"C1: dtype={C1.dtype} shape={C1.shape} tensor={C1}")
    print()

    print(f"C2: dtype={C2.dtype} shape={C2.shape} tensor={C2}")
    print()

    print(f"C3: dtype={C3.dtype} shape={C3.shape} tensor={C3}")
    print()

    torch_C0 = torch.relu(torch.matmul(A0.float(), B0.float()))
    torch_C1 = torch.relu(torch.matmul(A1.float(), B1.float()))
    torch_C2 = torch.relu(torch.matmul(A2.float(), B2.float()))
    torch_C3 = torch.relu(torch.matmul(A3.float(), B3.float()))
    torch.cuda.synchronize()

    print("Verify results")
    matched, mismatch_count = diff(C0, torch_C0)
    if matched:
        print(f"OK: C0 matches torch_C0")
    else:
        print(f"NG: There are {mismatch_count} mismatches between C0 and torch_C0")
        sys.exit(1)
    print()

    matched, mismatch_count = diff(C1, torch_C1)
    if matched:
        print(f"OK: C1 matches torch_C1")
    else:
        print(f"NG: There are {mismatch_count} mismatches between C1 and torch_C1")
        sys.exit(1)
    print()

    matched, mismatch_count = diff(C2, torch_C2)
    if matched:
        print(f"OK: C2 matches torch_C2")
    else:
        print(f"NG: There are {mismatch_count} mismatches between C2 and torch_C2")
        sys.exit(1)
    print()

    matched, mismatch_count = diff(C3, torch_C3)
    if matched:
        print(f"OK: C3 matches torch_C3")
    else:
        print(f"NG: There are {mismatch_count} mismatches between C3 and torch_C3")
        sys.exit(1)
    print()


main()
