import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


def matmul_get_configs(pre_hook=None):
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GM}, 
                      num_stages=s,
                      num_warps=8,  # <--- FIXED: Force 8 warps for Warp Specialization
                      pre_hook=pre_hook)
        for BM in [64]
        for BN in [64]
        for BK in [64]          # Recommend sticking to 64 for BK to save SRAM
        for GM in [4, 8]
        for s in ([3, 4, 5])    # TMA benefits from more stages (3+)
    ]


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_N, BLOCK_K]
    nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]


@triton.autotune(
    # configs=matmul_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    configs=matmul_get_configs(),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
    cache_results=True
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma(a_desc, b_desc, c_desc,
                      M, N, K,
                      BLOCK_SIZE_M: tl.constexpr,
                      BLOCK_SIZE_N: tl.constexpr,
                      BLOCK_SIZE_K: tl.constexpr,
                      GROUP_SIZE_M: tl.constexpr,
                      WARP_SPECIALIZE: tl.constexpr,
                      ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(k_tiles, warp_specialize=WARP_SPECIALIZE):
        offs_k = k * BLOCK_SIZE_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

    offs_cm = pid_m * BLOCK_SIZE_M
    offs_cn = pid_n * BLOCK_SIZE_N
    c_desc.store([offs_cm, offs_cn], accumulator)


def matmul_tma(a, b, warp_specialize: bool):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # A dummy block value that will be overwritten when we have the real block size
    block_shape = [64, 64]
    a_desc = TensorDescriptor.from_tensor(a, block_shape)
    b_desc = TensorDescriptor.from_tensor(b, block_shape)
    c_desc = TensorDescriptor.from_tensor(c, block_shape)

    def grid(META):
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    matmul_kernel_tma[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        WARP_SPECIALIZE=warp_specialize
    )

    return c

# Grok
# https://grok.com/c/180b8e7f-04a0-44b8-a0b4-117e358a66f2
"""
This is an optimal Triton implementation of GEMM for Hopper GPUs, leveraging the Tensor Memory Accelerator (TMA) for asynchronous memory transfers. 
The matmul_kernel_tma function is the key kernel using TMA descriptors for loads and stores. 
It supports FP16 and FP8 outputs and includes autotuning for block sizes and warp specialization for better performance on Hopper.
To use it, call `matmul_tma(a, b.T, warp_specialize=True)` where a is (M, K) and b is (K, N) (note that b needs to be transposed for the kernel).
"""

def main():
    a = (torch.rand(256, 1024, dtype=torch.float16).cuda() - 0.5)
    print(f"a.stride: {a.stride()}")
    b = (torch.rand(512, 1024, dtype=torch.float16).cuda() - 0.5)
    print(f"b.stride: {b.stride()}")

    print(f"a: dtype={a.dtype} shape={a.shape} tensor={a}")
    print(f"b: dtype={b.dtype} shape={b.shape} tensor={b}")

    c_torch_matmul = torch.matmul(a, b.T)
    print(f"c_torch_matmul: dtype={c_torch_matmul.dtype} shape={c_torch_matmul.shape} tensor={c_torch_matmul}")

    c_matmul_tma = matmul_tma(a, b, warp_specialize=True)
    print(f"c_matmul_tma: dtype={c_matmul_tma.dtype} shape={c_matmul_tma.shape} tensor={c_matmul_tma}")
    print()

    print("Verify results")
    EPSILON = 1e-2
    matched = True
    mismatch_count = 0
    for i in range(c_torch_matmul.shape[0]):
        for j in range(c_torch_matmul.shape[1]):
            diff = abs((c_torch_matmul[i, j] - c_matmul_tma[i, j]))
            if diff > EPSILON:
                matched = False
                print(f"{c_torch_matmul[i, j]} != {c_matmul_tma[i, j]}")
                mismatch_count += 1

    if matched:
        print(f"OK: matmul_tma matches torch.matmul")
    else:
        print(f"NG: There are {mismatch_count} mismatches between matmul_tma and torch.matmul")

if __name__ == "__main__":
    main()
