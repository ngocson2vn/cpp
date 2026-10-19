#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda/std/type_traits>

using fp16_t = __nv_half;
using bf16_t = __nv_bfloat16;

#define DEVICE_FUNC __device__ __forceinline__

#define CHECK_CUDA_ERROR(callExpr)                                             \
  do {                                                                         \
    auto ret = callExpr;                                                       \
    if (ret != cudaSuccess) {                                                  \
      auto errorName = cudaGetErrorName(ret);                                  \
      auto errorStr = cudaGetErrorString(ret);                                 \
      fprintf(stderr, "ERROR %s:%d %s: %s\n", __FILE__, __LINE__, errorName,   \
              errorStr);                                                       \
    }                                                                          \
  } while (0)


//======================================================================================================================
// Device functions
//======================================================================================================================
namespace PTX {

DEVICE_FUNC void syncCTAAligned() {
  asm volatile(
    "{\n"
    "\t\tbarrier.sync.aligned \t0;\n"
    "\t}"
    :
    :
    : "memory"
  );
}

DEVICE_FUNC void sync256Threads() {
  asm volatile(
    "{\n"
    "\t\tbarrier.sync.aligned \t1, 256;\n"
    "\t}"
    :
    :
    : "memory"
  );
}

DEVICE_FUNC void syncCTAUnaligned() {
  asm volatile(
    "{\n"
    "\t\tbarrier.sync \t2;\n"
    "\t}"
    :
    :
    : "memory"
  );
}


// Elect a leader thread in the executing wrap
// NOTE: Writing to %0 in any case to ensure that %0 holds either 0 or 1
// Otherwise, an uninitialized register may cause a correctness bug.
DEVICE_FUNC bool electLeaderLane() {
  uint32_t pred = 0;
  asm volatile(
    "{\n"
    "\t\t.reg .pred \t%p;\n"
    "\t\telect.sync \t_|%p, 0xFFFFFFFF;\n"
    "\t\tselp.u32 \t%0, 1, 0, %p;\n"
    "\t}"
    : "=r"(pred)
  );

  return (pred == 1);
}

// NOTE: ONLY the leader lane should execute this
DEVICE_FUNC void mbarInit(const void* mbarPtr, uint32_t arriveCount) {
  auto mbarU32 = static_cast<uint32_t>(__cvta_generic_to_shared(mbarPtr));
  asm volatile(
    "{\n"
    "\t\tmbarrier.init.shared::cta.b64 \t[%0], %1;\n"
    "\t}"
    : 
    : "r"(mbarU32), "r"(arriveCount)
    : "memory"
  );
}

// NOTE: ONLY the leader lane should execute this
DEVICE_FUNC void mbarArrive(const void* mbarPtr) {
  asm volatile(
    "{\n"
    "\t\tmbarrier.arrive.shared::cta.b64 \t_, [%0];\n"
    "\t}"
    :
    : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbarPtr)))
    : "memory"
  );
}

// NOTE: ONLY the leader lane should execute this
DEVICE_FUNC void mbarWait(const void* mbarPtr, uint32_t phaseParity) {
  // Arbitrarily large timer value after which try-wait expires and re-tries.
  uint32_t ticks = 0x989680;
  asm volatile(
    "{\n"
      "\t\t.reg .pred \tcomplete;\n"
      "\t\tLAB_WAIT:\n"
      "\t\t\tmbarrier.try_wait.parity.shared.b64 \tcomplete, [%0], %1, %2;\n"
      "\t\t\t@!complete bra.uni LAB_WAIT;\n"
    "\t}"
    :
    : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbarPtr))),
      "r"(phaseParity),
      "r"(ticks)
  );
}

DEVICE_FUNC void mbarArriveAndExpectTxBytes(void const* mbarPtr, 
                                                  uint32_t txBytes) {
  asm volatile(
    "{\n"
    "\t\tmbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
    "\t}"
    :
    : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbarPtr))),
      "r"(txBytes)
    : "memory"
  );
}

// Allocate TMEM
// Require 1 warp to execute this instruction
DEVICE_FUNC void tmemAlloc(uint32_t* tmemBaseHandlePtr, uint32_t cols) {
  auto tmemHandleU32 = static_cast<uint32_t>(
                          __cvta_generic_to_shared(tmemBaseHandlePtr));
  asm volatile(
    "{\n"
    "\t\ttcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
    "\t}"
    :
    : "r"(tmemHandleU32), "r"(cols)
    : "memory"
  );
}

// Require the same warp that previously executed `tcgen05.alloc`
// to execute this instruction
DEVICE_FUNC void tmemRelinquishAllocPermit() {
  asm volatile(
    "{\n"
    "\t\ttcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n"
    "\t}"
    :
    :
    : "memory"
  );
}

// NOTE: Require 1 warp to execute this instruction
DEVICE_FUNC void tmemDealloc(uint32_t tmemBaseHandle) {
  asm volatile(
    "{\n"
    "\t\ttcgen05.dealloc.cta_group::1.sync.aligned.b32 \t%0, 64;\n"
    "\t}"
    :
    : "r"(tmemBaseHandle)
    : "memory"
  );
}

// NOTE: `.sync.aligned` requires that all threads in 
// the executing warp must execute the same conditional code.
// i.e, they must have the same program counter.
DEVICE_FUNC void tmemReset(uint32_t tmemHandlePerWarp) {
  asm volatile(
    "{\n"
    "\t\ttcgen05.st.sync.aligned.16x32bx2.x16.b32 \t[%0 + 0], 16, "
                        "{%1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1};\n"
    "\t}"
    :
    : "r"(tmemHandlePerWarp), "r"(0)
    : "memory"
  );
}

DEVICE_FUNC void tmemLoad(float* acc, uint32_t tmemHandle) {
  asm volatile(
    "{\n"
    "\t\ttcgen05.ld.sync.aligned.16x32bx2.x16.b32 \t"
    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16 + 0], 16;\n"
    "\t}"
    : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]),  "=f"(acc[3]),
      "=f"(acc[4]), "=f"(acc[5]), "=f"(acc[6]),  "=f"(acc[7]),
      "=f"(acc[8]), "=f"(acc[9]), "=f"(acc[10]), "=f"(acc[11]),
      "=f"(acc[12]), "=f"(acc[13]), "=f"(acc[14]), "=f"(acc[15])
    : "r"(tmemHandle)
    : "memory"
  );
}

// TMA
DEVICE_FUNC void tmaCopyTile(const void* smemPtr, const void* tmapPtr, uint32_t col, uint32_t row, const void* mbarPtr) {
    asm volatile(
      "{\n"
      "\t\tcp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
      "\t}"
         :
         : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smemPtr)))
           "l"(reinterpret_cast<uint64_t>(tmapPtr)),
           "r"(col),
           "r"(row),
           "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbarPtr)))
         : "memory"
    );
}


// Marks the commit point for one or more sized batch of warpgroup MMAs.
DEVICE_FUNC void tmemCommitBatch(uint32_t pred, const void* mbarPtr) {
  asm volatile(
    "{\n"
    "\t\t.reg .pred \t%p;\n"
    "\t\tsetp.eq.b32 \t%p, %0, 1;\n"
    "\t\t@%p tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%1];\n"
    "\t}"
    :
    : "r"(pred), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbarPtr)))
    : "memory"
  );
}

DEVICE_FUNC void tmemWaitStore() {
  asm volatile(
    "{\n"
    "\t\ttcgen05.wait::st.sync.aligned;\n"
    "\t}"
    :
    :
    : "memory");
}

DEVICE_FUNC void tmemWaitLoad() {
  asm volatile(
    "{\n"
    "\t\ttcgen05.wait::ld.sync.aligned;\n"
    "\t}"
    :
    :
    : "memory");
}

DEVICE_FUNC void mma(
  uint32_t tmemHandle,
  uint64_t descA,
  uint64_t descB,
  uint32_t instDesc
) {
  // [ %0 + 0 ] is primarily for canonical consistency and architectural alignment.
  // The result will be accumulated in TMEM tiles
  asm volatile(
    "{\n"
    "\t\t.reg .pred \t%p;\n"
    "\t\tsetp.eq.u32 \t%p, 1, 1;\n"
    "\t\ttcgen05.mma.cta_group::1.kind::f16 \t[ %0 + 0 ], %1, %2, %3, %p;\n"
    "\t}"
    :
    : "r"(tmemHandle),
      "l"(descA),
      "l"(descB),
      "r"(instDesc)
    : "memory"
  );
}

DEVICE_FUNC void commitMMABatch(const uint32_t smemAddr) {
  asm volatile(
    "{\n"
    "\t\ttcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
    "\t}"
    :
    : "r"(smemAddr)
    : "memory"
  );
}

} // namespace PTX


//======================================================================================================================
// SMEM types
//======================================================================================================================
namespace CUDA {

template <typename ElemType, int ROW_SIZE, int COL_SIZE>
struct StageData {
 public:
  constexpr DEVICE_FUNC const ElemType* rawPtr() const {
    return &data_[0];
  }

 private:
  ElemType data_[ROW_SIZE * COL_SIZE];
};


// A type for encapsulating SMEM data
// Every parameter is derived from `MMAType` so that 
// a `SMEMData` type is always consistent with `MMAType`.
template <typename MMAType>
struct SMEMData {
  using InputType = typename MMAType::MMAInputType;
  using StageDataTypeA = StageData<InputType,
                                   MMAType::tileSizeM(), MMAType::tileSizeK()>;
  using StageDataTypeB = StageData<InputType,
                                   MMAType::tileSizeN(), MMAType::tileSizeK()>;

  uint32_t tmemBaseHandle;
  StageDataTypeA stageDataA[MMAType::stageCount()];
  StageDataTypeB stageDataB[MMAType::stageCount()];
  uint64_t producerMbars[MMAType::stageCount()];
  uint64_t consumerMbars[MMAType::stageCount()];
  uint64_t finalMbar;

  constexpr DEVICE_FUNC const InputType* dataA(const uint32_t stageIdx) const {
    return stageDataA[stageIdx].rawPtr();
  }

  constexpr DEVICE_FUNC const InputType* dataB(const uint32_t stageIdx) const {
    return stageDataB[stageIdx].rawPtr();
  }

  DEVICE_FUNC uint32_t producerMbar(const uint32_t stageIdx) const {
    return static_cast<uint32_t>(
      __cvta_generic_to_shared(producerMbars + stageIdx));
  }

  DEVICE_FUNC uint32_t consumerMbar(const uint32_t stageIdx) const {
    return static_cast<uint32_t>(
      __cvta_generic_to_shared(consumerMbars + stageIdx));
  }
};


// A type for getting Matrix Descriptors for a `MMAType`.
// Ref: https://docs.nvidia.com/cuda/archive/12.4.1/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
// Every parameter is derived from `MMAType` so that 
// a `MatDescType` is always consistent with `MMAType`.
template <typename MMAType>
struct MatDesc {
  // Assert MMAType::tileSizeM()
  static_assert(MMAType::tileSizeM() % 64 == 0,
                "This MatDesc only supports a MMAType "
                "that its TILE_M must be divisible by 64");

  // Assert MMAType::tileSizeK()
  static_assert(MMAType::tileSizeK() % 16 == 0,
                "This MatDesc only supports a MMAType "
                "that its TILE_K must be divisible by 16");

  using SMEMDataType = typename MMAType::SMEMDataType;

  // The number of parts along M is TILE_M divided by 64
  static constexpr uint32_t PART_COUNT_M = (MMAType::tileSizeM() >> 6);

  // The number of parts along K is TILE_K divided by 16
  static constexpr uint32_t PART_COUNT_K = (MMAType::tileSizeK() >> 4);

  uint64_t descArrayA[MMAType::stageCount() * PART_COUNT_M * PART_COUNT_K];
  uint64_t descArrayB[MMAType::stageCount() * PART_COUNT_K];

  DEVICE_FUNC uint64_t descA(uint32_t descIdxA) const {
    return descArrayA[descIdxA];
  }

  DEVICE_FUNC uint64_t descB(uint32_t descIdxB) const {
    return descArrayB[descIdxB];
  }

  // Create matrix descriptors
  // Core matrix shape: 8x8 elements
  // Matrix A: 8x2 core matrices => 64x16 elements
  // Matrix B: 2x(TILE_N/8) core matrices => 16xTILE_N elements
  static DEVICE_FUNC MatDesc get(const SMEMDataType& smem) {
    MatDesc md;

    #pragma unroll
    for (uint32_t stageIdx = 0; stageIdx < MMAType::stageCount(); stageIdx++) {
      #pragma unroll
      for (uint32_t k = 0; k < PART_COUNT_K; k++) {
        // Compute descArrayA
        #pragma unroll
        for (uint32_t m = 0; m < PART_COUNT_M; m++) {
          // m is strided by 64 * TILE_K elements
          // k is strided by 16 elements
          auto smemPtrA = smem.dataA(stageIdx) + m * (64 * MMAType::tileSizeK()) + k * 16;
          auto descIdxA = stageIdx * (PART_COUNT_M * PART_COUNT_K) + m * PART_COUNT_K + k;
          md.descArrayA[descIdxA] = getMatDesc(smemPtrA);
        }

        // Compute descArrayB
        auto smemPtrB = smem.dataB(stageIdx) + k * 16;
        auto descIdxB = stageIdx * PART_COUNT_K + k;
        md.descArrayB[descIdxB] = getMatDesc(smemPtrB);
      }
    }

    return md;
  }

 private:
  // Ref: https://docs.nvidia.com/cuda/archive/12.4.1/parallel-thread-execution/index.html#shared-memory-matrix-layout
  // Matrices in shared memory are organized into a number of smaller matrices called core matrices. 
  // Each core matrix has 8 rows or columns and the size of each row is 16 bytes. 
  // The core matrices occupy contiguous space in shared memory.
  // Matrix A is made up of 8x2 core matrices and Matrix B is made up of 2x(N/8) core matrices.
  // Core matrix A(8x8): Each row is made up of eight .f16/ .bf16 elements.
  // Core matrix B(8x8): Each column is made up of eight .f16/ .bf16 elements.
  // 
  // @param smemPtr:
  //   The pointer to SMEM location where matrix A or B starts.
  static DEVICE_FUNC uint64_t getMatDesc(const void* smemPtr) {
    // Compute stride dimension byte offset (SBO)
    // Stride dimension byte offset of matrix A or B is the distance, in bytes, 
    // between two adjacent core matrices in the M or N dimension.
    // (1) Each core matrix has 8 rows.
    // (2) The next core matrix along M or N is strided 
    // from the first one by 8 * TILE_K elements.
    constexpr uint32_t SBO = 8 * MMAType::tileSizeK() 
                               * sizeof(MMAType::MMAInputType);

    // matrix-descriptor-encode(x)
    auto mde = [](uint64_t x) -> uint64_t { return ((x & 0x3FFFF) >> 0x4); };

    uint64_t smemAddr = mde(static_cast<uint64_t>(__cvta_generic_to_shared(smemPtr)));

    uint64_t desc = 0;

    // Matrix start address: bits 13-0 (14 bits)
    desc = desc | smemAddr;

    // Leading dimension byte offset: bits 29–16 (14 bits)
    // This bit field is ignored when a swizzling mode is set,
    // so we set it to 0
    desc = desc | (mde(0) << 16);

    // Stride dimension byte offset: bits 45–32 (14 bits)
    desc = desc | (mde(SBO) << 32);

    // Matrix base offset: bits 51–49 (3 bits)
    // This is valid for all swizzling modes except the no-swizzle mode.

    // Swizzle mode to be used: bits 63–62 (2 bits)
    desc = desc | (uint64_t(1) << 62);

    return desc;
  }
};


//======================================================================================================================
// MMA type
//======================================================================================================================
struct MMABase {};


// 5-th generation Tensor Core MMA instruction for m64nNk16fp16
// NOTE: F16 means that the `InputType` is either FP16 or BF16.
template <typename InputType, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t STAGE_COUNT>
struct MMA_M64_NN_K16_F16 : public MMABase {
  // Assert TILE_M
  static_assert(TILE_M % 64 == 0,
                "TILE_M must divisible by 64 "
                "because MMA requires 64 elements along M dimension");
  static_assert(TILE_M <= 128,
                "TILE_M must <= 128 "
                "because TMEM can support a maximum of 128 rows");

  // Assert TILE_N
  static_assert(TILE_N % 8 == 0,
                "TILE_N must divisible by 8 "
                "because each MMA core matrix requires 8 rows/cols");
  static_assert(TILE_N <= 256,
                "TILE_N must <= 256 "
                "because MMA can support a maximum of 256 cols");

  // Assert TILE_K
  static_assert(TILE_K % 16 == 0,
                "TILE_K must divisible by 16 "
                "because MMA requires 16 elements along K dimension");

  using MMAInputType = InputType;

  // Short form alias
  using MMASelfType = MMA_M64_NN_K16_F16;

  // MatDescType alias
  using MatDescType = MatDesc<MMASelfType>;

  // SMEMDataType alias
  using SMEMDataType = SMEMData<MMASelfType>;

  DEVICE_FUNC MMA_M64_NN_K16_F16(const SMEMDataType& smem)
    : smem(smem), md(MatDescType::get(smem)) {
  }

  static constexpr uint32_t PART_COUNT_M = MatDescType::PART_COUNT_M;
  static constexpr uint32_t PART_COUNT_K = MatDescType::PART_COUNT_K;

  static constexpr DEVICE_FUNC uint32_t tileSizeM() {
    return TILE_M;
  }

  static constexpr DEVICE_FUNC uint32_t tileSizeN() {
    return TILE_N;
  }

  static constexpr DEVICE_FUNC uint32_t tileSizeK() {
    return TILE_K;
  }

  static constexpr DEVICE_FUNC uint32_t stageCount() {
    return STAGE_COUNT;
  }

  DEVICE_FUNC void execute(const uint32_t stageIdx) {
    // Loop over parts of tile A and tile B
    #pragma unroll
    for (uint32_t k = 0; k < PART_COUNT_K; k++) {
      auto descB = md.descB(stageIdx * PART_COUNT_K + k);
      #pragma unroll
      for (uint32_t m = 0; m < PART_COUNT_M; m++) {
        uint32_t descIdxA = stageIdx * (PART_COUNT_M * PART_COUNT_K) + m * PART_COUNT_K + k;
        auto descA = md.descA(descIdxA);

        uint32_t rowOffset = (m * 64) << 16;
        PTX::mma(smem.tmemBaseHandle + rowOffset, descA, descB, INST_DESC);
      }
    }
  }

  // Marks the commit point for one or more sized batch of async MMA instructions.
  DEVICE_FUNC void commit(const uint32_t stageIdx) {
    PTX::commitMMABatch(smem.consumerMbar(stageIdx));
  }

 private:
  const SMEMDataType& smem;
  const MatDescType md;

  // Compute MMA instruction descriptor
  // Ref: https://docs.nvidia.com/cuda/archive/13.1.1/parallel-thread-execution/index.html#tcgen05-instruction-descriptor
  // For example, 68157456 = 000|00100|0|001000|00000000000010000
  static constexpr uint32_t INST_DESC = []() constexpr -> uint32_t {
    uint32_t desc = 0;

    // Set bits for dtype
    desc = desc | (1 << 4);

    // Set bits for atype and btype
    // NOTE: The default atype and btype are f16
    if constexpr(cuda::std::is_same<InputType, bf16_t>()) {
      desc = desc | ((1 << 10) + (1 << 7));
    }

    // Set bits for "Transpose B Matrix"
    desc = desc | (1 << 16);

    // Set bits for TILE_N cols
    desc = desc | ((TILE_N >> 3) << 17);

    // Set bits for 64 rows
    desc = desc | ((64 >> 4) << 24);

    return desc;
  }();
};


//======================================================================================================================
// GPU GEMM kernel
//======================================================================================================================
/*Features
Within 1 CTA:
1. Warp specialization
One or several warps are responsible for loading data from GMEM onto SMEM/TMEM - Producer warp(s)
The rest of warps issue async MMA instructions to Tensor Core to consume SMEM/TMEM - Consumer warp(s)

2. Software pipeline
Since MMA instructions are asynchronous, Consumer warp(s) can issue multiple MMA instructions before 
waiting for them to finish, so Consumer warp(s) need more than one tile of A and B.

To feed hungry Consumer warp(s), Producer warp(s) prefetch a number of A and B tiles from GMEM onto SMEM.
Consumer and Producer collaborate with each other via mbarriers to form a data processing pipeline.
*/
template <typename HostMMAType>
__device__ void GEMMFunction(const CUtensorMap& tmapA,
                             const CUtensorMap& tmapB,
                             const CUtensorMap& tmapD,
                             const uint32_t N,
                             const uint32_t K,
                             const uint32_t tileCountN,
                             const uint32_t tileCountK,
                             const uint32_t totalTiles) {
  using InputType = typename HostMMAType::MMAInputType;
  constexpr uint32_t TILE_M = HostMMAType::tileSizeM();
  constexpr uint32_t TILE_N = HostMMAType::tileSizeN();
  constexpr uint32_t TILE_K = HostMMAType::tileSizeK();
  constexpr uint32_t STAGE_COUNT = HostMMAType::stageCount();

  // MMA type
  // NOTE: Technically, we can use `HostMMAType` directly
  // but we explicitly create an alias so that IntelliSense can recognize it.
  using MMAType = MMA_M64_NN_K16_F16<InputType,
                                     TILE_M, TILE_N, TILE_K, STAGE_COUNT>;
  static_assert(cuda::std::is_same<MMAType, HostMMAType>(), "MMAType mismatch");

  // SMEM data type
  // NOTE: Technically, we can use `MMAType::SMEMDataType` alias instead
  // but we explicitly create another alias so that IntelliSense
  // can recognize it.
  using SMEMDataType = SMEMData<MMAType>;
  static_assert(
    cuda::std::is_same<SMEMDataType, typename MMAType::SMEMDataType>(),
    "SMEMDataType mismatch"
  );

  // SMEM buffer
  extern __shared__ char globalSMEM[];
  SMEMDataType& smem = *(reinterpret_cast<SMEMDataType*>(globalSMEM));
  MMAType mma(smem);

  // Compute warpIdx
  // If `blockDim.y != 1`, then `threadIdx.x` may not increase linearly 
  // within a warp. For example, `blockDim = (48, 2)`, then 
  // warp 1 will have the following threads 
  //   - (threadIdx.x=(32, ..., 47), threadIdx.y=(0)) and 
  //   - (threadIdx.x=(0, ..., 15), threadIdx.y=(1))
  // In this case, `threadIdx.x >> 5` produces either 1 or 0.
  // 
  // Get `threadIdx.x >> 5` from source lane 0 to ensure that `warpIdx`
  // is uniform across threads within a warp.
  auto warpIdx = __shfl_sync(0xFFFFFFFF, threadIdx.x >> 5, 0);

  //==================================
  // Warps 0-7: Prologue
  //==================================
  #define REMARK_GEMM_PROLOGUE
  if (warpIdx < 8) {
    // 
    // * Prologue (warps 0-7)
    // 

    // 1. Allocate a TMEM tile of 128 rows by 64 columns (128x64) for 
    // holding the resulting tile D later.
    if (warpIdx == 0) {
      // Since TILE_N = 64,
      // we need to allocate 64 cols
      PTX::tmemAlloc(&smem.tmemBaseHandle, TILE_N);

      // Release TMEM allocation permit
      PTX::tmemRelinquishAllocPermit();
    }

    // 2. Reset all elements of the TMEM tile to zeros

    // Warps 0-255:
    // Reset tile D in TMEM to zeros
    // tile D has 64 * 64 = 4096 elements
    // Each thread reset 16 elements
    // => Each warp reset 32 * 16 = 512 elements
    // => Require 4096 / 512 = 8 warps
    {
      // Ensure that all 256 threads have the same `smem.tmemBaseHandle` value
      PTX::sync256Threads();

      // Compute row offset
      uint32_t rowOffset = (warpIdx << 21) & 6291456; // 6291456 = 2^22 + 2^21;

      // Compute col offset
      uint32_t colOffset = (warpIdx * 8) & 32;

      // Compute TMEM handle for each warp
      uint32_t tmemHandle = smem.tmemBaseHandle + rowOffset + colOffset;

      // Execute TMEM reset
      // NOTE: untile now, we have not created any thread-level code branch,
      // all threads in every warp should have the same program counter.
      PTX::tmemReset(tmemHandle);

      // Ensure that all 256 threads have done resetting tile D in TMEM
      PTX::sync256Threads();
    }


    // 3. Init Producer mbarriers
    // Warp 1
    if (warpIdx == 1) {
      auto isLeaderLane = PTX::electLeaderLane();
      if (isLeaderLane) {
        auto producerMbars = smem.producerMbars;

        #pragma unroll
        for (int i = 0; i < STAGE_COUNT; i++) {
          PTX::mbarInit(producerMbars + i, 1);
        }
      }
    }


    // 4. Init Consumer mbarriers
    // Warp 2:
    else if (warpIdx == 2) {
      auto isLeaderLane = PTX::electLeaderLane();
      if (isLeaderLane) {
        auto consumerMbars = smem.consumerMbars;

        #pragma unroll
        for (int i = 0; i < STAGE_COUNT; i++) {
          PTX::mbarInit(consumerMbars + i, 1);
          // phaseParity: 0
        }

        // 5. Init final mbarrier
        PTX::mbarInit(&smem.finalMbar, 1);
        // phaseParity: 0

        // 6. Prime Consumer mbarriers
        #pragma unroll
        for (int i = 0; i < STAGE_COUNT; i++) {
          PTX::mbarArrive(consumerMbars + i);
          // phaseParity: 0 -> 1
        }
      }
    }
  }


  // Total blocks
  const uint32_t blocks = gridDim.x;

  // Initial tile idx
  uint32_t tileIdx = blockIdx.x;

  // mbarrier phase parity
  uint32_t phaseParity = 0;


  // Loop over all tiles of the resulting matrix D
  while (tileIdx < totalTiles) {
    // Compute dimensional tile indices
    uint32_t tileM = tileIdx / tileCountN;
    uint32_t tileN = tileIdx % tileCountN;

    uint32_t tileK = 0;
    uint32_t stageIdx = 0;

    if (warpIdx < 8) {
      // 
      // * Enter Wait State (warps 0-7)
      // 

      // 7. Arrive at the 1st CTA sync point
      PTX::syncCTAUnaligned();

      // 8. Arrive at the 2nd CTA sync point
      // syncCTAUnaligned();

      // 9. Wait for the final mbarrier to be satisfied
      PTX::mbarWait(&smem.finalMbar, phaseParity);


      // 
      // * Epilogue (warps 0-7)
      // 
      #define REMARK_GEMM_EPILOGUE
      // 10. Load the resulting TMEM tiles into registers. 
      // Since a TMEM tile has 64×64 elements, this is the key reason we need up to 8 warps: 
      // each warp loads a 16×32 sub-tile, and each thread holds 16 elements of the loaded sub-tile.

    } // Warps 0-7 code block


    //==================================
    // Warp 8: Producer
    //==================================
    else if (warpIdx == 8) {
      // 
      // * Prologue (Warp 8)
      // 

      // 1. Arrive at the 1st CTA sync point
      PTX::syncCTAUnaligned();

      auto isLeaderLane = PTX::electLeaderLane();

      const uint32_t txBytes = sizeof(SMEMDataType::StageDataTypeA) + 
                               sizeof(SMEMDataType::StageDataTypeB);

      // 
      // * Enter Copy Loop (Warp 8)
      //

      #define REMARK_GEMM_COPY_LOOP
      while (tileK < tileCountK) {
        // 2. Wait for Consumer mbarriers to be satisfied
        auto consumerMbarPtr = smem.consumerMbars + stageIdx;
        PTX::mbarWait(consumerMbarPtr, phaseParity);

        // 3. TMA 64x64xf16 tile A and tile B to SMEM
        // NOTE: Only leader lane executes TMA instructions
        if (isLeaderLane) {
          PTX::mbarArriveAndExpectTxBytes(consumerMbarPtr, txBytes);

          auto col = tileK * TILE_K;
          auto rowA = tileM * TILE_M;
          auto rowB = tileN * TILE_N;
          PTX::tmaCopyTile(smem.dataA(stageIdx), &tmapA, col, rowA, consumerMbarPtr);
          PTX::tmaCopyTile(smem.dataB(stageIdx), &tmapB, col, rowB, consumerMbarPtr);

          // 4. Satisfy Producer mbarriers
          auto producerMbarPtr = smem.producerMbars + stageIdx;
          PTX::mbarArrive(producerMbarPtr);
        }

        // Move to next tile K
        tileK++;

        // Move to next stage data or the first one
        auto nextStageIdx = stageIdx + 1;
        stageIdx = (nextStageIdx < STAGE_COUNT) ? nextStageIdx : 0;

        // Flip phase parity
        // Only fip it if stageIdx warps around
        phaseParity = (stageIdx == 0) ? (phaseParity ^ 1) : phaseParity;
      }


      // 
      // * Epilogue (Warp 8)
      // 

      // 5. Arrive at the 2nd CTA sync point
      // syncCTAUnaligned();

    } // Warp 8 code block


    //==================================
    // Warp 9: Consumer
    //==================================
    else if (warpIdx == 9) {
      // 
      // * Epilogue (Warp 9)
      // 

      // 1. Arrive at the 1st CTA sync point
      PTX::syncCTAUnaligned();

      // 2. Wait for the 1st Producer mbarrier to be satisfied
      // // NOTE: at this point, tileK is 0 and stageIdx is 0
      PTX::mbarWait(smem.producerMbars + stageIdx, phaseParity);

      // 3. MMA the 1st pair of 64x64 tile A and tile B
      // NOTE: Only leader lane executes MMA instructions
      auto isLeaderLane = PTX::electLeaderLane();
      if (isLeaderLane) {
        mma.execute(stageIdx);

        // 4. Satisfy the 1st Consumer mbarrier, 
        // which signifies that the 1st pair of tile A and tile B is consumed.
        mma.commit(stageIdx);
      }

      // Move to next stage
      tileK++;
      stageIdx++;

      // NOTE: at this point, if `tileK < tileCountK`, warp 9 enters MMA loop.
      // Otherwise, it satisfies the final mbarrier and finishes this iteration.

      // 
      // * Enter MMA Loop (Warp 9)
      // 

      #define REMARK_GEMM_MMA_LOOP
      while (tileK < tileCountK) {
        // 6. Wait for Producer mbarriers to be satisfied
        PTX::mbarWait(smem.producerMbars + stageIdx, phaseParity);

        // 7. MMA the next pair of tile A and tile B
        // NOTE: Only leader lane executes MMA instructions
        if (isLeaderLane) {
          mma.execute(stageIdx);

          // 8. Satisfy Consumer mbarriers
          mma.commit(stageIdx);
        }

        // Move to next tile K
        tileK++;

        // Move to next stage data or go back to the first one
        auto nextStageIdx = stageIdx + 1;
        stageIdx = (nextStageIdx < STAGE_COUNT) ? nextStageIdx : 0;

        // Flip phase parity
        // Only fip it if stageIdx warps around
        phaseParity = (stageIdx == 0) ? (phaseParity ^ 1) : phaseParity;
      }

      // 5. Satisfy the final mbarrier
      // 9. Satisfy the final mbarrier
      if (isLeaderLane) {
        PTX::mbarArrive(&smem.finalMbar);
      }

      // 
      // * Epilogue (Warp 9)
      // 

      // 10. Arrive at the 2nd CTA sync point
      // PTX::syncCTAAligned();

    } // Warp 9 code block

    // Finally, ensure that warps 0-9 in CTA has done
    // their jobs before starting next iteration.
    PTX::syncCTAAligned();

    // NOTE: 
    // Copy TMEM tiles to GMEM is done by the Epilogue in warps 0-7.
    // So at this point, this CTA has done computing this `tileIdx`.
    // It will move to next tile strided by `blocks`.
    tileIdx += blocks;
  }
}


//======================================================================================================================
// Host functions
//======================================================================================================================
template <typename ElemType, CUtensorMapSwizzle swizzleMode>
static CUtensorMap createTensorMap(ElemType *gmemPtr,
                            const uint32_t gmemRows,
                            const uint32_t gmemCols,
                            const uint32_t smemRows,
                            const uint32_t smemCols) {
  // tensorRank is the number of dimensions of the array.
  constexpr uint32_t tensorRank = 2;

  // NOTE: globalDim[0] must be the contiguous dimension.
  // For a row-major layout, globalDim[0] is the number of columns
  uint64_t globalDim[] = {gmemCols, gmemRows};

  // The `globalStrides[0]` is the number of bytes to traverse
  // from the first element of one row to the next row.
  // It must be a multiple of 16.
  uint64_t globalStrides[] = {gmemCols * sizeof(ElemType)};

  // The boxDim is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  uint32_t boxDim[] = {smemCols, smemRows};

  // per-dimension traversal stride, measured in elements.
  uint32_t elementStrides[] = {1, 1};

  CUtensorMapDataType tensorDataType;
  if constexpr (std::is_same<ElemType, int>()) {
    tensorDataType = CU_TENSOR_MAP_DATA_TYPE_INT32;
  } else if constexpr (std::is_same<ElemType, fp16_t>()) {
    tensorDataType = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if constexpr (std::is_same<ElemType, bf16_t>()) {
    tensorDataType = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if constexpr (std::is_same<ElemType, float>()) {
    tensorDataType = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else {
    fprintf(stderr,
            "ERROR %s:%d: createTensorMap does not support the provided DataType.\n",
            __FILE__, __LINE__);
    std::exit(1);
  }

  CUtensorMapL2promotion l2Promotion;
  if constexpr (swizzleMode == CU_TENSOR_MAP_SWIZZLE_NONE) {
    l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
  } else if constexpr (swizzleMode == CU_TENSOR_MAP_SWIZZLE_32B) {
    l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
  } else if constexpr (swizzleMode == CU_TENSOR_MAP_SWIZZLE_64B) {
    l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
  } else if constexpr (swizzleMode == CU_TENSOR_MAP_SWIZZLE_128B) {
    l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  } else {
    fprintf(stderr, "ERROR %s:%d: Unsupported swizzle mode %d", __FILE__, __LINE__,
            swizzleMode);
    std::exit(1);
  }

  // Create the tensor descriptor.
  CUtensorMap tensorMap;
  CUresult ret = cuTensorMapEncodeTiled(
      &tensorMap,        // CUtensorMap *tensorMap,
      tensorDataType,    // CUtensorMapDataType tensorDataType
      tensorRank,        // cuuint32_t tensorRank
      gmemPtr,           // void *globalAddress
      globalDim,         // const cuuint64_t *globalDim
      globalStrides + 1, // const cuuint64_t *globalStrides
      boxDim,            // const cuuint32_t *boxDim
      elementStrides,    // const cuuint32_t *elementStrides

      // Interleave patterns can be used to accelerate loading of values that
      // are less than 4 bytes long.
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,

      // Swizzling mode can be used to avoid shared memory bank conflicts.
      swizzleMode,

      // L2 Promotion can be used to widen the effect of a cache-policy to a
      // wider set of L2 cache lines.
      l2Promotion,

      // Any element that is outside of bounds will be set to zero by the TMA
      // transfer.
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  if (ret != CUDA_SUCCESS) {
    fprintf(stderr,
            "ERROR %s:%d: Failed to create CUtensorMap object: "
            "gmemRows=%d gmemCols=%d smemRows=%d smemCols=%d\n",
            __FILE__, __LINE__, gmemRows, gmemCols, smemRows, smemCols);
    const char *errName;
    cuGetErrorName(ret, &errName);
    const char *errMsg;
    cuGetErrorString(ret, &errMsg);
    fprintf(stderr, "ERROR %s:%d: %s: %s\n", __FILE__, __LINE__, errName, errMsg);
    std::exit(1);
  }

  printf("Successfully created CUtensorMap object for "
         "gmemRows=%d gmemCols=%d smemRows=%d smemCols=%d swizzleMode=%d\n", 
         gmemRows, gmemCols, smemRows, smemCols, swizzleMode);

  return tensorMap;
}

} // namespace CUDA


namespace GEMM {

static constexpr int TILE_M = 64;
static constexpr int TILE_N = 64;
static constexpr int TILE_K = 64;
static constexpr int STAGE_COUNT = 5;

using MMAFP16 = CUDA::MMA_M64_NN_K16_F16<fp16_t,
                                          TILE_M, TILE_N, TILE_K, STAGE_COUNT>;
using MMABF16 = CUDA::MMA_M64_NN_K16_F16<bf16_t,
                                          TILE_M, TILE_N, TILE_K, STAGE_COUNT>;

enum Mat {
  MatA = 0,
  MatB = 1,
  MatD = 2
};

struct GEMMConfig {
  CUtensorMap tmapA;
  CUtensorMap tmapB;
  CUtensorMap tmapD;
  uint32_t N;
  uint32_t K;
  uint32_t tileCountN;
  uint32_t tileCountK;
  uint32_t totalTiles;
  void* kernelFn;
  int smemBytes;
  int threads;
  int blocks;

  void** getKernelArgs() {
    kernelArgs[0] = &tmapA;
    kernelArgs[1] = &tmapB;
    kernelArgs[2] = &tmapD;
    kernelArgs[3] = &N;
    kernelArgs[4] = &K;
    kernelArgs[5] = &tileCountN;
    kernelArgs[6] = &tileCountK;
    kernelArgs[7] = &totalTiles;

    return kernelArgs;
  }

 private:
  void* kernelArgs[8];
};


extern "C" __global__ void GEMMFP16Kernel(
                          const __grid_constant__ CUtensorMap tmapA,
                          const __grid_constant__ CUtensorMap tmapB,
                          const __grid_constant__ CUtensorMap tmapD,
                          const uint32_t N,
                          const uint32_t K,
                          const uint32_t tileCountN,
                          const uint32_t tileCountK,
                          const uint32_t totalTiles) {
  CUDA::GEMMFunction<MMAFP16>(
    tmapA, tmapB, tmapD,
    N, K, tileCountN, tileCountK, totalTiles
  );
}

extern "C" __global__ void GEMMBF16Kernel(
                        const __grid_constant__ CUtensorMap tmapA,
                        const __grid_constant__ CUtensorMap tmapB,
                        const __grid_constant__ CUtensorMap tmapD,
                        const uint32_t N,
                        const uint32_t K,
                        const uint32_t tileCountN,
                        const uint32_t tileCountK,
                        const uint32_t totalTiles) {
  CUDA::GEMMFunction<MMABF16>(
    tmapA, tmapB, tmapD,
    N, K, tileCountN, tileCountK, totalTiles
  );
}


template <typename InputType, typename OutputType = float>
static void launch(
                    InputType* gmemPtrA,
                    InputType* gmemPtrB,
                    OutputType* gmemPtrD,
                    const uint32_t M,
                    const uint32_t N,
                    const uint32_t K
                  ) {
  // Compute GEMM config
  GEMMConfig config{};
  {
    config.tmapA = CUDA::createTensorMap<InputType, CU_TENSOR_MAP_SWIZZLE_128B>(
                                                gmemPtrA, M, K, TILE_M, TILE_K);
    config.tmapB = CUDA::createTensorMap<InputType, CU_TENSOR_MAP_SWIZZLE_128B>(
                                                gmemPtrB, N, K, TILE_N, TILE_K);
    config.tmapD = CUDA::createTensorMap<OutputType, CU_TENSOR_MAP_SWIZZLE_128B>(
                                                gmemPtrD, M, N, TILE_M, TILE_N/2);

    config.N = N;
    config.K = K;
    config.tileCountN = (N + TILE_N - 1) / TILE_N;
    config.tileCountK = (K + TILE_K - 1) / TILE_K;
    const uint32_t tileCountM = (M + TILE_M - 1) / TILE_M;
    config.totalTiles = tileCountM * config.tileCountN;

    if constexpr(std::is_same<InputType, fp16_t>()) {
      config.kernelFn = reinterpret_cast<void *>(GEMMFP16Kernel);
      config.smemBytes = sizeof(typename MMAFP16::SMEMDataType);
    } else {
      config.kernelFn = reinterpret_cast<void *>(GEMMBF16Kernel);
      config.smemBytes = sizeof(typename MMABF16::SMEMDataType);
    }

    // Compute threads and blocks
    // 10 warps
    config.threads = 10 * 32;

    int smCount = 64;
    cudaDeviceProp devProp;
    auto status = cudaGetDeviceProperties(&devProp, 0);
    if (status == cudaSuccess) {
      smCount = devProp.multiProcessorCount;
    }
    printf("SM count: %d\n", smCount);

    int blocksPerSm = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm,
                                                  config.kernelFn,
                                                  config.threads,
                                                  config.smemBytes);

    config.blocks = smCount * blocksPerSm;
    if (config.blocks > config.totalTiles) {
      config.blocks = config.totalTiles;
    }
    printf("blocks = %d threads = %d\n", config.blocks, config.threads);
  }

  // Set smemBytes
  CHECK_CUDA_ERROR(
    cudaFuncSetAttribute(
      config.kernelFn,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      config.smemBytes
    )
  );
  printf("smemBytes = %d\n", config.smemBytes);

  // Launch kernel
  CHECK_CUDA_ERROR(
    cudaLaunchKernel(
      config.kernelFn,
      dim3(config.blocks),
      dim3(config.threads),
      config.getKernelArgs(),
      config.smemBytes,
      0
    )
  );
}

} // namespace GEMM


template <typename T> std::string getTypeName() {
  constexpr std::string_view funcName = __PRETTY_FUNCTION__;
  constexpr auto eqPos = funcName.find("=") + 2;
  constexpr auto len = funcName.size() - eqPos - 1;
  constexpr auto rstr = funcName.substr(eqPos, len);
  constexpr auto endPos = rstr.find(";\n");
  if (endPos != std::string_view::npos) {
    return std::string(rstr.substr(0, endPos));
  }

  return std::string(rstr);
}

// A layout: K-major
// B layout: K-major
// C layout: N-major
template <typename OutputType, typename InputType>
std::vector<OutputType> cpuGemm(const std::vector<InputType> &A,
                                       const std::vector<InputType> &B,
                                       const int M, const int N, const int K) {
  std::vector<OutputType> cpuResult(M * N, 0);

  using ConvertToOutputTypeFn = OutputType (*)(InputType);
  ConvertToOutputTypeFn convertToFloatFn = nullptr;

  if constexpr (std::is_same<InputType, fp16_t>()) {
    convertToFloatFn = __half2float;
  } else if constexpr (std::is_same<InputType, bf16_t>()) {
    convertToFloatFn = __bfloat162float;
  } else {
    fprintf(stderr, "ERROR InputType = %s is unsupported!\n",
            getTypeName<InputType>().c_str());
    std::exit(1);
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      auto acc = OutputType(0);
      #pragma unroll
      for (int k = 0; k < K; k++) {
        // NOTE: Sinc B layout is K-major
        // K-dimension has a stride of 1
        // N-dimension has a stride of K
        // => Linear index = (k, j) x (1, K) = k + j * K
        acc += convertToFloatFn(A[i * K + k]) * convertToFloatFn(B[k + j * K]);
      }

      cpuResult[i * N + j] = acc;
    }
  }

  return cpuResult;
}

// A layout: K-major
// B layout: K-major
// C layout: N-major
template <typename OutputType, typename InputType>
std::vector<OutputType> gpuGemm(const std::vector<InputType> &A,
                                       const std::vector<InputType> &B,
                                       const uint32_t M,
                                       const uint32_t N,
                                       const uint32_t K) {
  std::vector<OutputType> gpuResult(M * N, 0);
  constexpr int elemBytes = sizeof(InputType);

  const int numBytesA = M * K * elemBytes;
  const int numBytesB = N * K * elemBytes;
  const int numBytesD = M * N * elemBytes;

  // Allocate device buffers
  InputType *gmemPtrA, *gmemPtrB;
  CHECK_CUDA_ERROR(
    cudaMalloc(&gmemPtrA, numBytesA)
  );
  CHECK_CUDA_ERROR(
    cudaMalloc(&gmemPtrB, numBytesB)
  );

  OutputType* gmemPtrD;
  CHECK_CUDA_ERROR(
    cudaMalloc(&gmemPtrD, numBytesD)
  );

  // Copy inputs
  CHECK_CUDA_ERROR(
    cudaMemcpy(gmemPtrA, A.data(), numBytesA, cudaMemcpyHostToDevice)
  );
  CHECK_CUDA_ERROR(
    cudaMemcpy(gmemPtrB, B.data(), numBytesB, cudaMemcpyHostToDevice)
  );

  GEMM::launch<InputType, OutputType>(gmemPtrA, gmemPtrB, gmemPtrD, M, N, K);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // Copy matrix D
  CHECK_CUDA_ERROR(
    cudaMemcpy(gpuResult.data(), gmemPtrD, numBytesD, cudaMemcpyDeviceToHost)
  );

  return gpuResult;
}

int main(int argc, char **argv) {
  printf("Start main\n");
  using OutputType = float;
  using InputType = fp16_t;

  constexpr uint32_t M = 512;
  constexpr uint32_t N = 1024;
  constexpr uint32_t K = 1024;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 0.1);

  std::vector<InputType> A(M * K, 0);
  for (auto &e : A) {
    e = InputType(dist(gen));
  }

  std::vector<InputType> B(N * K, 0);
  for (auto &e : B) {
    e = InputType(dist(gen));
  }

  // printf("CPU Compute GEMM\n");
  // auto cpuResult = cpuGemm<OutputType, InputType>(A, B, M, N, K);

  printf("GPU Compute GEMM\n");
  auto gpuResult = gpuGemm<OutputType, InputType>(A, B, M, N, K);
  fflush(stdout);
  fflush(stderr);

  return EXIT_SUCCESS;
}
