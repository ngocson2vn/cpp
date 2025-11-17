# References
https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html

https://www.ibiblio.org/gferg/ldp/GCC-Inline-Assembly-HOWTO.html

https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html

```C++
template <int32_t scaleD = 1, int32_t scaleA = 1, int32_t scaleB = 1, int32_t tnspA = 0, int32_t tnspB = 0>
struct MMA_64x64x16_F32F16F16_SS
{
  __device__ static void
  fma(
      float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
      float& d08, float& d09, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
      float& d16, float& d17, float& d18, float& d19, float& d20, float& d21, float& d22, float& d23,
      float& d24, float& d25, float& d26, float& d27, float& d28, float& d29, float& d30, float& d31,
      uint64_t const& desc_a,
      uint64_t const& desc_b)
  {
    asm volatile(
      "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        "  %8,  %9, %10, %11, %12, %13, %14, %15,  "
        " %16, %17, %18, %19, %20, %21, %22, %23,  "
        " %24, %25, %26, %27, %28, %29, %30, %31 },"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
      "}\n"
        : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
          "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
          "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
          "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31)
        :  "l"(desc_a), "l"(desc_b),
           "n"(scaleD), "n"(scaleA), "n"(scaleB), "n"(tnspA), "n"(tnspB)
    );
  }
};
```
`"+f"(d00)` tells the compiler that `d00` is a read-write floating operand. The compiler will ensure a value is available in a floating-point register for the inline PTX, and that the final value produced by the PTX is written back to `d00`.
- The constraint `"+f"`:
  - `f` requests a floating-point register class for the operand.
  - `+` marks it as read-write: the asm reads the current value and writes the new value.
- What happens around the inline asm:
  - If `d00` currently lives in memory (e.g., spilled), the compiler will insert loads to materialize its value into a register before the asm block.
  - The compiler binds that register to operand `%0` (or whichever position corresponds to `d00`) for the PTX instruction.
  - After the asm executes, the compiler treats the operand as updated. If `d00` must reside in memory afterward (e.g., ABI or register pressure), it will insert a store from the register back to the variable.
- Important nuance for CUDA/NVPTX:
  - You don’t see explicit `ld`/`st` for `d00` inside your inline assembly; those are handled by the compiler around the asm if needed.
  - If `d00` already resides in a register and stays live in a register, the compiler may avoid any memory traffic entirely—no loads/stores are necessary beyond the register binding.
  - The inline PTX instruction (e.g., `wgmma.mma_async... { %0, ... }`) reads the accumulator inputs from those registers and overwrites them with the results.