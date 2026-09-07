# Convert Address
## cvta.to.global.u64
```mlir
.visible .entry add_vectors(
	.param .u64 .ptr .align 1 add_vectors_param_0,
	.param .u64 .ptr .align 1 add_vectors_param_1,
	.param .u64 .ptr .align 1 add_vectors_param_2,
	.param .u32 add_vectors_param_3
)
{
	.reg .pred 	%p<6>;
	.reg .b32 	%r<44>;
	.reg .b64 	%rd<18>;

  ld.param.b64 	%rd7, [add_vectors_param_0];
  cvta.to.global.u64 	%rd3, %rd7;

  // Omit for brevity
}
```

`cvta.to.global.u64 	%rd3, %rd7` takes the 64-bit generic pointer in `%rd7` and explicitly converts it into a `.global` state space pointer, storing the result in `%rd12`.

### Why the Conversion is Needed

Starting with the Fermi architecture (PTX ISA 2.0), CUDA introduced the **Generic Address Space**. A single generic pointer can point to `.global` (device memory), `.shared` (block memory), or `.local` (thread memory) memory.

While the GPU *can* use generic load/store instructions (e.g., `ld.u32`) to read from a generic pointer, explicitly converting it to a `.global` pointer is done for a few key reasons:

* **State-Space Specific Instructions:** Converting the pointer allows the compiler to use `.global`-specific load and store instructions (like `ld.global.f32`).
* **Cache Control Operations:** Only state-space-specific instructions support advanced cache operators. For example, if you want to bypass the L1 cache and read directly from L2 (using `ld.global.cg`), the pointer *must* be in the `.global` state space.
* **Compiler Optimizations:** By tagging the pointer as strictly `.global`, the PTX compiler (and subsequently `ptxas`, which compiles PTX to hardware SASS code) can make better optimization guarantees. It assures the compiler that this pointer will never alias with shared or local memory.

### Under the Hood (Hardware Reality)

At the hardware SASS level, the generic address space is constructed by mapping `.shared` and `.local` memory into specific "windows" of the 64-bit unified virtual address space.

Because standard global memory addresses map 1:1 with generic addresses, `cvta.to.global.u64` usually ends up being a **no-op** (or just a simple register move, like `MOV`) when compiled down to the final machine code. However, it remains structurally required at the PTX level to satisfy the type system and unlock downstream compiler optimizations.


## cvta.to.shared.u64
```mlir
// 1. You have a 64-bit generic pointer (occupies 2 registers or 64-bit width)
//    This might point to Shared Memory, but it's in the 64-bit format.
.reg .u64 %generic_ptr; 

// 2. Convert it to a 32-bit Shared Memory Offset
//    This instruction checks if the pointer is valid shared memory, 
//    strips the high bits, and gives you the compact 32-bit offset.
.reg .u32 %shared_offset;
cvta.to.shared.u64 %shared_offset, %generic_ptr;

// 3. Use the 32-bit offset for the load (Saves register pressure!)
ld.shared.f32 %f1, [%shared_offset]; 
```
