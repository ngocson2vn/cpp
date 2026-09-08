# Vectorized Load and Store
## ld.global.v4.f32
```cpp
__device__ void ld_v4_f32(float& v0, float& v1, float& v2, float& v3, const float* ptr) {
  asm volatile(
    "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
    : "=f"(v0), "=f"(v1), "=f"(v2), "=f"(v3)
    : "l"(ptr)
  );
}
```

## st.global.v4.f32
```cpp
__device__ void st_v4_f32(float* ptr, float v0, float v1, float v2, float v3) {
  asm volatile(
    "st.global.v4.f32 [%0], {%1, %2, %3, %4};"
    :
    : "l"(ptr), "f"(v0), "f"(v1), "f"(v2), "f"(v3)
  );
}
```

## Instruction-Level Parallelism
```mlir
	add.f32 	%r25, %r17, %r21;
	add.f32 	%r26, %r18, %r22;
	add.f32 	%r27, %r19, %r23;
	add.f32 	%r28, %r20, %r24;
```

Each CUDA core has a **FP32 Fused Multiply-Add (FMA)** pipeline, which consists of several stages s1-s5. <br/>
For each clock cycle, the warp scheduler will schedule instructions to each CUDA core's FP32 FMA pipeline as follows:

```
   | s1 | s2 | s3 | s4 | s5 |
---|----|----|----|----|----|
c1 | i1 |    |    |    |    |
---|----|----|----|----|----|
c2 | i2 | i1 |    |    |    |
---|----|----|----|----|----|
c3 | i3 | i2 | i1 |    |    |
---|----|----|----|----|----|
c4 | i4 | i3 | i2 | i1 |    |
---|----|----|----|----|----|
c5 | i5 | i4 | i3 | i2 | i1 |
---|----|----|----|----|----|
```
