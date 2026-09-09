# Bypass L1 With `.release` And `.acquire` Memory Semantic Qualifiers

## Memory Semantic Qualifiers
Read [../../docs/PTX_memory_semantic_qualifiers.md](../../docs/PTX_memory_semantic_qualifiers.md)

## Producer
```mlir
st.global.release.gpu.b8 [%0], %1;
```

## Consumer
```mlir
ld.global.acquire.gpu.b8 %0, [%1];
```
