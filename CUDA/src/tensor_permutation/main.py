import torch

t1 = torch.rand(5, 32, 128, dtype=torch.float32)
print(f"t1.shape = {t1.shape}")
print(f"t1.strides = {t1.stride()}")
print(f"t1 linear index = i * {t1.stride(0)} + j * {t1.stride(1)} + k * {t1.stride(2)}")
# Linear index = i * 4096 + j * 128 + k * 1
print()

# P: (i, j, k) -> (k, j, i)
# The permuation P maps (i, j, k) to (k, j, i)
# This means t1[i, j, k] == t2[k, j, i]
# To realize this mapping, we must permute both shape and strides of t1:
print("P: t1[i, j, k] -> t2[k, j, i]")
perm = [2, 1, 0]
t2 = t1.as_strided(
  size=[t1.size(p) for p in perm],
  stride=[t1.stride(p) for p in perm],
)
print(f"t2.shape = {t2.shape}")
print(f"t2.strides = {t2.stride()}")
print(f"t2 linear index = k * {t2.stride(0)} + j * {t2.stride(1)} + i * {t2.stride(2)}")
print()

# Re-use the same buffer
# Non-contiguous strides
t2 = t1.permute(2, 1, 0)

# Create new buffer with contiguous strides
t3 = t1.permute(2, 1, 0).contiguous()