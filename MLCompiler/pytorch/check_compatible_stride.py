def is_compatible_stride(base_stride: list[int], other_stride: list[int]):
    if len(other_stride) != len(base_stride):
        return False
    if sum(other_stride) != sum(base_stride):
        return False
    indices = set[int]()
    for s in other_stride:
        try:
            idx = base_stride.index(s)
            indices.add(idx)
        except ValueError:
            pass
    if len(indices) != len(base_stride):
        return False
    return True

base_stride = [7936, 4, 31, 64]
permuted_stride   = [7936, 31, 4, 64]

if is_compatible_stride(base_stride, permuted_stride):
    print(f"OK: {permuted_stride} is compatible with {base_stride}")
else:
    print(f"NG: {permuted_stride} is NOT compatible with {base_stride}")