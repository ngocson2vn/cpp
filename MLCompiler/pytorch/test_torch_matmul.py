import torch

low, high = 0, 1 #range of uniform distribution
M, N, K = 512, 256, 1024
A = torch.distributions.uniform.Uniform(low, high).sample([M, K]).cuda()
B = torch.distributions.uniform.Uniform(low, high).sample([N, K]).cuda()
print(f"A: shape={A.shape} dtype={A.dtype} {A}")
print(f"B: shape={B.shape} dtype={B.dtype} {B}")

torch.set_float32_matmul_precision("highest")
C = torch.matmul(A, B.T)
print(f"C: shape={C.shape} dtype={C.dtype} {C}")

count = 0
MAX_COUNT = 10
for i in range(C.shape[0]):
  for j in range(C.shape[1]):
    print(f"C[{i}, {j}] = {C[i, j]}")
    count += 1
    if count > MAX_COUNT:
      break
  if count > MAX_COUNT:
    break