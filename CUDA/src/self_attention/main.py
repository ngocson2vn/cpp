import torch
torch.set_default_device("cuda")

Q = torch.rand(5, 32, 128, dtype=torch.float32)
K = torch.rand(5,  8, 128, dtype=torch.float32)
V = torch.rand(5,  8, 128, dtype=torch.float32)

# Attention Score
S = torch.empty(32, 5, 5, dtype=torch.float32, device="cuda:0")
Qb = Q.permute(1, 0, 2)            # [32, 5, 128]
Kr = K.repeat_interleave(4, dim=1) # [5, 32, 128]
Kb = Kr.permute(1, 2, 0)           # [32, 128, 5]
S = torch.bmm(Qb, Kb)
print(S.shape, S.stride(), S.dtype, S.device)
