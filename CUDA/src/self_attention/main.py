import math
import torch
torch.set_default_device("cuda")

# T (tokens)   = 4
# H (heads)    = 32
# D (head dim) = 128
Q = torch.rand(5, 32, 128, dtype=torch.float32)

K = torch.rand(5,  8, 128, dtype=torch.float32)
V = torch.rand(5,  8, 128, dtype=torch.float32)

#================================================================================
# Compute Attention Scores
# Formula: S[h] = torch.matmul(Q[:, h, :], K[:, h//4, :].T) / math.sqrt(128)
# Note that Q[:, h, :] and K[:, h//4, :] are 2D tensors
#================================================================================
S = torch.empty(32, 5, 5, dtype=torch.float32)

# Make heads=32 become batch_size in bmm
Qb = Q.permute(1, 0, 2).contiguous()            # [32, 5, 128]

# Repeat each head row 4 times
# Interleave means repeating row r 4 times,
# then repeating row r+1 4 times, and so on.
# 
Kr = K.repeat_interleave(4, dim=1)              # [5, 32, 128]

# Make heads=32 become batch_size 
# and other dims be compatible with Qb
Kb = Kr.permute(1, 2, 0).contiguous()           # [32, 128, 5]

# Batch matmul
S = torch.bmm(Qb, Kb) / math.sqrt(128)

print(S.shape, S.stride(), S.dtype, S.device)


#================================================================================
# Compute Causal Mask
# Formula: S[h, i, j] = -torch.inf if j > i
#================================================================================
# for h in range(32):
#   for i in range(5):
#     for j in range(5):
#       if j > i:
#         S[h, i, j] = -torch.inf

mask = torch.ones(5, 5, dtype=torch.bool).triu(diagonal=1)
S.masked_fill_(mask, -torch.inf)
print("Attention Scores S:")
print(S)
print()

#================================================================================
# Compute Softmax
#================================================================================
A = S.softmax(dim=-1)
print("Softmax A:")
print(A)

#================================================================================
# Compute Weighted Sum of Values
# Formula: O[i, h] = Σⱼ A[h, i, j] · V[j, h//4]               # [5, 32, 128]
#================================================================================
