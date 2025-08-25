import dis
import torch

ok = torch.cuda.is_available()
print(f"torch.cuda.is_available: {ok}")

def torch_cuda():
  a = torch.randn(3, 3).cuda()
  print(a)

dis.dis(torch_cuda)
