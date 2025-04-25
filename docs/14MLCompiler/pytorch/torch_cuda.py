import dis

def torch_cuda():
  import torch
  a = torch.randn(3, 3).cuda()
  print(a)

dis.dis(torch_cuda)
