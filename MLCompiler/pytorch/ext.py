import torch
def convert_to_bytes(tensor: torch.Tensor):
  if tensor.is_nested():
    rows = tensor.cpu().contiguous().unbind(0)
    return [bytes(t.tolist()) for t in rows]
  else:
    return [bytes()]
