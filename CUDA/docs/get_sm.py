import torch

major, minor = torch.cuda.get_device_capability()
print(f"sm_{major}{minor}")

