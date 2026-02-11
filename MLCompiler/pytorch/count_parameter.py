import torch
from torch import nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden_layer = nn.Linear(2, 2)
        self.output_layer = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.sigmoid(self.output_layer(x))
        return x
    
model = SimpleNN()

print("Parameters:")
for name, p in model.named_parameters():
    print(f"{name}: shape={p.shape}, dtype={p.dtype}")

print()
pcount = sum(p.numel() for p in model.parameters())
print(f"Parameter count: {pcount}")
