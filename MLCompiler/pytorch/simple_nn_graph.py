import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network class
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(self.output(x))
        return x
    
simple = SimpleNN()
from torch.fx import symbolic_trace
gm: torch.fx.GraphModule = symbolic_trace(simple)

print(gm.graph)
