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

# Generate synthetic data (e.g., for binary classification)
torch.manual_seed(42)  # For reproducibility
num_samples = 100
X = torch.randn(num_samples, 2)  # 100 samples with 2 features
y = torch.where(X[:, 0] + X[:, 1] > 0, torch.ones(num_samples, 1), torch.zeros(num_samples, 1)).float()  # Simple rule: sum > 0 -> 1, else 0

# Create model, loss function, and optimizer
model = SimpleNN()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    test_input = torch.tensor([[0.5, 0.3]])
    prediction = model(test_input)
    print(f'Prediction for [0.5, 0.3]: {prediction.item():.4f}')