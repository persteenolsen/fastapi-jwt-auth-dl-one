import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# XOR dataset
# -----------------------
# Inputs: all combinations of two binary values
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=torch.float32)

# Targets: XOR truth table
# 0 XOR 0 = 0
# 0 XOR 1 = 1
# 1 XOR 0 = 1
# 1 XOR 1 = 0
y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
], dtype=torch.float32)


# -----------------------
# Neural Network
# -----------------------
# Define a small feedforward neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Sequential = layers applied one after another
        self.net = nn.Sequential(
            nn.Linear(2, 8),   # Layer 1: 2 inputs → 8 neurons (hidden layer)
            nn.ReLU(),         # Activation: introduces non-linearity
            nn.Linear(8, 1),   # Layer 2: 8 → 1 output neuron
            nn.Sigmoid()       # Squashes output to range [0,1] (probability)
        )

    # Defines forward pass (how input flows through network)
    def forward(self, x):
        return self.net(x)


# Create model instance
model = SimpleNet()

# -----------------------
# Training setup
# -----------------------
# Adam optimizer updates weights using gradients
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Binary Cross Entropy loss (for binary classification)
loss_fn = nn.BCELoss()

# -----------------------
# Training loop
# -----------------------
for epoch in range(2000):

    # Clear old gradients (PyTorch accumulates gradients by default)
    optimizer.zero_grad()

    # Forward pass: compute predictions from inputs
    pred = model(X)

    # Compute loss: how far predictions are from true values
    loss = loss_fn(pred, y)

    # Backward pass: compute gradients (∂loss/∂weights)
    loss.backward()

    # Update weights using optimizer
    optimizer.step()

    # Print progress every 200 epochs (useful for debugging)
    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

print("\nTraining complete")

# -----------------------
# Verify training worked
# -----------------------
# Disable gradient tracking (faster + no memory overhead)
with torch.no_grad():
    print("\nModel predictions on XOR:")
    print(model(X))  # Should be close to [0, 1, 1, 0]


# -----------------------
# Export to ONNX
# -----------------------
# Dummy input is required to trace model structure
dummy_input = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

# Export trained model to ONNX format (portable to other frameworks)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],     # name of input node
    output_names=["output"],   # name of output node
    dynamic_axes={             # allows variable batch sizes
        "input": {0: "batch"},
        "output": {0: "batch"}
    }
)

print("\nModel exported to model.onnx")
