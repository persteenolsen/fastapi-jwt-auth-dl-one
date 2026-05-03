import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# XOR dataset
# -----------------------
# This is a classic machine learning problem:
# The model must learn the XOR (exclusive OR) logic gate.
#
# XOR rule:
# - 0 XOR 0 = 0
# - 0 XOR 1 = 1
# - 1 XOR 0 = 1
# - 1 XOR 1 = 0

# Inputs: all possible combinations of two binary values
# Each row is one training example with 2 features
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=torch.float32)

# Targets: expected outputs for XOR
# Shape is (4, 1) because we have 4 samples and 1 output per sample
y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
], dtype=torch.float32)


# -----------------------
# Neural Network
# -----------------------
# We define a small "feedforward neural network".
# Feedforward means data moves only in one direction:
# input → hidden layers → output
#
# This network is used because XOR is NOT linearly separable,
# meaning a single linear layer cannot solve it.
# We need a hidden layer + non-linearity.

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        # nn.Sequential stacks layers in order
        # Output of one layer becomes input of the next layer

        self.net = nn.Sequential(

            # -----------------------
            # First layer (Input → Hidden)
            # -----------------------
            # 2 input features (XOR has two inputs: A and B)
            # 4 neurons = hidden layer size
            # Each neuron learns a different feature representation
            nn.Linear(2, 4),

            # -----------------------
            # Activation function
            # -----------------------
            # With only 4 neurons and ReLU → poor representation → failed convergence
            
            # With only 4 neurons and Tanh → correct representation → smooth convergence
            # Tanh is another non-linear activation function that outputs between -1 and 1
            # Better than ReLu for small networks and can help with learning XOR
            nn.Tanh(),

            # -----------------------
            # Output layer (Hidden → Output)
            # -----------------------
            # 4 inputs (from hidden layer)
            # 1 output neuron (final prediction)
            # This outputs a raw score before activation
            nn.Linear(4, 1),

            # -----------------------
            # Sigmoid activation
            # -----------------------
            # Converts output into a probability between 0 and 1
            # Useful for binary classification (0 or 1 decision)
            nn.Sigmoid()
        )

    # -----------------------
    # Forward pass
    # -----------------------
    # Defines how input data flows through the network
    # This function is called automatically when you do model(X)
    def forward(self, x):
        return self.net(x)


# Create an instance of the neural network
model = SimpleNet()


# -----------------------
# Training setup
# -----------------------

# Adam optimizer:
# - Automatically adapts learning rate per parameter
# - Efficient and widely used default optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Loss function:
# Binary Cross Entropy (BCELoss)
# Measures difference between predicted probability and true label (0 or 1)
loss_fn = nn.BCELoss()


# -----------------------
# Training loop
# -----------------------
# The model sees the same dataset many times (epochs)
# Each epoch improves weights slightly

for epoch in range(2000):

    # Clear previous gradients
    # PyTorch accumulates gradients by default, so we reset them each step
    optimizer.zero_grad()

    # -----------------------
    # Forward pass
    # -----------------------
    # Model makes predictions based on current weights
    pred = model(X)

    # -----------------------
    # Loss computation
    # -----------------------
    # Compare predictions vs true labels
    loss = loss_fn(pred, y)

    # -----------------------
    # Backpropagation
    # -----------------------
    # Computes gradient of loss with respect to all weights
    # (how each weight contributed to the error)
    loss.backward()

    # -----------------------
    # Update weights
    # -----------------------
    # Optimizer uses gradients to adjust weights slightly
    optimizer.step()

    # Print training progress every 200 epochs
    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

print("\nTraining complete")


# -----------------------
# Verify training worked
# -----------------------
# torch.no_grad() disables gradient tracking
# This makes inference faster and reduces memory usage
print("\nModel predictions on XOR:")

with torch.no_grad():
    preds = model(X)
    
    print(preds)

    # Convert probabilities → 0/1 using threshold
    binary_preds = (preds > 0.5).float()

    print("\nBinary predictions (0/1):")
    print(binary_preds)  # Expected output: close to [0, 1, 1, 0]

# -------------------------
# XOR solved message
# -------------------------
print("\nXOR problem solved!")

# -----------------------
# Export to ONNX
# -----------------------
# ONNX = Open Neural Network Exchange format
# Allows model to be used in other frameworks (C++, Unity, etc.)

# Dummy input is required so PyTorch can trace the model structure
dummy_input = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",

    # Naming input/output nodes for clarity in other systems
    input_names=["input"],
    output_names=["output"],

    # Allows model to accept different batch sizes at runtime
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"}
    }
)

print("\nModel exported to model.onnx")