import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# XOR dataset
# -----------------------
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=torch.float32)

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
], dtype=torch.float32)


# -----------------------
# Neural Network
# -----------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),   # more capacity than 4 neurons
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


model = SimpleNet()

# -----------------------
# Training setup
# -----------------------
optimizer = optim.Adam(model.parameters(), lr=0.01)  # FIXED learning rate
loss_fn = nn.BCELoss()

# -----------------------
# Training loop
# -----------------------
for epoch in range(2000):
    optimizer.zero_grad()

    pred = model(X)
    loss = loss_fn(pred, y)

    loss.backward()
    optimizer.step()

    # print progress (important for debugging)
    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

print("\nTraining complete")

# -----------------------
# Verify training worked
# -----------------------
with torch.no_grad():
    print("\nModel predictions on XOR:")
    print(model(X))


# -----------------------
# Export to ONNX
# -----------------------
dummy_input = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"}
    }
)

print("\nModel exported to model.onnx")