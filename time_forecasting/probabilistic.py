import torch
import torch.nn as nn
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================
# Define Probabilistic Model
# ================================
class ProbabilisticTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)  # Mean of Gaussian
        self.fc_std = nn.Linear(hidden_dim, output_dim)  # Std deviation of Gaussian

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        mean = self.fc_mean(lstm_out)
        std = torch.exp(self.fc_std(lstm_out))  # Ensure std is positive
        return mean, std


# Pyro Model Wrapper
class PyroProbabilisticModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def model_fn(self, x, y=None):
        pyro.module("model", self.model)  # Register parameters in Pyro
        with pyro.plate("data", x.size(0)):
            mean, std = self.model(x)
            if y is not None:
                pyro.sample("obs", dist.Normal(mean, std).to_event(2), obs=y)
            else:
                pyro.sample("obs", dist.Normal(mean, std).to_event(2))

    def guide_fn(self, x, y=None):
        pyro.module("model", self.model)


# ================================
# Generate Synthetic Data
# ================================
def generate_synthetic_data(seq_len, input_dim, output_dim, num_samples):
    inputs = np.random.randn(num_samples, seq_len, input_dim)
    weights = np.random.randn(input_dim, output_dim)
    outputs = np.einsum(
        "bij,jk->bik", inputs, weights
    )  # Simulate linear transformation
    return inputs, outputs


seq_len = 50
input_dim = 10
hidden_dim = 64
output_dim = 5
num_samples = 1000

inputs, outputs = generate_synthetic_data(seq_len, input_dim, output_dim, num_samples)
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
outputs_tensor = torch.tensor(outputs, dtype=torch.float32)

# Split into train and validation sets
train_split = int(0.8 * num_samples)
train_inputs, val_inputs = inputs_tensor[:train_split], inputs_tensor[train_split:]
train_outputs, val_outputs = outputs_tensor[:train_split], outputs_tensor[train_split:]

train_dataset = TensorDataset(train_inputs, train_outputs)
val_dataset = TensorDataset(val_inputs, val_outputs)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Instantiate model
base_model = ProbabilisticTimeSeriesModel(input_dim, hidden_dim, output_dim).to(device)
pyro_model = PyroProbabilisticModel(base_model)

# SVI Optimizer and Loss
optimizer = Adam({"lr": 0.001})
svi = SVI(pyro_model.model_fn, pyro_model.guide_fn, optimizer, loss=Trace_ELBO())

# ================================
# Train the Model
# ================================
num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    pyro.clear_param_store()
    train_loss = 0.0
    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        train_loss += svi.step(inputs, targets)

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation
    val_loss = 0.0
    for batch in val_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        val_loss += svi.evaluate_loss(inputs, targets)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
    )

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# ================================
# Evaluate the Model
# ================================
# Test on a single sequence
test_idx = 0
test_input = val_inputs[test_idx : test_idx + 1].to(device)
test_target = val_outputs[test_idx : test_idx + 1].cpu().numpy()

base_model.eval()
with torch.no_grad():
    mean, std = base_model(test_input)

# Plot predicted mean and uncertainty
mean = mean.cpu().numpy()
std = std.cpu().numpy()

plt.figure(figsize=(10, 6))
for dim in range(output_dim):
    plt.plot(mean[0, :, dim], label=f"Predicted Mean Dim {dim}")
    plt.fill_between(
        np.arange(seq_len),
        mean[0, :, dim] - 2 * std[0, :, dim],
        mean[0, :, dim] + 2 * std[0, :, dim],
        alpha=0.3,
        label=f"Uncertainty Dim {dim}",
    )
    plt.plot(test_target[0, :, dim], linestyle="dashed", label=f"True Dim {dim}")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("Predicted Mean and Uncertainty")
plt.legend()
plt.grid(True)
plt.show()
