import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ================================
# Temporal Fusion Transformer Components
# ================================


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_ = self.layer1(x)
        x_ = torch.relu(x_)
        x_ = self.layer2(x_)
        gated_x = torch.sigmoid(self.gate(x_)) * x_
        return self.layer_norm(self.dropout(gated_x) + x)


class TemporalFusionTransformer(nn.Module):
    def __init__(
        self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 500, model_dim)
        )  # Max sequence length = 500
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        self.gate = GatedResidualNetwork(model_dim, model_dim, model_dim, dropout)
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        x = self.transformer_encoder(x)
        x = self.gate(x)
        return self.fc_out(x)


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


# Prepare Data
seq_len = 50
input_dim = 10
model_dim = 64
num_heads = 4
num_layers = 2
output_dim = 5
num_samples = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputs, outputs = generate_synthetic_data(seq_len, input_dim, output_dim, num_samples)
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
outputs_tensor = torch.tensor(outputs, dtype=torch.float32)

# Split data into training and validation sets
train_split = int(0.8 * num_samples)
train_inputs, val_inputs = inputs_tensor[:train_split], inputs_tensor[train_split:]
train_outputs, val_outputs = outputs_tensor[:train_split], outputs_tensor[train_split:]

train_dataset = TensorDataset(train_inputs, train_outputs)
val_dataset = TensorDataset(val_inputs, val_outputs)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ================================
# Instantiate Model
# ================================

model = TemporalFusionTransformer(
    input_dim, model_dim, num_heads, num_layers, output_dim
).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ================================
# Train the Model
# ================================
num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        train_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            val_loss += loss.item()

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
plt.title("TFT Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# ================================
# Evaluate the Model
# ================================
# Test on a single sequence from validation set
test_idx = 0
test_input = val_inputs[test_idx : test_idx + 1].to(device)
test_target = val_outputs[test_idx : test_idx + 1].cpu().numpy()

model.eval()
with torch.no_grad():
    test_prediction = model(test_input).cpu().numpy()

# Plot true vs predicted output for each dimension
plt.figure(figsize=(15, 8))
for dim in range(output_dim):
    plt.plot(test_prediction[0, :, dim], label=f"Predicted Dim {dim}")
    plt.plot(test_target[0, :, dim], linestyle="dashed", label=f"True Dim {dim}")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("True vs Predicted Outputs for Test Sequence")
plt.legend()
plt.grid(True)
plt.show()
