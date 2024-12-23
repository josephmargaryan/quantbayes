import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transformer Model
class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 500, model_dim)
        )  # Max seq length = 500
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        x = self.transformer_encoder(x)
        return self.fc(x)


# Generate synthetic data
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

inputs, outputs = generate_synthetic_data(seq_len, input_dim, output_dim, num_samples)
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
outputs_tensor = torch.tensor(outputs, dtype=torch.float32)

dataset = TensorDataset(inputs_tensor, outputs_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate model
model = TransformerTimeSeriesModel(
    input_dim, model_dim, num_heads, num_layers, output_dim
).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
losses = []
# ================================
# Add Visualizations to Analyze Model Performance
# ================================

# Split data into training and validation sets
train_split = int(0.8 * num_samples)
train_inputs, val_inputs = inputs_tensor[:train_split], inputs_tensor[train_split:]
train_outputs, val_outputs = outputs_tensor[:train_split], outputs_tensor[train_split:]

train_dataset = TensorDataset(train_inputs, train_outputs)
val_dataset = TensorDataset(val_inputs, val_outputs)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Validation Loss Tracking
val_losses = []

# Updated Training Loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        predictions = model(inputs)

        # Compute loss
        loss = criterion(predictions, targets)
        train_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
    )

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), losses, label="Train Loss", marker="o")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# ================================
# Predictions vs Ground Truth
# ================================
# Evaluate on a single test sequence
test_idx = 0
test_input = val_inputs[test_idx : test_idx + 1].to(device)
test_target = val_outputs[test_idx : test_idx + 1].numpy()

model.eval()
with torch.no_grad():
    prediction = model(test_input).cpu().numpy()

# Plot true vs predicted output for each dimension
plt.figure(figsize=(15, 8))
for dim in range(output_dim):
    plt.plot(prediction[0, :, dim], label=f"Predicted Dim {dim}")
    plt.plot(test_target[0, :, dim], linestyle="dashed", label=f"True Dim {dim}")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("True vs Predicted Outputs for Test Sequence")
plt.legend()
plt.grid(True)
plt.show()

# ================================
# Error Heatmap
# ================================
# Compute absolute errors for the test set
test_inputs, test_outputs = val_inputs.to(device), val_outputs.numpy()
with torch.no_grad():
    test_predictions = model(test_inputs).cpu().numpy()

error_matrix = np.abs(test_outputs - test_predictions).mean(
    axis=0
)  # Mean error per sequence step

plt.figure(figsize=(12, 6))
plt.imshow(error_matrix.T, aspect="auto", cmap="viridis", origin="lower")
plt.colorbar(label="Absolute Error")
plt.xlabel("Time Step")
plt.ylabel("Output Dimension")
plt.title("Error Heatmap Across Time Steps and Output Dimensions")
plt.show()


# ================================
# (Optional) Attention Weight Visualization
# ================================
# Modify the model to return attention weights
class TransformerTimeSeriesModelWithAttention(TransformerTimeSeriesModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, return_attention=False):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        if return_attention:
            attentions = []
            for layer in self.transformer_encoder.layers:
                x, attention = layer(x, return_attention=True)
                attentions.append(attention)
            return self.fc(x), attentions
        return self.fc(x)


# Include a plot for attention weights if applicable
