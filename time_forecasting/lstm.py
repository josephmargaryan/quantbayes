import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================
# LSTM Model
# ================================
class LSTMTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMTimeSeriesModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # Output shape: (batch, seq_len, hidden_dim)
        # Fully connected layer applied at each time step
        return self.fc(lstm_out)


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
hidden_dim = 64
num_layers = 2
output_dim = 5
num_samples = 1000

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

# Instantiate model
model = LSTMTimeSeriesModel(input_dim, hidden_dim, num_layers, output_dim).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ================================
# Train the Model
# ================================
train_losses = []
val_losses = []
num_epochs = 10

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
    train_losses.append(train_loss)
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
plt.title("LSTM Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# ================================
# Evaluate the Model
# ================================
# Test on a single sequence from validation set
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
