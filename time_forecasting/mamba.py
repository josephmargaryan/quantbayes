import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================
# Step 1: Define the Mamba-inspired State-Space Model
# ================================
class MambaStateSpaceModel(nn.Module):
    def __init__(self, state_dim, input_dim, output_dim):
        super().__init__()
        self.state_transition = nn.Linear(state_dim, state_dim)
        self.input_transform = nn.Linear(input_dim, state_dim)
        self.output_layer = nn.Linear(state_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        state = torch.zeros(batch_size, self.state_transition.out_features).to(x.device)
        outputs = []
        for t in range(seq_len):
            input_t = x[:, t, :]
            transformed_input = self.input_transform(input_t)
            state = torch.tanh(self.state_transition(state) + transformed_input)
            output_t = self.output_layer(state)
            outputs.append(output_t)
        return torch.stack(outputs, dim=1)


# ================================
# Step 2: Generate Synthetic Data
# ================================
def generate_synthetic_data(seq_len, input_dim, state_dim, output_dim, num_samples):
    true_state_transition = np.random.randn(state_dim, state_dim)
    true_input_transform = np.random.randn(input_dim, state_dim)
    true_output_layer = np.random.randn(state_dim, output_dim)

    inputs_list = []
    outputs_list = []
    for _ in range(num_samples):
        inputs = np.random.randn(seq_len, input_dim)
        states = np.zeros((seq_len, state_dim))
        outputs = np.zeros((seq_len, output_dim))
        for t in range(seq_len):
            if t == 0:
                states[t] = np.random.randn(state_dim)
            else:
                states[t] = np.tanh(
                    states[t - 1] @ true_state_transition
                    + inputs[t] @ true_input_transform
                )
            outputs[t] = states[t] @ true_output_layer
        inputs_list.append(inputs)
        outputs_list.append(outputs)
    return np.array(inputs_list), np.array(outputs_list)


# ================================
# Step 3: Instantiate Model and Prepare Data
# ================================
seq_len = 100
input_dim = 64
state_dim = 128
output_dim = 32
num_samples = 1000

# Generate synthetic data
inputs, outputs = generate_synthetic_data(
    seq_len, input_dim, state_dim, output_dim, num_samples
)
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
outputs_tensor = torch.tensor(outputs, dtype=torch.float32)

# Create DataLoader
batch_size = 32
dataset = TensorDataset(inputs_tensor, outputs_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate model
model = MambaStateSpaceModel(state_dim, input_dim, output_dim)
model = model.to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ================================
# Step 4: Train the Model
# ================================
individual_losses = []
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        predictions = model(inputs)

        # Compute loss
        loss = criterion(predictions, targets)
        epoch_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    individual_losses.append(epoch_loss / len(dataloader))

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), individual_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.show()

# ================================
# Step 5: Evaluate the Model
# ================================
# Generate test synthetic data
test_inputs, test_outputs = generate_synthetic_data(
    seq_len, input_dim, state_dim, output_dim, num_samples=200
)
test_inputs_tensor = torch.tensor(test_inputs, dtype=torch.float32)
test_outputs_tensor = torch.tensor(test_outputs, dtype=torch.float32)

model.eval()
with torch.no_grad():
    test_predictions = model(test_inputs_tensor.to(device))

# Convert predictions and ground truth to numpy for evaluation
true_outputs = test_outputs_tensor.numpy()
pred_outputs = test_predictions.cpu().numpy()

# Compute metrics
mse = mean_squared_error(true_outputs.flatten(), pred_outputs.flatten())
r2 = r2_score(true_outputs.flatten(), pred_outputs.flatten())
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# ================================
# Step 6: Visualize Results
# ================================
# Plot true vs predicted outputs for a single sequence and dimension
time_steps = np.arange(seq_len)
true_output_dim = true_outputs[0, :, 0]
pred_output_dim = pred_outputs[0, :, 0]

plt.figure(figsize=(10, 6))
plt.plot(
    time_steps, true_output_dim, label="True Output (Dimension 0)", linestyle="dashed"
)
plt.plot(time_steps, pred_output_dim, label="Predicted Output (Dimension 0)", alpha=0.7)
plt.xlabel("Time Step")
plt.ylabel("Output Value")
plt.title("True vs Predicted Outputs")
plt.legend()
plt.show()

# Advanced Visualization: Error Heatmap
error_matrix = np.abs(true_outputs - pred_outputs).mean(axis=0)

plt.figure(figsize=(12, 6))
plt.imshow(error_matrix.T, aspect="auto", cmap="viridis", origin="lower")
plt.colorbar(label="Absolute Error")
plt.xlabel("Time Step")
plt.ylabel("Output Dimension")
plt.title("Error Heatmap Across Time Steps and Output Dimensions")
plt.show()
