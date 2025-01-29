import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

class RLTimeSeriesModel(nn.Module):
    """
    A simple LSTM-based policy network that outputs a distribution over two classes (Up/Down).
    """
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(RLTimeSeriesModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.size()
        lstm_out, _ = self.lstm(x)
        # Use the last timestep's output
        last_step_out = lstm_out[:, -1, :]  # shape: (batch_size, hidden_dim)
        logits = self.fc(last_step_out)     # shape: (batch_size, output_dim)
        probs = self.softmax(logits)        # shape: (batch_size, output_dim)
        return probs

def generate_time_series_data(num_samples=100, seq_len=20, input_dim=5, seed=42):
    """
    Generate synthetic time-series data. 
    We interpret the target as binary (0 or 1) based on whether 
    the final step in the first feature is higher or lower than the second to last step.
    """
    np.random.seed(seed)
    data = np.cumsum(np.random.randn(num_samples, seq_len, input_dim), axis=1)
    labels = (data[:, -1, 0] > data[:, -2, 0]).astype(int)  # 1 if up, 0 if down
    X = torch.tensor(data, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y

def visualize_time_series_predictions(X, y_pred, title="Time Series Predictions"):
    """
    Simple visualization comparing the final up/down point of a few series.
    """
    plt.figure(figsize=(10, 5))
    max_plots = min(5, len(X))
    for i in range(max_plots):
        plt.plot(X[i, :, 0].numpy(), label=f"Series {i+1}")
        final_val = X[i, -1, 0].item()
        color = "red" if y_pred[i].item() == 1 else "blue"
        label = "Up" if y_pred[i].item() == 1 else "Down"
        plt.scatter([X.shape[1]-1], [final_val], c=color, marker='o', label=f"Pred: {label}")
    plt.title(title)
    plt.legend()
    plt.show()

def train_rl_time_series_model(model, data_loader, optimizer, num_epochs=10):
    """
    Simple policy gradient training loop. 
    Reward = +1 for correct classification, -1 for incorrect.
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for X_batch, y_batch in data_loader:
            # Forward pass
            probs = model(X_batch)               # (batch_size, 2)
            dist = Categorical(probs)
            actions = dist.sample()              # (batch_size,)
            log_probs = dist.log_prob(actions)   # (batch_size,)

            # Compute reward
            rewards = torch.where(actions == y_batch, 1.0, -1.0)

            # Policy gradient loss
            loss = -(log_probs * rewards).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss / len(data_loader):.4f}")

if __name__ == "__main__":
    # Hyperparams
    input_dim = 5
    hidden_dim = 32
    batch_size = 16
    lr = 0.001
    epochs = 5

    # Generate data
    X, y = generate_time_series_data(num_samples=100, seq_len=20, input_dim=input_dim)
    dataset = torch.utils.data.TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model & optimizer
    model = RLTimeSeriesModel(input_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    train_rl_time_series_model(model, data_loader, optimizer, num_epochs=epochs)

    # Quick test visualization
    model.eval()
    with torch.no_grad():
        sample_X = X[:8]
        sample_probs = model(sample_X)
        sample_preds = torch.argmax(sample_probs, dim=-1)
        visualize_time_series_predictions(sample_X, sample_preds, title="Sample Time Series Predictions")
