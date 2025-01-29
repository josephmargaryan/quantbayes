import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

# Transformer-based time series model
class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, emb_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(TransformerTimeSeriesModel, self).__init__()
        self.embedding = nn.Linear(input_dim, emb_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, emb_dim))  # Maximum 100 time steps
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(emb_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.embedding(x)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.transformer_encoder(x)
        logits = self.fc(x[:, -1, :])  # Use the last time step for classification
        probs = self.softmax(logits)
        return probs

# Learned reward model
class LearnedRewardModel(nn.Module):
    def __init__(self, input_dim):
        super(LearnedRewardModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Scalar reward output
        )

    def forward(self, x):
        return self.network(x)

# Simulate stock price data
def simulate_data(num_samples, sequence_length, num_features):
    np.random.seed(42)
    data = np.cumsum(np.random.randn(num_samples, sequence_length, num_features), axis=1)
    labels = (data[:, -1, 0] > data[:, -2, 0]).astype(int)  # Predict up (1) or down (0)
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# Visualization function
def visualize_predictions(data, predictions, title="Predictions vs True Data"):
    plt.figure(figsize=(12, 6))
    for i in range(min(len(data), 5)):  # Plot up to 5 sequences
        plt.plot(data[i, :, 0].numpy(), label=f"True Sequence {i + 1}")
        pred_label = "Up" if predictions[i] == 1 else "Down"
        plt.scatter(len(data[i]) - 1, data[i, -1, 0].numpy(), c="red" if predictions[i] == 1 else "blue", label=f"Predicted: {pred_label}")
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

# Training loop with learned reward model
def train_with_learned_rewards(model, reward_model, data_loader, optimizer, reward_optimizer, gamma=0.99, visualize_after_epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            inputs, targets = batch

            # Forward pass
            probs = model(inputs)
            dist = Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            # Compute rewards using the learned reward model
            rewards = reward_model(inputs[:, -1, :])  # Use last time step as input to reward model
            rewards = rewards.squeeze() * (actions == targets).float()  # Scale rewards by correctness

            # Compute returns
            returns = []
            G = 0
            for r in reversed(rewards.tolist()):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)

            # Policy gradient loss
            loss = -(log_probs * returns).mean()

            # Update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Train reward model
            reward_loss = nn.MSELoss()(rewards, (actions == targets).float())  # Fit reward to correctness
            reward_optimizer.zero_grad()
            reward_loss.backward()
            reward_optimizer.step()

            total_loss += loss.item()

        # Visualization after specific epochs
        if (epoch + 1) % visualize_after_epochs == 0:
            model.eval()
            sample_data = data[:10]  # Take first 10 samples
            predictions = torch.argmax(model(sample_data), dim=-1)
            visualize_predictions(sample_data, predictions, title=f"Epoch {epoch + 1} Predictions")

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader)}")

if __name__ == "__main__":
    # Hyperparameters
    input_dim = 5  # Number of features
    emb_dim = 128  # Embedding dimension
    num_heads = 4  # Number of attention heads
    num_layers = 2  # Number of Transformer encoder layers
    output_dim = 2  # Up or down
    epochs = 10
    batch_size = 16
    lr = 0.001

    # Generate synthetic financial data
    sequence_length = 30
    num_samples = 100
    features = 5
    data, labels = simulate_data(num_samples, sequence_length, features)
    dataset = torch.utils.data.TensorDataset(data, labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, reward model, and optimizers
    model = TransformerTimeSeriesModel(input_dim, emb_dim, num_heads, num_layers, output_dim)
    reward_model = LearnedRewardModel(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    reward_optimizer = optim.Adam(reward_model.parameters(), lr=lr)

    # Train the model and visualize
    train_with_learned_rewards(model, reward_model, data_loader, optimizer, reward_optimizer)
