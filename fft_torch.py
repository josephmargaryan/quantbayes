import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# Generate synthetic data
def generate_synthetic_data(n_samples=100, n_features=8, noise_std=0.1):
    torch.manual_seed(0)
    X = torch.randn(n_samples, n_features)  # Input features
    true_weights = torch.randn(n_features)  # True weights
    true_bias = torch.randn(1)  # Scalar bias
    y = torch.matmul(X, true_weights) + true_bias  # Linear combination
    y += noise_std * torch.randn(n_samples)  # Add noise
    y = y.unsqueeze(1)  # Reshape y to [n_samples, 1]
    return X, y


# Create DataLoader
def create_dataloader(X, y, batch_size=32):
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class FFTCirculantLayer(nn.Module):
    def __init__(self, n):
        super(FFTCirculantLayer, self).__init__()
        self.n = n
        self.c = nn.Parameter(
            torch.randn(n)
        )  # Learnable parameters for the circulant matrix

    def forward(self, x):
        # FFT-based multiplication
        c_fft = torch.fft.fft(self.c)
        x_fft = torch.fft.fft(x, n=self.n)
        result_fft = c_fft * x_fft
        result = torch.fft.ifft(result_fft)
        return result.real


# Example usage in a model
class FFTModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FFTModel, self).__init__()
        self.layer1 = FFTCirculantLayer(input_dim)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.fc(x)
        return x


# Training loop
def train_model(model, dataloader, epochs=100, learning_rate=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()  # Mean Squared Error loss
    losses = []  # Store scalar loss values

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()  # Accumulate epoch loss
            losses.append(loss.item())  # Append scalar loss for plotting

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Plot training loss
    import matplotlib.pyplot as plt

    plt.plot(losses)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Generate synthetic data
    n_samples = 500
    n_features = 8
    output_dim = 8
    X, y = generate_synthetic_data(n_samples=500, n_features=8)
    dataloader = create_dataloader(X, y, batch_size=32)
    model = FFTModel(input_dim=8, output_dim=1)  # Adjust output dimension
    train_model(model, dataloader, epochs=100, learning_rate=0.01)
