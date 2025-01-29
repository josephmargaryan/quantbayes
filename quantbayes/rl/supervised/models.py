import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np


class RLClassifier(nn.Module):
    """
    Simple feedforward network that outputs a discrete distribution for binary or multiclass classification.
    """

    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(RLClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        logits = self.net(x)
        probs = self.softmax(logits)
        return probs


class RLRegressor(nn.Module):
    """
    For regression, we parametrize a Normal distribution: we output a mean (and optionally a log_std).
    For simplicity, we fix std = 1.
    """

    def __init__(self, input_dim, hidden_dim):
        super(RLRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        # If you want a learnable log_std, uncomment below:
        # self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = self.net(x)
        # std = self.log_std.exp()  # if you want dynamic std
        std = torch.ones_like(mean)  # fixed for simplicity
        return mean, std


def generate_synthetic_data(
    mode="binary", num_samples=200, input_dim=5, num_classes=3, seed=42
):
    """
    Generate random data for different supervised tasks:
      - binary classification
      - multiclass classification
      - regression
    """
    np.random.seed(seed)
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    if mode == "binary":
        # random labels 0 or 1
        y = np.random.randint(0, 2, size=(num_samples,))
        y = torch.tensor(y, dtype=torch.long)
    elif mode == "multiclass":
        y = np.random.randint(0, num_classes, size=(num_samples,))
        y = torch.tensor(y, dtype=torch.long)
    else:  # "regression"
        # random continuous labels
        y = 3 * X[:, 0] + 2.0 * X[:, 1] - 1.5  # a made-up linear function
        y = y + np.random.randn(num_samples).astype(np.float32) * 0.5
        y = torch.tensor(y, dtype=torch.float32)
    return torch.tensor(X), y


def policy_gradient_train_binary(model, X, y, optimizer, epochs=5):
    """
    Binary classification as RL. Reward = +1 if correct, -1 if incorrect.
    """
    for epoch in range(epochs):
        optimizer.zero_grad()
        probs = model(X)  # shape: (N, 2)
        dist = Categorical(probs)
        actions = dist.sample()  # shape: (N,)
        log_probs = dist.log_prob(actions)  # shape: (N,)

        rewards = torch.where(actions == y, 1.0, -1.0)
        loss = -(log_probs * rewards).mean()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")


def policy_gradient_train_multiclass(model, X, y, optimizer, epochs=5):
    """
    Multiclass classification as RL. Reward = +1 if correct, -1 if incorrect.
    """
    for epoch in range(epochs):
        optimizer.zero_grad()
        probs = model(X)  # shape: (N, num_classes)
        dist = Categorical(probs)
        actions = dist.sample()  # shape: (N,)
        log_probs = dist.log_prob(actions)  # shape: (N,)

        rewards = torch.where(actions == y, 1.0, -1.0)
        loss = -(log_probs * rewards).mean()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")


def policy_gradient_train_regression(model, X, y, optimizer, epochs=5):
    """
    Regression as RL. We sample from N(mean, std).
    Reward could be negative of squared error. (You can get creative here.)
    """
    for epoch in range(epochs):
        optimizer.zero_grad()
        mean, std = model(X)  # shape: (N,1), (N,1)
        dist = Normal(mean, std)
        sampled_y = dist.sample()  # shape: (N,1)
        log_probs = dist.log_prob(sampled_y).squeeze()  # shape: (N,)

        # negative MSE as reward => reward = -(sampled_y - y)^2
        # We must broadcast y properly
        y_ = y.view(-1, 1)
        rewards = -((sampled_y - y_) ** 2)
        rewards = rewards.squeeze()

        loss = -(log_probs * rewards).mean()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")


if __name__ == "__main__":
    # ---------------- Binary Classification ----------------
    print("=== BINARY CLASSIFICATION ===")
    X_bin, y_bin = generate_synthetic_data(mode="binary", num_samples=50, input_dim=5)
    bin_model = RLClassifier(input_dim=5, hidden_dim=16, output_dim=2)
    bin_optim = optim.Adam(bin_model.parameters(), lr=0.01)
    policy_gradient_train_binary(bin_model, X_bin, y_bin, bin_optim, epochs=5)
    with torch.no_grad():
        probs = bin_model(X_bin)
        preds = torch.argmax(probs, dim=-1)
        accuracy = (preds == y_bin).float().mean()
        print(f"Binary Classification Accuracy: {accuracy*100:.2f}%\n")

    # ---------------- Multiclass Classification ----------------
    print("=== MULTICLASS CLASSIFICATION ===")
    X_mc, y_mc = generate_synthetic_data(
        mode="multiclass", num_samples=50, input_dim=5, num_classes=3
    )
    mc_model = RLClassifier(input_dim=5, hidden_dim=16, output_dim=3)
    mc_optim = optim.Adam(mc_model.parameters(), lr=0.01)
    policy_gradient_train_multiclass(mc_model, X_mc, y_mc, mc_optim, epochs=5)
    with torch.no_grad():
        probs = mc_model(X_mc)
        preds = torch.argmax(probs, dim=-1)
        accuracy = (preds == y_mc).float().mean()
        print(f"Multiclass Classification Accuracy: {accuracy*100:.2f}%\n")

    # ---------------- Regression ----------------
    print("=== REGRESSION ===")
    X_reg, y_reg = generate_synthetic_data(
        mode="regression", num_samples=50, input_dim=5
    )
    reg_model = RLRegressor(input_dim=5, hidden_dim=16)
    reg_optim = optim.Adam(reg_model.parameters(), lr=0.01)
    policy_gradient_train_regression(reg_model, X_reg, y_reg, reg_optim, epochs=5)
    with torch.no_grad():
        mean, std = reg_model(X_reg)
        # We'll treat the predicted mean as the final point estimate
        mse = ((mean.squeeze() - y_reg) ** 2).mean()
        print(f"Regression MSE: {mse.item():.4f}")
