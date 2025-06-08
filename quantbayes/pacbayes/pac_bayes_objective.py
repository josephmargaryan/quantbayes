import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


##############################################
# Bayesian Linear Layer Implementation
##############################################
class BayesianLinear(nn.Module):
    """
    A Bayesian linear layer with learnable posterior parameters (mean and log-variance)
    and a fixed Gaussian prior.
    """

    def __init__(self, in_features, out_features, prior_var=1.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Posterior parameters for weights and bias:
        self.weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features).normal_(0, 0.1)
        )
        self.weight_logvar = nn.Parameter(
            torch.Tensor(out_features, in_features).fill_(-3)
        )
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features).fill_(-3))
        self.prior_var = prior_var  # Variance of the fixed Gaussian prior

    def forward(self, input):
        # Reparameterization: sample from N(weight_mu, exp(weight_logvar)) and same for bias.
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)
        epsilon_weight = torch.randn_like(self.weight_mu)
        epsilon_bias = torch.randn_like(self.bias_mu)
        weight_sample = self.weight_mu + weight_std * epsilon_weight
        bias_sample = self.bias_mu + bias_std * epsilon_bias
        return F.linear(input, weight_sample, bias_sample)

    def kl_divergence(self):
        """
        Computes the closed-form KL divergence between the posterior N(mu, sigma^2)
        and the prior N(0, prior_var) for both the weights and bias.
        """
        weight_var = torch.exp(self.weight_logvar)
        kl_weight = (
            0.5
            * (
                ((weight_var + self.weight_mu**2) / self.prior_var)
                - 1
                - self.weight_logvar
                + math.log(self.prior_var)
            ).sum()
        )
        bias_var = torch.exp(self.bias_logvar)
        kl_bias = (
            0.5
            * (
                ((bias_var + self.bias_mu**2) / self.prior_var)
                - 1
                - self.bias_logvar
                + math.log(self.prior_var)
            ).sum()
        )
        return kl_weight + kl_bias


##############################################
# Bayesian Neural Network for Classification
##############################################
class BayesianNet(nn.Module):
    """
    Two-layer Bayesian neural network for multi-class classification.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, prior_var=1.0):
        super(BayesianNet, self).__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim, prior_var)
        self.fc2 = BayesianLinear(hidden_dim, output_dim, prior_var)

    def forward(self, x):
        # ReLU activation for the hidden layer.
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def kl_divergence(self):
        # Sum the KL divergences from both layers.
        return self.fc1.kl_divergence() + self.fc2.kl_divergence()


##############################################
# Variance-Aware PAC-Bayes Training
##############################################
def train_bayesian_net_bernstein(
    model,
    train_loader,
    optimizer,
    lam=1e-3,
    delta=0.05,
    c=7 / 3,
    num_epochs=10,
    device="cpu",
):
    """
    Trains the Bayesian network using a PAC-Bayes-Empirical-Bernstein objective.

    The loss is composed of:
      - Empirical loss: the standard cross-entropy loss.
      - KL divergence regularization: penalizing the divergence from the prior.
      - An extra variance-aware term based on Bernstein's inequality that adapts
        according to the uncertainty in the batch losses.

    Hyperparameters:
      lam  : weight for the KL divergence regularizer.
      delta: confidence parameter for the bound.
      c    : constant scaling factor for the extra bound term.
    """
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Forward pass: sample weights from the posterior.
            outputs = model(data)
            # Per-sample empirical loss using cross entropy:
            losses = F.cross_entropy(outputs, target, reduction="none")
            empirical_loss = losses.mean()

            # Empirical variance of the batch losses:
            variance = losses.var(unbiased=False)

            # KL divergence regularizer:
            kl = model.kl_divergence()

            # Additional Bernstein-inspired term.
            # Computes an extra term: sqrt( (2 * variance * (KL + log(1/delta)))/n )
            #                           + c*(KL + log(1/delta))/n
            n = data.size(0)
            extra_term = (
                math.sqrt((2 * variance.item() * (kl.item() + math.log(1 / delta))) / n)
                + c * (kl.item() + math.log(1 / delta)) / n
            )

            # Final objective: empirical loss + weighted KL + variance extra term.
            loss = empirical_loss + lam * kl + extra_term
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_epoch_loss:.4f}")


##############################################
# Main Script
##############################################
if __name__ == "__main__":
    # Set manual seed for reproducibility
    torch.manual_seed(42)

    # Hyperparameters for synthetic data and network architecture
    n_samples = 1000
    input_dim = 20
    hidden_dim = 64
    output_dim = 3  # number of classes
    batch_size = 64

    # Generate synthetic dummy data: features from a standard normal distribution and random class labels.
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, output_dim, (n_samples,))

    # Create dataset and dataloader for training.
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set device: use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the Bayesian neural network and move it to the selected device.
    model = BayesianNet(input_dim, hidden_dim, output_dim, prior_var=1.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training hyperparameters for the PAC-Bayes objective.
    num_epochs = 1000  # Increase in further experiments
    lam = 1e-3  # Weight for KL divergence term
    delta = 0.05  # Confidence parameter for the bound

    # Train the network with the variance-aware PAC-Bayes objective.
    train_bayesian_net_bernstein(
        model,
        train_loader,
        optimizer,
        lam=lam,
        delta=delta,
        num_epochs=num_epochs,
        device=device,
    )

    # Optionally, you can add further evaluation or save the final model for future experiments.
    # For example, evaluate on a validation/test set (if available) or dump the model state.
