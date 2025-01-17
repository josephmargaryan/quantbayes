from neural_sde_base import NeuralSDE
from torch import nn
import torch


class LinearNeuralSDE(NeuralSDE):
    """
    Linear Neural SDE (LSDE):
    Drift and diffusion terms are modeled by linear neural networks.
    """

    def __init__(self, input_dim):
        """
        Initialize LSDE with simple linear networks.

        :param input_dim: Dimension of the input state.
        """
        drift_nn = nn.Sequential(nn.Linear(input_dim + 1, input_dim), nn.Tanh())
        diffusion_nn = nn.Sequential(nn.Linear(input_dim + 1, input_dim), nn.Softplus())
        super().__init__(drift_nn, diffusion_nn)

    def fit(self, t, y, lr=1e-3, epochs=100):
        """
        Train the LSDE using a simple loss function.

        :param t: 1D tensor of time points, shape (N,)
        :param y: 2D tensor of observations, shape (N, D)
        :param lr: Learning rate for the optimizer.
        :param epochs: Number of training epochs.
        """
        optimizer = torch.optim.Adam(
            list(self.drift_nn.parameters()) + list(self.diffusion_nn.parameters()),
            lr=lr,
        )

        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Generate simulated paths
            y_sim = self.simulate(
                t[0], y[0], T=t[-1] - t[0], n_paths=1, n_steps=len(t) - 1
            ).squeeze(0)

            # Compute loss between simulated and observed paths
            loss = loss_fn(y_sim, y)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
