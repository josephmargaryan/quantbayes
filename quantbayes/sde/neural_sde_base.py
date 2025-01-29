import torch
import torch.nn as nn
from quantbayes.sde.sde_base import BaseSDE


class NeuralSDE(BaseSDE):
    """
    Neural SDE Base Class:
    Defines the structure for Neural SDEs with customizable drift and diffusion.
    """

    def __init__(self, drift_nn, diffusion_nn):
        """
        Initialize Neural SDE with drift and diffusion networks.

        :param drift_nn: Neural network defining the drift term.
        :param diffusion_nn: Neural network defining the diffusion term.
        """
        super().__init__()
        self.drift_nn = drift_nn
        self.diffusion_nn = diffusion_nn

    def drift(self, t, x):
        """
        Drift function (f(t, x)) modeled by a neural network.
        """
        return self.drift_nn(torch.cat((t, x), dim=-1))

    def diffusion(self, t, x):
        """
        Diffusion function (g(t, x)) modeled by a neural network.
        """
        return self.diffusion_nn(torch.cat((t, x), dim=-1))

    def fit(self, t, y):
        """
        Fit the Neural SDE model to the data.
        This involves training the drift and diffusion networks.

        :param t: 1D tensor of time points, shape (N,)
        :param y: 2D tensor of observations, shape (N, D)
        """
        raise NotImplementedError("Define the training procedure in a derived class.")

    def simulate(self, t0, y0, T, n_paths=1, n_steps=100):
        """
        Simulate paths using the Neural SDE.

        :param t0: Initial time (scalar or 0D tensor).
        :param y0: Initial state, shape (D,).
        :param T: Time horizon (scalar or 0D tensor).
        :param n_paths: Number of simulation paths.
        :param n_steps: Number of time steps.
        :return: Simulated paths, shape (n_paths, n_steps + 1, D).
        """
        # Convert t0 and T to scalars if they are tensors
        t0 = t0.item() if isinstance(t0, torch.Tensor) else t0
        T = T.item() if isinstance(T, torch.Tensor) else T

        dt = T / n_steps
        t = torch.linspace(t0, t0 + T, n_steps + 1).unsqueeze(1)  # Time points
        y = torch.zeros(n_paths, n_steps + 1, y0.shape[0])  # Initialize paths
        y[:, 0, :] = y0

        for i in range(n_steps):
            t_i = t[i].unsqueeze(0).expand(n_paths, -1)
            y_i = y[:, i, :]
            drift = self.drift(t_i, y_i)
            diffusion = self.diffusion(t_i, y_i)
            dW = torch.randn_like(y_i) * torch.sqrt(torch.tensor(dt))

            y[:, i + 1, :] = y_i + drift * dt + diffusion * dW

        return y
