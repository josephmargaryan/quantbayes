import numpy as np
import torch
import abc
from typing import Optional


class BaseSDE(abc.ABC):
    """
    Abstract Base Class for Stochastic Differential Equations.
    """

    def __init__(self):
        self.fitted = False

    @abc.abstractmethod
    def fit(self, t: np.ndarray, y: np.ndarray):
        """
        Fit the SDE's parameters to the observed time series data.

        :param t: 1D array of time points, shape (N,)
        :param y: 1D array of observations, shape (N,)
        """
        pass

    @abc.abstractmethod
    def simulate(
        self, t0: float, y0: float, T: float, n_paths: int = 10, n_steps: int = 100
    ) -> np.ndarray:
        """
        Simulate the SDE paths forward in time using Euler-Maruyama.

        :param t0: Initial time.
        :param y0: Initial value (at time t0).
        :param T: Length of time horizon over which to simulate.
        :param n_paths: Number of simulation paths.
        :param n_steps: Number of time steps in each path.
        :return: 2D array of shape (n_paths, n_steps+1) containing simulated paths.
        """
        pass

    def predict(
        self, t0: float, y0: float, T: float, n_paths: int = 10, n_steps: int = 100
    ) -> np.ndarray:
        """
        Predict future trajectories of the SDE using the simulate method.

        :param t0: Initial time.
        :param y0: Initial value (at time t0).
        :param T: Length of time horizon over which to predict.
        :param n_paths: Number of simulated trajectories.
        :param n_steps: Number of time steps in each trajectory.
        :return: 2D array of shape (n_steps+1, n_paths) containing predicted paths.
        """
        trajectories = self.simulate(t0, y0, T, n_paths, n_steps)

        if isinstance(trajectories, torch.Tensor):
            # For Neural SDEs returning PyTorch tensors
            return trajectories.permute(1, 0, 2).squeeze(-1).detach().numpy()
        elif isinstance(trajectories, np.ndarray):
            # For classical SDEs returning NumPy arrays
            return trajectories.T
        else:
            raise TypeError("Unsupported data type returned by simulate.")
