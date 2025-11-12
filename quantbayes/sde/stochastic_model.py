import matplotlib.pyplot as plt
import numpy as np
import torch

from quantbayes.sde.sde_base import BaseSDE


class StochasticModel:
    """
    A 'manager' class that can hold any SDE that inherits from BaseSDE.
    """

    def __init__(self, sde: BaseSDE):
        """
        :param sde: An instance of a class that inherits from BaseSDE.
        """
        self.sde = sde
        self.t = None
        self.y = None

    def fit(self, t: np.ndarray, y: np.ndarray, **kwargs):
        """
        Fit the underlying SDE to the data.
        """
        self.t = t
        self.y = y
        self.sde.fit(t, y, **kwargs)

    def visualize_simulation(
        self, future_horizon: float = 1.0, n_paths: int = 10, n_steps: int = 100
    ):
        """
        Plot the historical data (in one color) and then sample possible
        future paths from the last time point of y.
        """
        if self.t is None or self.y is None:
            raise ValueError("Must fit the model before visualization.")

        t0 = self.t[-1]
        y0 = self.y[-1]

        # Simulate future paths
        paths = self.sde.simulate(
            t0=t0, y0=y0, T=future_horizon, n_paths=n_paths, n_steps=n_steps
        )
        if isinstance(paths, torch.Tensor):
            paths = paths.detach().numpy()

        # Build full time axis for the future
        dt = future_horizon / n_steps
        future_t = np.linspace(t0, t0 + future_horizon, n_steps + 1)

        # Plot
        plt.figure(figsize=(10, 6))
        # Plot historical
        plt.plot(self.t, self.y, label="Historical Data", color="blue", linewidth=2)

        # Plot future paths
        for i in range(n_paths):
            plt.plot(future_t, paths[i], color="red", alpha=0.7, linewidth=1)

        plt.title(
            "Historical Data + Future Simulations ({})".format(
                self.sde.__class__.__name__
            )
        )
        plt.xlabel("Time")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.show()
