import numpy as np
from quantbayes.sde.sde_base import BaseSDE


class OrnsteinUhlenbeck(BaseSDE):
    """
    Ornstein-Uhlenbeck Process:
        dX_t = theta * (mu - X_t) dt + sigma * dW_t

    This is a mean-reverting stochastic process.
    """

    def __init__(self, mu: float = 0.0, theta: float = 1.0, sigma: float = 0.1):
        """
        Initialize the OU process parameters.

        :param mu: Long-term mean of the process.
        :param theta: Speed of mean reversion.
        :param sigma: Volatility of the process.
        """
        super().__init__()
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

    def fit(self, t: np.ndarray, y: np.ndarray):
        """
        Fit the OU process parameters to observed data.
        Use method of moments to estimate theta, mu, and sigma.

        :param t: 1D array of time points, shape (N,)
        :param y: 1D array of observations, shape (N,)
        """
        if len(t) < 2 or len(y) < 2:
            raise ValueError("At least two time points are required for fitting.")

        dt = (t[-1] - t[0]) / (len(t) - 1)  # Assume uniform time steps
        y_diff = np.diff(y)

        # Estimate theta (mean-reversion speed) from autocorrelation
        y_mean = np.mean(y)
        numerator = np.sum((y[:-1] - y_mean) * (y_diff))
        denominator = np.sum((y[:-1] - y_mean) ** 2)
        theta_hat = -numerator / (denominator * dt)

        # Estimate mu (long-term mean) and sigma (volatility)
        mu_hat = np.mean(y)
        sigma_hat = np.sqrt(np.var(y_diff) / (2 * theta_hat * dt))

        self.theta = theta_hat
        self.mu = mu_hat
        self.sigma = sigma_hat
        self.fitted = True

    def simulate(
        self, t0: float, y0: float, T: float, n_paths: int = 10, n_steps: int = 100
    ) -> np.ndarray:
        """
        Simulate OU process paths using Euler-Maruyama.

        :param t0: Initial time.
        :param y0: Initial value of the process.
        :param T: Time horizon for the simulation.
        :param n_paths: Number of simulation paths.
        :param n_steps: Number of time steps in each path.
        :return: 2D array of shape (n_paths, n_steps+1) containing simulated paths.
        """
        if not self.fitted:
            print("Warning: Using default parameters (unfitted model).")

        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = y0

        for i in range(n_steps):
            dW = np.random.normal(0.0, np.sqrt(dt), size=n_paths)
            paths[:, i + 1] = (
                paths[:, i]
                + self.theta * (self.mu - paths[:, i]) * dt
                + self.sigma * dW
            )

        return paths
