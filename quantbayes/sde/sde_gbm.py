import numpy as np
from sde_base import BaseSDE


class GeometricBrownianMotion(BaseSDE):
    """
    Geometric Brownian Motion:
        dX_t = mu * X_t dt + sigma * X_t dW_t
    """

    def __init__(self, mu: float = 0.0, sigma: float = 0.1):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def fit(self, t: np.ndarray, y: np.ndarray):
        """
        Fit GBM parameters using log-returns approach.
        We assume y > 0. For example, y could be a stock price series.

        mu_hat = (1 / dt) * mean( log(y_{i+1} / y_i) )
        sigma_hat = (1 / sqrt(dt)) * std( log(y_{i+1} / y_i) )
        """
        # Basic checks
        if np.any(y <= 0):
            raise ValueError("All y values must be strictly positive for a GBM fit.")
        if len(t) < 2:
            raise ValueError("Need at least 2 time points to fit GBM.")

        # Convert to numpy arrays for safety
        t = np.array(t)
        y = np.array(y)

        # Time steps (assuming uniform spacing for simplicity)
        dt = (t[-1] - t[0]) / (len(t) - 1)

        # Compute log returns
        log_returns = np.diff(np.log(y))

        # Estimate parameters
        mu_hat = np.mean(log_returns) / dt
        sigma_hat = np.std(log_returns, ddof=1) / np.sqrt(dt)

        self.mu = mu_hat
        self.sigma = sigma_hat
        self.fitted = True

    def simulate(
        self, t0: float, y0: float, T: float, n_paths: int = 10, n_steps: int = 100
    ) -> np.ndarray:
        """
        Simulate using Euler-Maruyama or exact solution approach.
        For demonstration, we use an Euler scheme, but for GBM the
        exact solution is also straightforward.

        :return: shape (n_paths, n_steps+1)
        """
        if not self.fitted:
            print("Warning: Using default parameters (unfitted model).")

        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = y0

        for i in range(n_steps):
            dW = np.random.normal(0.0, np.sqrt(dt), size=n_paths)
            paths[:, i + 1] = (
                paths[:, i] + self.mu * paths[:, i] * dt + self.sigma * paths[:, i] * dW
            )

        return paths
