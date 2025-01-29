import numpy as np
from quantbayes.sde.sde_base import BaseSDE


class HestonModel(BaseSDE):
    """
    Heston Model:
        dS_t = mu * S_t dt + sqrt(v_t) * S_t dW_t^{(1)}
        dv_t = kappa*(theta - v_t) dt + sigma * sqrt(v_t) dW_t^{(2)}

    This is a 2-dimensional SDE. For simplicity, let's track only S_t
    as the 'observable' in y, ignoring direct observations of volatility.
    """

    def __init__(
        self,
        mu: float = 0.05,
        kappa: float = 1.0,
        theta: float = 0.04,
        sigma: float = 0.3,
        v0: float = 0.04,
        rho: float = 0.0,
    ):
        super().__init__()
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.v0 = v0
        self.rho = rho  # correlation between dW^(1) and dW^(2)

    def fit(self, t: np.ndarray, y: np.ndarray):
        """
        Dummy fit: We will just guess parameters based on sample means
        and variances, ignoring the complexities of true Heston calibration.
        In practice, you'd do a 2D calibration for volatility and price.
        """
        if np.any(y <= 0):
            raise ValueError("All y values must be positive for Heston fit.")
        if len(t) < 2:
            raise ValueError("Need at least 2 time points to fit Heston.")

        # For demonstration, do something naive:
        dt = (t[-1] - t[0]) / (len(t) - 1)
        log_returns = np.diff(np.log(y))

        # "Estimate" drift from mean log-return
        self.mu = np.mean(log_returns) / dt

        # The rest remain as initial guesses or user-provided values
        # (kappa, theta, sigma, v0, rho)
        self.fitted = True

    def simulate(
        self,
        t0: float,
        y0: float,  # Change from S0 to y0
        T: float,
        n_paths: int = 10,
        n_steps: int = 100,
    ) -> np.ndarray:
        """
        Simulate the Heston model using Euler-Maruyama for both S_t and v_t.
        """
        if not self.fitted:
            print("Warning: Using default parameters (unfitted model).")

        dt = T / n_steps
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))

        # Initialize
        S[:, 0] = y0  # Change from S0 to y0
        v[:, 0] = self.v0

        # Correlation
        # Generate correlated Brownian increments using Cholesky decomposition
        cov_matrix = np.array([[1.0, self.rho], [self.rho, 1.0]])
        L = np.linalg.cholesky(cov_matrix)

        for i in range(n_steps):
            # Generate two independent standard normals
            Z = np.random.normal(0.0, 1.0, size=(2, n_paths))
            # Correlate them
            dW = L @ Z

            # For each path
            S_t = S[:, i]
            v_t = v[:, i]

            # Volatility can't go negative in Heston; typical approach is to ensure positivity
            v_t_plus = np.maximum(
                v_t
                + self.kappa * (self.theta - v_t) * dt
                + self.sigma * np.sqrt(np.maximum(v_t, 0.0)) * np.sqrt(dt) * dW[1, :],
                1e-8,
            )

            S_t_plus = (
                S_t
                + self.mu * S_t * dt
                + np.sqrt(np.maximum(v_t, 0.0)) * S_t * np.sqrt(dt) * dW[0, :]
            )

            v[:, i + 1] = v_t_plus
            S[:, i + 1] = S_t_plus

        return S
