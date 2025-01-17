import numpy as np
from sde_base import BaseSDE


class MertonJumpDiffusion(BaseSDE):
    """
    Merton's Jump Diffusion:
        dS_t = S_t * (mu dt + sigma dW_t) + S_t * (J - 1) dN_t
    where N_t is a Poisson process with intensity lambda_ (the jump intensity),
    and J ~ lognormal(m, s^2) is the jump size distribution.
    """

    def __init__(
        self,
        mu: float = 0.05,
        sigma: float = 0.2,
        lambda_: float = 0.1,
        jump_mean: float = 0.0,
        jump_std: float = 0.1,
    ):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.lambda_ = lambda_
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def fit(self, t: np.ndarray, y: np.ndarray):
        """
        Dummy fit for demonstration.
        Real fitting would be more involved (MLE with jump components, etc.).
        """
        if np.any(y <= 0):
            raise ValueError(
                "All y values must be positive for Merton Jump Diffusion fit."
            )
        if len(t) < 2:
            raise ValueError("Need at least 2 time points to fit Merton model.")

        dt = (t[-1] - t[0]) / (len(t) - 1)
        log_returns = np.diff(np.log(y))

        # Estimate drift and vol from "small" moves ignoring jumps
        self.mu = np.mean(log_returns) / dt
        self.sigma = np.std(log_returns, ddof=1) / np.sqrt(dt)
        # Jumps parameters remain user-defined or could be guessed from large outliers, etc.
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
        Simulate Merton's Jump Diffusion using Euler-Maruyama approach
        plus Poisson jumps.
        """
        if not self.fitted:
            print("Warning: Using default parameters (unfitted model).")

        dt = T / n_steps
        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = y0  # Change from S0 to y0

        for i in range(n_steps):
            dW = np.random.normal(0.0, np.sqrt(dt), size=n_paths)

            # Number of jumps in each path over this dt
            dN = np.random.poisson(self.lambda_ * dt, size=n_paths)

            # Jump multipliers. J = exp( jump_mean + jump_std * Z )
            # If dN=0 => no jump => multiplier is 1
            jump_multiplier = np.ones(n_paths)
            jump_idx = dN > 0
            if np.any(jump_idx):
                jump_multiplier[jump_idx] = (
                    np.exp(
                        self.jump_mean
                        + self.jump_std
                        * np.random.normal(0.0, 1.0, size=np.sum(jump_idx))
                    )
                    ** dN[jump_idx]
                )

            S[:, i + 1] = (
                S[:, i]
                * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dW)
                * jump_multiplier
            )

        return S
