import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from quantbayes.sde.sde_base import BaseSDE


def sanitize_mle_params(mle_params):
    """
    Sanitize MLE parameters for the Geometric Brownian Motion process.
    """
    sanitized = {}
    sanitized["mu"] = mle_params.get("mu", 0)  # Allow negative mu
    sanitized["sigma"] = max(
        abs(mle_params.get("sigma", 0)), 1e-6
    )  # Ensure positive sigma
    return sanitized


class BayesianGeometricBrownianMotion(BaseSDE):
    def __init__(self):
        super().__init__()
        self.posterior_samples = None
        self.mu = None
        self.sigma = None

    def fit(self, t, y, num_samples=1000, num_warmup=500):
        dt = (t[-1] - t[0]) / (len(t) - 1)
        log_returns = np.diff(np.log(y))

        # Step 1: MLE for initial estimates
        mle_params = {
            "mu": np.mean(log_returns) / dt,
            "sigma": np.std(log_returns) / np.sqrt(dt),
        }

        # Step 2: Sanitize MLE parameters
        sanitized_params = sanitize_mle_params(mle_params)

        # Step 3: Bayesian inference
        def model(t, y):
            mu = numpyro.sample(
                "mu",
                dist.Normal(
                    sanitized_params["mu"], abs(sanitized_params["mu"]) * 0.1 + 1e-6
                ),
            )
            sigma = numpyro.sample(
                "sigma", dist.LogNormal(np.log(sanitized_params["sigma"]), 0.1)
            )
            numpyro.sample(
                "obs", dist.Normal(mu * dt, sigma * jnp.sqrt(dt)), obs=log_returns
            )

        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(jax.random.key(0), t=t, y=y)
        self.posterior_samples = mcmc.get_samples()

        # Store posterior means
        self.mu = jnp.mean(self.posterior_samples["mu"])
        self.sigma = jnp.mean(self.posterior_samples["sigma"])
        self.fitted = True

    def simulate(self, t0, y0, T, n_paths=10, n_steps=100):
        if not self.fitted:
            raise ValueError("The model must be fitted before simulation.")

        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = y0

        posterior_indices = np.random.choice(len(self.posterior_samples["mu"]), n_paths)
        mus = self.posterior_samples["mu"][posterior_indices]
        sigmas = self.posterior_samples["sigma"][posterior_indices]

        for i in range(n_steps):
            dW = np.random.normal(0.0, np.sqrt(dt), size=n_paths)
            paths[:, i + 1] = paths[:, i] * np.exp(
                (mus - 0.5 * sigmas**2) * dt + sigmas * dW
            )

        return paths
