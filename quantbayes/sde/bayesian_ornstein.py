import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.optimize import minimize

from quantbayes.sde.sde_base import BaseSDE


def sanitize_mle_params(mle_params):
    """
    Sanitize MLE parameters for the Ornstein-Uhlenbeck process.
    """
    sanitized = {}
    sanitized["theta"] = max(
        abs(mle_params.get("theta", 0)), 1e-6
    )  # Ensure positive theta
    sanitized["mu"] = mle_params.get("mu", 0)  # No constraints
    sanitized["sigma"] = max(
        abs(mle_params.get("sigma", 0)), 1e-6
    )  # Ensure positive sigma
    return sanitized


class BayesianOrnsteinUhlenbeck(BaseSDE):
    def __init__(self):
        super().__init__()
        self.posterior_samples = None
        self.theta = None
        self.mu = None
        self.sigma = None

    def fit(self, t, y, num_samples=1000, num_warmup=500):
        dt = (t[-1] - t[0]) / (len(t) - 1)

        # Step 1: MLE for initial estimates
        def negative_log_likelihood(params):
            theta, mu, sigma = params
            y_diff = np.diff(y)
            mean = theta * (mu - y[:-1]) * dt
            variance = sigma**2 * dt
            log_likelihood = -0.5 * np.sum(
                (y_diff - mean) ** 2 / variance + np.log(2 * np.pi * variance)
            )
            return -log_likelihood

        initial_params = [0.5, np.mean(y), np.std(y)]
        bounds = [(1e-3, None), (None, None), (1e-3, None)]
        result = minimize(negative_log_likelihood, initial_params, bounds=bounds)
        mle_theta, mle_mu, mle_sigma = result.x

        # Step 2: Sanitize MLE parameters
        mle_params = {"theta": mle_theta, "mu": mle_mu, "sigma": mle_sigma}
        sanitized_params = sanitize_mle_params(mle_params)

        # Debugging: Print sanitized parameters (optional)
        for key, value in sanitized_params.items():
            print(f"{key}: {value}")

        # Step 3: Bayesian inference
        def model(t, y):
            theta = numpyro.sample(
                "theta",
                dist.Normal(sanitized_params["theta"], sanitized_params["theta"] * 0.1),
            )
            mu = numpyro.sample(
                "mu",
                dist.Normal(
                    sanitized_params["mu"], abs(sanitized_params["mu"]) * 0.1 + 1e-6
                ),
            )
            sigma = numpyro.sample(
                "sigma", dist.LogNormal(np.log(sanitized_params["sigma"]), 0.1)
            )
            y_diff = jnp.diff(y)
            mean = theta * (mu - y[:-1]) * dt
            numpyro.sample("obs", dist.Normal(mean, sigma * jnp.sqrt(dt)), obs=y_diff)

        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(jax.random.key(0), t=t, y=y)
        self.posterior_samples = mcmc.get_samples()

        # Step 4: Store posterior means as attributes
        self.theta = jnp.mean(self.posterior_samples["theta"])
        self.mu = jnp.mean(self.posterior_samples["mu"])
        self.sigma = jnp.mean(self.posterior_samples["sigma"])
        self.fitted = True

    def simulate(self, t0, y0, T, n_paths=10, n_steps=100):
        if not self.fitted:
            raise ValueError("The model must be fitted before simulation.")

        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = y0

        posterior_indices = np.random.choice(
            len(self.posterior_samples["theta"]), n_paths
        )
        thetas = self.posterior_samples["theta"][posterior_indices]
        mus = self.posterior_samples["mu"][posterior_indices]
        sigmas = self.posterior_samples["sigma"][posterior_indices]

        for i in range(n_steps):
            dW = np.random.normal(0.0, np.sqrt(dt), size=n_paths)
            paths[:, i + 1] = (
                paths[:, i] + thetas * (mus - paths[:, i]) * dt + sigmas * dW
            )

        return paths
