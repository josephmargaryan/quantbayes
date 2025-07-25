import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from quantbayes.sde.sde_base import BaseSDE


def sanitize_mle_params(mle_params):
    """
    Sanitize MLE parameters for the Merton Jump Diffusion process.
    """
    sanitized = {}
    sanitized["mu"] = mle_params.get("mu", 0)  # No constraints
    sanitized["sigma"] = max(
        abs(mle_params.get("sigma", 0)), 1e-6
    )  # Ensure positive sigma
    sanitized["lambda"] = max(
        abs(mle_params.get("lambda", 0)), 1e-6
    )  # Ensure positive lambda
    sanitized["jump_mean"] = mle_params.get("jump_mean", 0)  # No constraints
    sanitized["jump_std"] = max(
        abs(mle_params.get("jump_std", 0)), 1e-6
    )  # Ensure positive jump_std
    return sanitized


class BayesianMertonJumpDiffusion(BaseSDE):
    def __init__(self):
        super().__init__()
        self.posterior_samples = None
        self.mu = None
        self.sigma = None
        self.lambda_ = None
        self.jump_mean = None
        self.jump_std = None

    def fit(self, t, y, num_samples=1000, num_warmup=500):
        dt = (t[-1] - t[0]) / (len(t) - 1)
        log_returns = np.diff(np.log(y))

        # Step 1: MLE for initial estimates
        mle_params = {
            "mu": np.mean(log_returns) / dt,
            "sigma": np.std(log_returns) / np.sqrt(dt),
            "lambda": 0.1,  # Initial guess for jump intensity
            "jump_mean": 0.0,
            "jump_std": 0.1,
        }
        sanitized_params = sanitize_mle_params(mle_params)

        # Debugging: Print sanitized parameters (optional)
        for key, value in sanitized_params.items():
            print(f"{key}: {value}")

        # Step 2: Bayesian inference
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
            lambda_ = numpyro.sample("lambda", dist.Exponential(1.0))
            jump_mean = numpyro.sample(
                "jump_mean", dist.Normal(sanitized_params["jump_mean"], 0.1)
            )
            jump_std = numpyro.sample(
                "jump_std", dist.LogNormal(np.log(sanitized_params["jump_std"]), 0.1)
            )

            # Poisson jumps
            jump_probs = jnp.exp(dist.Poisson(lambda_ * dt).log_prob(jnp.arange(0, 10)))
            jump_effects = jnp.arange(0, 10) * jump_mean
            combined_effect = jump_effects + log_returns[:, None]

            # Likelihood
            likelihood = dist.Normal(mu * dt, sigma * jnp.sqrt(dt)).log_prob(
                combined_effect
            )
            log_likelihood = jnp.sum(
                jnp.log(jnp.sum(jump_probs * jnp.exp(likelihood), axis=1))
            )
            numpyro.factor("log_likelihood", log_likelihood)

        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(jax.random.key(0), t=t, y=y)
        self.posterior_samples = mcmc.get_samples()

        # Step 3: Store posterior means as attributes
        self.mu = jnp.mean(self.posterior_samples["mu"])
        self.sigma = jnp.mean(self.posterior_samples["sigma"])
        self.lambda_ = jnp.mean(self.posterior_samples["lambda"])
        self.jump_mean = jnp.mean(self.posterior_samples["jump_mean"])
        self.jump_std = jnp.mean(self.posterior_samples["jump_std"])
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
        lambdas = self.posterior_samples["lambda"][posterior_indices]
        jump_means = self.posterior_samples["jump_mean"][posterior_indices]
        jump_stds = self.posterior_samples["jump_std"][posterior_indices]

        for i in range(n_steps):
            dW = np.random.normal(0.0, np.sqrt(dt), size=n_paths)
            dN = np.random.poisson(lambdas * dt, size=n_paths)
            jump_multiplier = (
                np.exp(
                    jump_means + jump_stds * np.random.normal(0.0, 1.0, size=n_paths)
                )
                ** dN
            )
            paths[:, i + 1] = (
                paths[:, i]
                * np.exp((mus - 0.5 * sigmas**2) * dt + sigmas * dW)
                * jump_multiplier
            )

        return paths
