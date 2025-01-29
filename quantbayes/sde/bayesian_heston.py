import jax.numpy as jnp
import numpy as np
import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.optimize import minimize
from quantbayes.sde.sde_base import BaseSDE


def sanitize_mle_params_heston(mle_params):
    """
    Sanitize MLE parameters for the Heston model.
    """
    sanitized = {}
    sanitized["mu"] = mle_params.get("mu", 0)  # No constraints
    sanitized["kappa"] = max(
        abs(mle_params.get("kappa", 0)), 1e-6
    )  # Ensure positive kappa
    sanitized["theta"] = max(
        abs(mle_params.get("theta", 0)), 1e-6
    )  # Ensure positive long-term variance
    sanitized["sigma"] = max(
        abs(mle_params.get("sigma", 0)), 1e-6
    )  # Ensure positive sigma
    sanitized["rho"] = np.clip(
        mle_params.get("rho", 0), -1 + 1e-6, 1 - 1e-6
    )  # Ensure -1 < rho < 1
    return sanitized


class BayesianHestonModel(BaseSDE):
    def __init__(self):
        super().__init__()
        self.posterior_samples = None
        self.mu = None
        self.kappa = None
        self.theta = None
        self.sigma = None
        self.rho = None

    def fit(self, t, y, num_samples=1000, num_warmup=500):
        dt = (t[-1] - t[0]) / (len(t) - 1)
        log_returns = np.diff(np.log(y))

        # Step 1: MLE for initial estimates
        def negative_log_likelihood(params):
            mu, kappa, theta, sigma, rho = params
            variance = np.var(log_returns)
            mean_reverting_variance = kappa * (theta - variance) * dt
            diff_variance = mean_reverting_variance + sigma * np.sqrt(
                variance
            ) * np.sqrt(dt)

            drift = mu * y[:-1] * dt
            volatility = np.sqrt(variance) * y[:-1] * np.sqrt(dt)
            price_diff = drift + volatility
            likelihood = -0.5 * np.sum(
                (log_returns - price_diff) ** 2 / diff_variance
                + np.log(2 * np.pi * diff_variance)
            )
            return -likelihood

        initial_params = [0.05, 2.0, 0.04, 0.3, 0.0]
        bounds = [
            (None, None),
            (1e-3, None),
            (1e-3, None),
            (1e-3, None),
            (-1 + 1e-6, 1 - 1e-6),
        ]
        result = minimize(negative_log_likelihood, initial_params, bounds=bounds)
        mle_mu, mle_kappa, mle_theta, mle_sigma, mle_rho = result.x

        # Step 2: Sanitize MLE parameters
        mle_params = {
            "mu": mle_mu,
            "kappa": mle_kappa,
            "theta": mle_theta,
            "sigma": mle_sigma,
            "rho": mle_rho,
        }
        sanitized_params = sanitize_mle_params_heston(mle_params)

        # Step 3: Bayesian inference
        def model(t, y):
            mu = numpyro.sample(
                "mu",
                dist.Normal(
                    sanitized_params["mu"], abs(sanitized_params["mu"]) * 0.1 + 1e-6
                ),
            )
            kappa = numpyro.sample(
                "kappa", dist.LogNormal(np.log(sanitized_params["kappa"]), 0.1)
            )
            theta = numpyro.sample(
                "theta", dist.LogNormal(np.log(sanitized_params["theta"]), 0.1)
            )
            sigma = numpyro.sample(
                "sigma", dist.LogNormal(np.log(sanitized_params["sigma"]), 0.1)
            )
            rho = numpyro.sample("rho", dist.Uniform(-1 + 1e-6, 1 - 1e-6))

            variance = jnp.var(jnp.diff(jnp.log(y)))
            mean_reverting_variance = kappa * (theta - variance) * dt
            diff_variance = mean_reverting_variance + sigma * jnp.sqrt(
                variance
            ) * jnp.sqrt(dt)

            drift = mu * y[:-1] * dt
            volatility = jnp.sqrt(variance) * y[:-1] * jnp.sqrt(dt)

            price_diff = drift + volatility
            numpyro.sample(
                "obs", dist.Normal(price_diff, diff_variance), obs=log_returns
            )

        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(jax.random.PRNGKey(0), t=t, y=y)
        self.posterior_samples = mcmc.get_samples()

        # Step 4: Store posterior means as attributes
        self.mu = jnp.mean(self.posterior_samples["mu"])
        self.kappa = jnp.mean(self.posterior_samples["kappa"])
        self.theta = jnp.mean(self.posterior_samples["theta"])
        self.sigma = jnp.mean(self.posterior_samples["sigma"])
        self.rho = jnp.mean(self.posterior_samples["rho"])
        self.fitted = True

    def simulate(self, t0, y0, T, n_paths=10, n_steps=100):
        """
        Simulate paths using the Heston model.
        """
        if not self.fitted:
            raise ValueError("The model must be fitted before simulation.")

        dt = T / n_steps
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))

        # Initialize
        S[:, 0] = y0
        v[:, 0] = self.theta  # Start at long-term variance

        key = jax.random.PRNGKey(0)  # Initialize random key

        # Sample parameters from posterior
        posterior_indices = np.random.choice(len(self.posterior_samples["mu"]), n_paths)
        mus = self.posterior_samples["mu"][posterior_indices]
        kappas = self.posterior_samples["kappa"][posterior_indices]
        thetas = self.posterior_samples["theta"][posterior_indices]
        sigmas = self.posterior_samples["sigma"][posterior_indices]
        rhos = self.posterior_samples["rho"][posterior_indices]

        for i in range(n_steps):
            key, subkey1, subkey2 = jax.random.split(key, 3)
            z1 = jax.random.normal(subkey1, shape=(n_paths,))
            z2 = jax.random.normal(subkey2, shape=(n_paths,))
            w1 = z1
            w2 = rhos * z1 + np.sqrt(1 - rhos**2) * z2

            # Variance process
            v[:, i + 1] = (
                v[:, i]
                + kappas * (thetas - v[:, i]) * dt
                + sigmas * np.sqrt(np.maximum(v[:, i], 0)) * np.sqrt(dt) * w2
            )
            v[:, i + 1] = np.maximum(v[:, i + 1], 1e-8)  # Ensure positivity

            # Price process
            S[:, i + 1] = S[:, i] * np.exp(
                (mus - 0.5 * v[:, i]) * dt + np.sqrt(v[:, i]) * np.sqrt(dt) * w1
            )

        return S

    def predict(self, t0, y0, T, n_paths=10, n_steps=100):
        """
        Predict future trajectories of the Heston model.
        """
        trajectories = self.simulate(t0, y0, T, n_paths, n_steps)
        return trajectories.T  # Transpose to match the format (timesteps, paths)
