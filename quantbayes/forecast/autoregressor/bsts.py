import jax.numpy as jnp
from jax import random, lax
import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import matplotlib.pyplot as plt


def bsts_extended_model(y, seasonality=12, fourier_order=2, future=0):
    """
    BSTS model with local linear trend, Fourier-based seasonality, and robust StudentT noise.
    """

    # Priors
    sigma_level = numpyro.sample("sigma_level", dist.HalfCauchy(1.0))
    sigma_slope = numpyro.sample("sigma_slope", dist.HalfCauchy(1.0))
    nu = numpyro.sample("nu", dist.Uniform(2, 20))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))
    level_0 = numpyro.sample("level_0", dist.Normal(y[0], 10.0))
    slope_0 = numpyro.sample("slope_0", dist.Normal(0.0, 2.0))

    # Fourier coefficients
    alpha_list = [numpyro.sample(f"alpha_{k}", dist.Normal(0.0, 2.0)) for k in range(1, fourier_order + 1)]
    beta_list = [numpyro.sample(f"beta_{k}", dist.Normal(0.0, 2.0)) for k in range(1, fourier_order + 1)]
    alpha_vec, beta_vec = jnp.array(alpha_list), jnp.array(beta_list)

    N = y.shape[0]

    # Transition function
    def transition_fn(carry, t):
        level, slope, rng_key = carry
        rng_key, subkey = random.split(rng_key)

        # State updates
        level_next = level + slope + numpyro.sample(f"lvl_noise_{t}", dist.Normal(0, sigma_level))
        slope_next = slope + numpyro.sample(f"slp_noise_{t}", dist.Normal(0, sigma_slope))

        # Fourier-based seasonality
        angles = 2.0 * jnp.pi * jnp.arange(1, fourier_order + 1) * (t / seasonality)
        seasonal_term = jnp.sum(alpha_vec * jnp.sin(angles) + beta_vec * jnp.cos(angles))
        mean_t = level + seasonal_term

        # Observation model
        y_t = lax.cond(
            t < N,
            lambda _: y[t],  # Observed data
            lambda _: mean_t,  # Forecast
            operand=None
        )
        obs = numpyro.sample(
            f"y_{t}",
            dist.StudentT(nu, mean_t, sigma),
            obs=y_t,
            rng_key=subkey
        )

        return (level_next, slope_next, rng_key), obs

    # Initialize and scan
    carry_init = (level_0, slope_0, random.PRNGKey(0))
    _, ys = scan(transition_fn, carry_init, jnp.arange(N + future))

    if future > 0:
        numpyro.deterministic("y_forecast", ys[-future:])


class BSTSS:
    """
    BSTSSeasonalModel: A Bayesian Structural Time Series model with:
      - Local Linear Trend
      - Fourier-based Seasonality
      - Robust StudentT observations
    """

    def __init__(self, seasonality=12, fourier_order=2, num_warmup=500, num_samples=1000, rng_key=0):
        self.seasonality = seasonality
        self.fourier_order = fourier_order
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.rng_key = rng_key
        self.mcmc = None
        self.fitted_ = False
        self.y_train_ = None

    def fit(self, y):
        y = jnp.array(y)
        nuts_kernel = NUTS(bsts_extended_model)
        self.mcmc = MCMC(nuts_kernel, num_warmup=self.num_warmup, num_samples=self.num_samples)
        self.mcmc.run(
            random.PRNGKey(self.rng_key),
            y=y,
            seasonality=self.seasonality,
            fourier_order=self.fourier_order,
            future=0
        )
        self.fitted_ = True
        self.y_train_ = y
        return self

    def predict(self, steps):
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call .fit(...) first.")

        samples = self.mcmc.get_samples()
        predictive = Predictive(bsts_extended_model, samples, return_sites=["y_forecast"])
        out = predictive(
            random.PRNGKey(self.rng_key + 1),
            y=self.y_train_,
            seasonality=self.seasonality,
            fourier_order=self.fourier_order,
            future=steps
        )
        forecast_samples = out["y_forecast"]
        return forecast_samples

    def plot_forecast(self, y_train, forecast_samples, ax=None, credible_interval=(5, 95)):
        import numpy as np

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        y_train = np.array(y_train)
        future_steps = forecast_samples.shape[1]

        ax.plot(y_train, label="Training Data", color="blue")
        forecast_mean = jnp.mean(forecast_samples, axis=0)
        x_fore = range(len(y_train), len(y_train) + future_steps)
        ax.plot(x_fore, forecast_mean, label="Forecast Mean", color="orange", linestyle="--")

        lower, upper = credible_interval
        fc_lower = jnp.percentile(forecast_samples, lower, axis=0)
        fc_upper = jnp.percentile(forecast_samples, upper, axis=0)
        ax.fill_between(x_fore, fc_lower, fc_upper, color="orange", alpha=0.3,
                        label=f"{upper - lower}% Credible Interval")

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title("BSTS Forecast (Trend + Seasonality + StudentT)")
        ax.legend()
        ax.grid(True)
        if ax is None:
            plt.show()

    def get_samples(self):
        if not self.fitted_:
            raise RuntimeError("Model not fitted yet; no samples to retrieve.")
        return self.mcmc.get_samples()


if __name__ == "__main__":
    import numpy as np

    # Generate synthetic data
    np.random.seed(42)
    N = 100
    t = np.arange(N)
    level = 10.0
    slope = 0.05
    y_data = []
    for i in range(N):
        level += slope + np.random.normal(0, 0.2)
        slope += np.random.normal(0, 0.01)
        seas = 5.0 * np.sin(2 * np.pi * (i / 12.0)) + 3.0 * np.cos(4 * np.pi * (i / 12.0))
        y_val = level + seas + np.random.standard_t(df=8) * 0.5
        y_data.append(y_val)
    y_data = np.array(y_data)

    # Fit the model
    model = BSTSS(seasonality=12, fourier_order=2, num_warmup=500, num_samples=1000, rng_key=123)
    model.fit(y_data)

    # Predict future steps
    forecast_samples = model.predict(steps=12)

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 5))
    model.plot_forecast(y_data, forecast_samples, ax=ax, credible_interval=(5, 95))
    plt.show()
