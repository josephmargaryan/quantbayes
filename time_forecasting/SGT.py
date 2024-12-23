import jax.numpy as jnp
from jax import random
import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import matplotlib.pyplot as plt


# Define the SGT model
def sgt(y, seasonality, future=0):
    # Heuristic for Cauchy prior scale
    cauchy_sd = jnp.max(y) / 150

    # Priors
    nu = numpyro.sample("nu", dist.Uniform(2, 20))
    powx = numpyro.sample("powx", dist.Uniform(0, 1))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(cauchy_sd))
    offset_sigma = numpyro.sample(
        "offset_sigma", dist.TruncatedCauchy(low=1e-10, loc=1e-10, scale=cauchy_sd)
    )
    coef_trend = numpyro.sample("coef_trend", dist.Cauchy(0, cauchy_sd))
    pow_trend_beta = numpyro.sample("pow_trend_beta", dist.Beta(1, 1))
    pow_trend = 1.5 * pow_trend_beta - 0.5
    pow_season = numpyro.sample("pow_season", dist.Beta(1, 1))
    level_sm = numpyro.sample("level_sm", dist.Beta(1, 2))
    s_sm = numpyro.sample("s_sm", dist.Uniform(0, 1))
    init_s = numpyro.sample("init_s", dist.Cauchy(0, y[:seasonality] * 0.3))

    # Define transition function
    def transition_fn(carry, t):
        level, s, moving_sum, rng_key = carry
        rng_key, subkey = random.split(rng_key)  # Split RNG key for sampling

        season = s[0] * level**pow_season
        exp_val = level + coef_trend * level**pow_trend + season
        exp_val = jnp.clip(exp_val, 0)
        y_t = jnp.where(t >= N, exp_val, y[t])

        moving_sum = (
            moving_sum + y_t - jnp.where(t >= seasonality, y[t - seasonality], 0.0)
        )
        level_p = jnp.where(t >= seasonality, moving_sum / seasonality, y_t - season)
        level = level_sm * level_p + (1 - level_sm) * level
        level = jnp.clip(level, 0)

        new_s = (s_sm * (y_t - level) / season + (1 - s_sm)) * s[0]
        new_s = jnp.where(t >= N, s[0], new_s)
        s = jnp.concatenate([s[1:], new_s[None]], axis=0)

        omega = sigma * exp_val**powx + offset_sigma
        y_ = numpyro.sample("y", dist.StudentT(nu, exp_val, omega), rng_key=subkey)

        return (level, s, moving_sum, rng_key), y_

    # Initialize parameters
    N = y.shape[0]
    level_init = y[0]
    s_init = jnp.concatenate([init_s[1:], init_s[:1]], axis=0)
    moving_sum = level_init
    rng_key = random.PRNGKey(0)  # Proper RNG key initialization

    # Run scan
    with numpyro.handlers.condition(data={"y": y[1:]}):
        _, ys = scan(
            transition_fn,
            (level_init, s_init, moving_sum, rng_key),
            jnp.arange(1, N + future),
        )

    if future > 0:
        numpyro.deterministic("y_forecast", ys[-future:])


# Train and Test
def train_and_test_sgt():
    # Generate synthetic data
    seasonality = 12
    x = jnp.arange(200)
    trend = 0.05 * x
    season = 10 * jnp.sin(2 * jnp.pi * x / seasonality)
    key = random.PRNGKey(42)
    noise = random.normal(key, shape=(200,)) * 2
    y = 50 + trend + season + noise

    # Train the model
    nuts_kernel = NUTS(sgt)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
    mcmc.run(random.PRNGKey(0), y=y, seasonality=seasonality, future=24)

    # Predict
    samples = mcmc.get_samples()
    predictive = Predictive(sgt, samples, return_sites=["y_forecast"])
    forecast_marginal = predictive(
        random.PRNGKey(1), y=y, seasonality=seasonality, future=24
    )["y_forecast"]

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y, label="Observed", color="blue")
    plt.plot(
        range(len(y), len(y) + 24),
        forecast_marginal.mean(axis=0),
        label="Forecast Mean",
        color="orange",
        linestyle="--",
    )
    plt.fill_between(
        range(len(y), len(y) + 24),
        jnp.percentile(forecast_marginal, 5, axis=0),
        jnp.percentile(forecast_marginal, 95, axis=0),
        alpha=0.5,
        color="orange",
        label="95% Credible Interval",
    )
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("Forecast with Uncertainty")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train_and_test_sgt()
