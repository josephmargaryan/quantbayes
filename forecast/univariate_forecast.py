import jax.numpy as jnp
from jax import random
import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import matplotlib.pyplot as plt


# Define the SGT model
def sgt(y, seasonality, future=0):
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

    # Transition function
    def transition_fn(carry, t):
        level, s, moving_sum, rng_key = carry
        rng_key, subkey = random.split(rng_key)

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

    # Initialization
    N = y.shape[0]
    level_init = y[0]
    s_init = jnp.concatenate([init_s[1:], init_s[:1]], axis=0)
    moving_sum = level_init
    rng_key = random.PRNGKey(0)

    # Run scan
    with numpyro.handlers.condition(data={"y": y[1:]}):
        _, ys = scan(
            transition_fn,
            (level_init, s_init, moving_sum, rng_key),
            jnp.arange(1, N + future),
        )

    if future > 0:
        numpyro.deterministic("y_forecast", ys[-future:])


def test_sgt_with_train_test_split(y_train, y_test, seasonality, model):

    # Train the model on the training set
    mcmc = train_sgt(model, y_train, seasonality)

    # Predict future steps (aligned with the test set length)
    future_steps = len(y_test)
    forecast_marginal = predict_sgt(mcmc, y_train, model, seasonality, future_steps)

    # Visualize results: predictions vs ground truth
    plt.figure(figsize=(12, 6))

    # Plot training data
    plt.plot(y_train, label="Training Data", color="blue")

    # Plot ground truth for the test set
    plt.plot(
        range(len(y_train), len(y_train) + len(y_test)),
        y_test,
        label="Ground Truth (Test Data)",
        color="green",
    )

    # Plot forecast mean
    plt.plot(
        range(len(y_train), len(y_train) + future_steps),
        forecast_marginal.mean(axis=0),
        label="Forecast Mean",
        color="orange",
        linestyle="--",
    )

    # Plot uncertainty bands (credible intervals)
    plt.fill_between(
        range(len(y_train), len(y_train) + future_steps),
        jnp.percentile(forecast_marginal, 5, axis=0),
        jnp.percentile(forecast_marginal, 95, axis=0),
        alpha=0.5,
        color="orange",
        label="95% Credible Interval",
    )

    # Finalize plot
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.title("SGT Model Predictions vs Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.show()


# Train the SGT model
def train_sgt(model, y, seasonality, num_warmup=500, num_samples=1000):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(random.PRNGKey(0), y=y, seasonality=seasonality, future=0)
    return mcmc


# Predict with the SGT model
def predict_sgt(mcmc, y, model, seasonality, future):
    samples = mcmc.get_samples()
    predictive = Predictive(model, samples, return_sites=["y_forecast"])
    forecast_marginal = predictive(
        random.PRNGKey(1), y=y, seasonality=seasonality, future=future
    )["y_forecast"]
    return forecast_marginal


# Visualize the results
def visualize_sgt(y_train, forecast_marginal, future_steps, y_test=None):
    """
    Visualize SGT model predictions and optionally ground truth.

    Args:
        y_train (jnp.array): Training data for the target variable.
        forecast_marginal: Marginal distribution of forecasted values.
        future_steps (int): Number of forecasted steps.
        y_test (jnp.array, optional): Testing data for the target variable. Defaults to None.
    """
    plt.figure(figsize=(12, 6))

    # Plot training data
    plt.plot(y_train, label="Training Data", color="blue")

    # Plot test data (ground truth) if provided
    if y_test is not None:
        plt.plot(
            range(len(y_train), len(y_train) + len(y_test)),
            y_test,
            label="Ground Truth (Test Data)",
            color="green",
        )

    # Plot forecast mean
    plt.plot(
        range(len(y_train), len(y_train) + future_steps),
        forecast_marginal.mean(axis=0),
        label="Forecast Mean",
        color="orange",
        linestyle="--",
    )

    # Plot uncertainty (credible intervals)
    plt.fill_between(
        range(len(y_train), len(y_train) + future_steps),
        jnp.percentile(forecast_marginal, 5, axis=0),
        jnp.percentile(forecast_marginal, 95, axis=0),
        alpha=0.5,
        color="orange",
        label="95% Credible Interval",
    )

    # Finalize plot
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.title("SGT Model Predictions with Uncertainty")
    plt.legend()
    plt.grid(True)
    plt.show()


# Test the pipeline
def test_sgt_pipeline():
    # Generate synthetic data
    seasonality = 12
    x = jnp.arange(200)
    trend = 0.05 * x
    season = 10 * jnp.sin(2 * jnp.pi * x / seasonality)
    key = random.PRNGKey(42)
    noise = random.normal(key, shape=(200,)) * 2
    y = 50 + trend + season + noise

    # Train the model
    mcmc = train_sgt(y, seasonality)

    # Predict future steps
    future_steps = 24
    forecast_marginal = predict_sgt(mcmc, y, seasonality, future_steps)

    # Visualize results
    visualize_sgt(y, forecast_marginal, future_steps)


if __name__ == "__main__":
    test_sgt_pipeline()
