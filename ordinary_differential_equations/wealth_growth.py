# Full Script for Wealth Growth Model
import functools
import matplotlib.pyplot as plt
import jax
from jax.experimental.ode import odeint
import jax.numpy as jnp
from jax.random import PRNGKey
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_sample


# Wealth Growth Model
def dz_dt(z, t, theta):
    W = z[0]  # Wealth
    r, sigma = theta
    dW_dt = r * W  # Deterministic growth (ignoring stochastic term for simplicity)
    return jnp.array([dW_dt])


def model(ts, y_init, y=None):
    z_init = numpyro.sample(
        "z_init", dist.LogNormal(jnp.log(y_init), jnp.ones_like(y_init))
    )
    theta = numpyro.sample(
        "theta",
        dist.TruncatedNormal(
            low=0.0, loc=jnp.array([0.05, 0.02]), scale=jnp.array([0.01, 0.005])
        ),
    )
    odeint_with_kwargs = functools.partial(odeint, rtol=1e-6, atol=1e-5, mxstep=1000)
    zs = odeint_with_kwargs(dz_dt, z_init, ts, theta)
    sigma = numpyro.sample("sigma", dist.LogNormal(-1, 1).expand([1]))
    if y is not None:
        mask = jnp.isfinite(jnp.log(y))
        numpyro.sample("y", dist.LogNormal(jnp.log(zs), sigma).mask(mask), obs=y)
    else:
        numpyro.sample("y", dist.LogNormal(jnp.log(zs), sigma))


def generate_synthetic_data(key, n_timesteps=100):
    ts = jnp.linspace(0, 10, n_timesteps)
    true_params = jnp.array([0.05, 0.02])  # r, sigma
    y_init = jnp.array([1.0])  # Initial wealth
    zs = odeint(dz_dt, y_init, ts, true_params)
    sigma_noise = 0.02
    noisy_data = zs + jax.random.normal(key, zs.shape) * sigma_noise
    return ts, y_init, noisy_data


def run_inference(ts, y_init, y):
    nuts_kernel = NUTS(model, init_strategy=init_to_sample)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=2000)
    mcmc.run(PRNGKey(1), ts=ts, y_init=y_init, y=y)
    return mcmc.get_samples()


def plot_results(ts, y, zs_true, predictions):
    """Plots the observed, true, and predicted results with uncertainty."""
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))

    # Predicted mean and credible intervals
    predicted_mean = predictions.mean(axis=0)
    predicted_low = jnp.percentile(predictions, 5, axis=0)  # 5th percentile
    predicted_high = jnp.percentile(predictions, 95, axis=0)  # 95th percentile

    # Plot observed data
    axs.plot(ts, y[:, 0], "ko", label="Observed Data", alpha=0.7)

    # Plot true curve
    axs.plot(ts, zs_true[:, 0], "r-", label="True Curve", alpha=0.7)

    # Plot predicted curve with uncertainty
    axs.plot(ts, predicted_mean[:, 0], "b--", label="Predicted Mean", alpha=0.7)
    axs.fill_between(
        ts,
        predicted_low[:, 0],
        predicted_high[:, 0],
        color="blue",
        alpha=0.2,
        label="95% Credible Interval",
    )

    axs.legend()
    axs.set_xlabel("Time")
    axs.set_ylabel("Interest Rate")
    plt.tight_layout()
    plt.show()


def main():
    numpyro.enable_x64(True)
    key = PRNGKey(0)
    ts, y_init, y = generate_synthetic_data(key)
    posterior_samples = run_inference(ts, y_init, y)
    predictive = Predictive(model, posterior_samples)
    predictions = predictive(PRNGKey(2), ts=ts, y_init=y_init)["y"]
    plot_results(ts, y, odeint(dz_dt, y_init, ts, jnp.array([0.05, 0.02])), predictions)


if __name__ == "__main__":
    main()
