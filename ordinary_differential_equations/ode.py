import functools
import matplotlib.pyplot as plt
import jax
from jax.experimental.ode import odeint
import jax.numpy as jnp
from jax.random import PRNGKey
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_sample


def dz_dt(z, t, theta):
    """
    Lotka–Volterra equations. Real positive parameters `alpha`, `beta`, `gamma`, `delta`
    describe the interaction of two species.
    """
    u, v = z
    alpha, beta, gamma, delta = theta

    du_dt = (alpha - beta * v) * u
    dv_dt = (-gamma + delta * u) * v
    return jnp.stack([du_dt, dv_dt])


def model(ts, y_init, y=None):
    """
    :param numpy.ndarray ts: measurement times
    :param numpy.ndarray y_init: measured initial conditions
    :param numpy.ndarray y: measured populations
    """
    # Initial population
    z_init = numpyro.sample(
        "z_init", dist.LogNormal(jnp.log(y_init), jnp.ones_like(y_init))
    )

    # Parameters alpha, beta, gamma, delta of dz_dt
    theta = numpyro.sample(
        "theta",
        dist.TruncatedNormal(
            low=0.0,
            loc=jnp.array([1.0, 0.05, 1.0, 0.05]),
            scale=jnp.array([0.2, 0.01, 0.2, 0.01]),
        ),
    )

    # Solve ODEs
    odeint_with_kwargs = functools.partial(odeint, rtol=1e-6, atol=1e-5, mxstep=1000)
    zs = odeint_with_kwargs(dz_dt, z_init, ts, theta)

    # Measurement errors
    sigma = numpyro.sample("sigma", dist.LogNormal(-1, 1).expand([2]))

    # Measured populations
    if y is not None:
        # Mask missing observations in the observed y
        mask = jnp.isfinite(jnp.log(y))
        numpyro.sample("y", dist.LogNormal(jnp.log(zs), sigma).mask(mask), obs=y)
    else:
        numpyro.sample("y", dist.LogNormal(jnp.log(zs), sigma))


def generate_synthetic_data(key, n_timesteps=100):
    """Generates synthetic data based on the Lotka-Volterra model."""
    ts = jnp.linspace(0, 10, n_timesteps)
    true_params = jnp.array([1.0, 0.05, 1.0, 0.05])  # True parameters
    y_init = jnp.array([10.0, 5.0])  # Initial populations

    # Solve the ODE with true parameters
    zs = odeint(dz_dt, y_init, ts, true_params)

    # Add noise to simulate measurements
    sigma = jnp.array([0.1, 0.1])  # Standard deviation of noise
    noisy_data = zs + jax.random.normal(key, zs.shape) * sigma

    return ts, y_init, noisy_data


def run_inference(ts, y_init, y):
    """Runs MCMC to infer parameters."""
    nuts_kernel = NUTS(model, init_strategy=init_to_sample)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=2000)
    mcmc.run(PRNGKey(1), ts=ts, y_init=y_init, y=y)
    return mcmc.get_samples()


def plot_results(ts, y, zs_true, predictions):
    """Plots the observed, true, and predicted results."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Predicted mean and true data
    predicted_mean = predictions.mean(axis=0)

    axs[0].plot(ts, y[:, 0], "ko", label="Observed Hare", alpha=0.7)
    axs[0].plot(ts, zs_true[:, 0], "r-", label="True Hare", alpha=0.7)
    axs[0].plot(ts, predicted_mean[:, 0], "b--", label="Predicted Hare", alpha=0.7)

    axs[1].plot(ts, y[:, 1], "kx", label="Observed Lynx", alpha=0.7)
    axs[1].plot(ts, zs_true[:, 1], "r-", label="True Lynx", alpha=0.7)
    axs[1].plot(ts, predicted_mean[:, 1], "b--", label="Predicted Lynx", alpha=0.7)

    for ax in axs:
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Population")

    plt.tight_layout()
    plt.show()


def main():
    """Main function to execute the workflow."""
    numpyro.enable_x64(True)  # Enable higher numeric precision

    # Step 1: Generate synthetic data
    key = PRNGKey(0)
    ts, y_init, y = generate_synthetic_data(key)

    # Step 2: Run inference
    posterior_samples = run_inference(ts, y_init, y)

    # Step 3: Posterior predictive checks
    predictive = Predictive(model, posterior_samples)
    predictions = predictive(PRNGKey(2), ts=ts, y_init=y_init)["y"]

    # Step 4: Plot results
    plot_results(
        ts, y, odeint(dz_dt, y_init, ts, jnp.array([1.0, 0.05, 1.0, 0.05])), predictions
    )


if __name__ == "__main__":
    main()
