import jax.numpy as jnp
import jax
import numpy as np
from jax.experimental.ode import odeint
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import functools


def dz_dt(z, t, theta):
    """
    Lotka-Volterra differential equation system.

    Args:
        z: State variables (e.g., prey and predator populations).
        t: Time variable.
        theta: Model parameters [alpha, beta, gamma, delta].

    Returns:
        dz/dt: Change in state variables.
    """
    u, v = z  # Prey and predator populations
    alpha, beta, gamma, delta = theta
    du_dt = (alpha - beta * v) * u
    dv_dt = (-gamma + delta * u) * v
    return jnp.stack([du_dt, dv_dt])


def generalized_model(ts, y_init, y=None):
    """
    Probabilistic model for Lotka-Volterra (prey-predator) dynamics.

    Args:
        ts: Time steps.
        y_init: Initial conditions [initial prey, initial predator].
        y: Observed data (optional).

    Returns:
        Bayesian probabilistic model.
    """
    # Initial conditions
    z_init = numpyro.sample(
        "z_init", dist.LogNormal(jnp.log(y_init), jnp.ones_like(y_init))
    )

    # Parameters of Lotka-Volterra
    theta = numpyro.sample(
        "theta",
        dist.TruncatedNormal(
            low=0.0,
            loc=jnp.array([1.0, 0.05, 1.0, 0.05]),
            scale=jnp.array([0.2, 0.01, 0.2, 0.01]),
        ),
    )

    # Solve ODE
    odeint_with_kwargs = functools.partial(odeint, rtol=1e-6, atol=1e-5, mxstep=1000)
    zs = odeint_with_kwargs(dz_dt, z_init, ts, theta)

    # Measurement noise
    sigma = numpyro.sample("sigma", dist.LogNormal(-1, 1).expand([zs.shape[1]]))

    # Observations
    if y is not None:
        mask = jnp.isfinite(jnp.log(y))
        numpyro.sample("y", dist.LogNormal(jnp.log(zs), sigma).mask(mask), obs=y)
    else:
        numpyro.sample("y", dist.LogNormal(jnp.log(zs), sigma))


def generate_synthetic_data(ts):
    """
    Generate synthetic time series data using Lotka-Volterra equations.

    Args:
        ts: Time steps.

    Returns:
        Synthetic time series data (prey and predator populations).
    """
    # Parameters for Lotka-Volterra
    theta = jnp.array([1.0, 0.05, 1.0, 0.05])
    z_init = jnp.array([10.0, 5.0])  # Initial prey and predator populations

    # Solve ODE
    zs = odeint(dz_dt, z_init, ts, theta)

    # Add observation noise
    noise = np.random.lognormal(mean=0, sigma=0.1, size=zs.shape)
    return zs * noise


def demo():
    # Time steps
    ts = jnp.linspace(0, 20, 100)

    # Generate synthetic data
    y_init = jnp.array([10.0, 5.0])  # Initial conditions
    synthetic_data = generate_synthetic_data(ts)

    # Simulate missing data using JAX's immutable operations
    observed_data = synthetic_data.copy()
    observed_data = observed_data.at[::5, :].set(
        jnp.nan
    )  # Simulate missing observations

    # Run Bayesian inference using NUTS
    nuts_kernel = NUTS(generalized_model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
    mcmc.run(jax.random.PRNGKey(0), ts, y_init, y=observed_data)

    # Print posterior summary
    mcmc.print_summary()

    # Posterior predictive sampling
    predictive = Predictive(generalized_model, posterior_samples=mcmc.get_samples())
    predictions = predictive(jax.random.PRNGKey(1), ts, y_init)

    # Extract predicted means and credible intervals
    predicted_means = predictions["y"].mean(axis=0)
    lower_bounds = jnp.percentile(predictions["y"], 5, axis=0)
    upper_bounds = jnp.percentile(predictions["y"], 95, axis=0)

    # Visualization
    plt.figure(figsize=(10, 6))

    # Plot observed data (prey and predator)
    plt.plot(
        ts, observed_data[:, 0], "o", label="Prey (Observed)", color="blue", alpha=0.5
    )
    plt.plot(
        ts,
        observed_data[:, 1],
        "o",
        label="Predator (Observed)",
        color="red",
        alpha=0.5,
    )

    # Plot predicted means
    plt.plot(ts, predicted_means[:, 0], label="Prey (Predicted)", color="blue")
    plt.plot(ts, predicted_means[:, 1], label="Predator (Predicted)", color="red")

    # Plot uncertainty bounds
    plt.fill_between(
        ts,
        lower_bounds[:, 0],
        upper_bounds[:, 0],
        color="blue",
        alpha=0.2,
        label="Prey Uncertainty",
    )
    plt.fill_between(
        ts,
        lower_bounds[:, 1],
        upper_bounds[:, 1],
        color="red",
        alpha=0.2,
        label="Predator Uncertainty",
    )

    # Final plot adjustments
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.legend()
    plt.title("Posterior Predictions with Uncertainty")
    plt.show()


# Run the demo
if __name__ == "__main__":
    demo()
