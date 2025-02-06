import numpy as np
import jax.numpy as jnp
import jax.random as jr
import diffrax

from quantbayes.stochax.gan_sde import run_sde


def fake_data_test(key=None):
    """
    Generates synthetic time-series data based on a stochastic differential equation (SDE).

    This function simulates multiple independent trajectories of a stochastic process
    using an Ornstein-Uhlenbeck-like SDE with drift and diffusion.

    Returns:
        tuple (numpy.ndarray, numpy.ndarray):
            - ts (shape: (num_trajectories, t_size)):
              Represents discrete time steps, duplicated across all trajectories.
              In real-world applications, this could correspond to timestamps in
              financial, weather, or biomedical datasets (e.g., stock prices over time,
              temperature readings, or physiological signals).

            - ys (shape: (num_trajectories, t_size, 1)):
              The actual observed values at each time step for each trajectory.
              Examples include multiple stock prices evolving stochastically,
              temperature fluctuations, or biological signal variations.

    Example:
        >>> ts, ys = fake_data_test()
        >>> print(ts.shape)  # (100, 64)
        >>> print(ys.shape)  # (100, 64, 1)

    """
    if key is None:
        key = jr.PRNGKey(42)  # Default seed for reproducibility

    # Define SDE parameters
    mu = 0.02  # Mean reversion level
    theta = 0.1  # Strength of reversion
    sigma = 0.4  # Volatility

    t0, t1 = 0, 63  # Start and end times
    t_size = 64  # Number of time steps
    num_trajectories = 100  # Number of independent trajectories

    # Define drift and diffusion functions
    def drift(t, y, args):
        return mu * t - theta * y

    def diffusion(t, y, args):
        return 2 * sigma * t / t1

    # Create an array to store trajectories
    trajectories = np.zeros((num_trajectories, t_size))

    # Generate multiple stochastic trajectories
    for i in range(num_trajectories):
        bm_key, y0_key = jr.split(jr.fold_in(key, i), 2)

        # Initialize Brownian motion and solver
        bm = diffrax.UnsafeBrownianPath(shape=(), key=bm_key)
        drift_term = diffrax.ODETerm(drift)
        diffusion_term = diffrax.ControlTerm(diffusion, bm)
        terms = diffrax.MultiTerm(drift_term, diffusion_term)
        solver = diffrax.Euler()

        # Initial condition
        y0 = jr.uniform(y0_key, (1,), minval=-1, maxval=1)

        # Time steps
        ts = jnp.linspace(t0, t1, t_size)
        saveat = diffrax.SaveAt(ts=ts)

        # Solve SDE
        sol = diffrax.diffeqsolve(
            terms,
            solver,
            t0,
            t1,
            dt0=0.1,
            y0=y0,
            saveat=saveat,
            adjoint=diffrax.DirectAdjoint(),
        )

        # Store trajectory
        trajectories[i] = np.array(sol.ys).flatten()

    # Format the final output
    ts = np.tile(np.linspace(t0, t1, t_size), (num_trajectories, 1))  # Shape: (100, 64)
    ys = trajectories[:, :, None]  # Shape: (100, 64, 1)

    return ts, ys


fake_data = fake_data_test()


df = run_sde(
    real_data=fake_data,  # Required parameter: Tuple of (ts, ys)
    initial_noise_size=5,
    noise_size=3,
    hidden_size=16,
    width_size=16,
    depth=1,
    generator_lr=2e-5,
    discriminator_lr=1e-4,
    batch_size=10,  # Adjust to match dataset size
    steps=10,
    steps_per_print=1,
    seed=5678,
    max_plot_trajectories=30,
)
