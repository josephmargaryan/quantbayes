import numpy as np
from gan_sde import run_sde


def fake_data_test():

    # Simulated real-world data
    time_steps = np.linspace(0, 63, 64)
    num_trajectories = 100  # Number of independent trajectories

    # Generate 10 different sine waves with added noise
    observations = np.sin(time_steps[:, None]) + np.random.normal(  # Base sine wave
        0, 0.1, (64, num_trajectories)
    )  # Add noise

    # Format `real_data` for multiple trajectories
    real_data = (
        np.tile(time_steps[None, :], (num_trajectories, 1)),  # Duplicate time steps
        observations.T[:, :, None],  # Shape: (10, 64, 1)
    )
    return real_data


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


