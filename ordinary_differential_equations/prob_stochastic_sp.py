import numpy as np
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_sample
from jax import random


# Stochastic Spatial Lotka-Volterra Model with Probabilistic Inference
def stochastic_spatial_lv(
    grid_size=50,
    time_steps=500,
    dt=0.1,
    alpha=1.0,
    beta=0.1,
    gamma=1.5,
    delta=0.075,
    D_u=0.1,
    D_v=0.05,
    noise_strength=0.01,
):
    """
    Simulates the stochastic spatial Lotka-Volterra predator-prey model on a 2D grid.

    Parameters:
        grid_size (int): Size of the 2D grid.
        time_steps (int): Number of time steps.
        dt (float): Time step size.
        alpha (float): Prey birth rate.
        beta (float): Predation rate.
        gamma (float): Predator death rate.
        delta (float): Predator reproduction rate.
        D_u (float): Diffusion coefficient for prey.
        D_v (float): Diffusion coefficient for predator.
        noise_strength (float): Strength of stochastic noise.

    Returns:
        u, v (ndarray, ndarray): Final prey and predator densities.
    """
    # Initialize prey and predator densities
    u = np.random.rand(grid_size, grid_size)
    v = np.random.rand(grid_size, grid_size)

    # Laplacian operator for diffusion
    def laplacian(Z):
        return (
            -4 * Z
            + np.roll(Z, 1, axis=0)
            + np.roll(Z, -1, axis=0)
            + np.roll(Z, 1, axis=1)
            + np.roll(Z, -1, axis=1)
        )

    # Time evolution
    for t in range(time_steps):
        # Compute reaction terms
        reaction_u = alpha * u - beta * u * v
        reaction_v = -gamma * v + delta * u * v

        # Compute diffusion terms
        diffusion_u = D_u * laplacian(u)
        diffusion_v = D_v * laplacian(v)

        # Add stochastic noise
        noise_u = noise_strength * np.random.normal(size=(grid_size, grid_size))
        noise_v = noise_strength * np.random.normal(size=(grid_size, grid_size))

        # Update densities
        u += dt * (reaction_u + diffusion_u + noise_u)
        v += dt * (reaction_v + diffusion_v + noise_v)

        # Enforce non-negativity
        u = np.clip(u, 0, None)
        v = np.clip(v, 0, None)

    return u, v


def generate_synthetic_data():
    """Generate synthetic data for the stochastic Lotka-Volterra model."""
    grid_size = 20
    time_steps = 200
    u, v = stochastic_spatial_lv(
        grid_size=grid_size,
        time_steps=time_steps,
        alpha=1.0,
        beta=0.1,
        gamma=1.5,
        delta=0.075,
        D_u=0.1,
        D_v=0.05,
        noise_strength=0.01,
    )
    return u, v


def lotka_volterra_model(observed_u, observed_v):
    """Define a probabilistic model for inferring parameters of the Lotka-Volterra system."""
    alpha = numpyro.sample("alpha", dist.Uniform(0.5, 2.0))
    beta = numpyro.sample("beta", dist.Uniform(0.01, 0.2))
    gamma = numpyro.sample("gamma", dist.Uniform(1.0, 2.0))
    delta = numpyro.sample("delta", dist.Uniform(0.01, 0.1))

    # Observed data likelihood
    numpyro.sample(
        "obs_u", dist.Normal(loc=alpha - beta * observed_v, scale=0.1), obs=observed_u
    )
    numpyro.sample(
        "obs_v",
        dist.Normal(loc=-gamma * observed_v + delta * observed_u, scale=0.1),
        obs=observed_v,
    )


def run_inference(observed_u, observed_v):
    """Run MCMC to infer Lotka-Volterra parameters."""
    nuts_kernel = NUTS(lotka_volterra_model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=2000)
    mcmc.run(random.PRNGKey(0), observed_u=observed_u, observed_v=observed_v)
    return mcmc.get_samples()


def main():
    # Generate synthetic data
    observed_u, observed_v = generate_synthetic_data()

    # Run inference
    posterior_samples = run_inference(observed_u.flatten(), observed_v.flatten())

    # Print inferred parameters
    for param, samples in posterior_samples.items():
        print(f"{param}: mean={samples.mean()}, std={samples.std()}")

    # Visualize the data
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(observed_u, cmap="viridis")
    plt.colorbar()
    plt.title("Prey Density")

    plt.subplot(1, 2, 2)
    plt.imshow(observed_v, cmap="plasma")
    plt.colorbar()
    plt.title("Predator Density")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
