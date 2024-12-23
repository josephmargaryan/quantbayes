import numpy as np
import matplotlib.pyplot as plt


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

        # Optional: Visualization during simulation
        if t % 50 == 0:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(u, cmap="viridis")
            plt.colorbar()
            plt.title(f"Prey Density (t={t})")

            plt.subplot(1, 2, 2)
            plt.imshow(v, cmap="plasma")
            plt.colorbar()
            plt.title(f"Predator Density (t={t})")

            plt.tight_layout()
            plt.show()

    return u, v


if __name__ == "__main__":
    # Parameters for the simulation
    final_u, final_v = stochastic_spatial_lv(
        grid_size=100, time_steps=1000, dt=0.1, noise_strength=0.02
    )

    # Final visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(final_u, cmap="viridis")
    plt.colorbar()
    plt.title("Final Prey Density")

    plt.subplot(1, 2, 2)
    plt.imshow(final_v, cmap="plasma")
    plt.colorbar()
    plt.title("Final Predator Density")

    plt.tight_layout()
    plt.show()
