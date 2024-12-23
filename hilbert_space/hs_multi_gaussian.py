import arviz as az
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from jax import random
import jax.numpy as jnp
import numpyro
from numpyro.contrib.hsgp.approximation import hsgp_squared_exponential
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import (
    diag_spectral_density_squared_exponential,
)
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from matplotlib import cm

plt.style.use("bmh")


# --- Data Generation ---
def generate_multidimensional_data(rng_key, grid_size, dims, scale):
    """Generate synthetic data for multidimensional inputs."""
    coords = [jnp.linspace(-1, 1, grid_size) for _ in range(dims)]
    grid = jnp.stack(jnp.meshgrid(*coords), axis=-1).reshape(-1, dims)
    rng_key, subkey = random.split(rng_key)
    y = jnp.sin(jnp.sum(grid, axis=1)) + scale * random.normal(
        subkey, shape=(grid.shape[0],)
    )
    return grid, y


def train_test_split(grid, y, split_ratio=0.8):
    """Split data into train and test sets."""
    n_train = int(split_ratio * len(grid))
    return grid[:n_train], y[:n_train], grid[n_train:], y[n_train:]


# --- Model Definition ---
def model(x, ell, m, non_centered, y=None):
    """HSGP approximation model."""
    alpha = numpyro.sample("alpha", dist.InverseGamma(concentration=12, rate=10))
    length = numpyro.sample("length", dist.InverseGamma(concentration=6, rate=1))
    noise = numpyro.sample("noise", dist.InverseGamma(concentration=12, rate=10))
    f = hsgp_squared_exponential(
        x=x, alpha=alpha, length=length, ell=ell, m=m, non_centered=non_centered
    )
    with numpyro.plate("data", x.shape[0]):
        numpyro.sample("likelihood", dist.Normal(loc=f, scale=noise), obs=y)


# --- Sampling and Inference ---
def run_inference(rng_key, x_train, y_train, ell, m, non_centered):
    """Run MCMC for posterior inference."""
    sampler = NUTS(model)
    mcmc = MCMC(sampler, num_warmup=1_000, num_samples=2_000, num_chains=2)
    mcmc.run(rng_key, x_train, ell, m, non_centered, y_train)
    idata = az.from_numpyro(posterior=mcmc)
    return idata, mcmc


def compute_predictive(mcmc, rng_key, x_test, ell, m, non_centered):
    """Compute posterior predictive for test data."""
    predictive = Predictive(model, mcmc.get_samples())
    return predictive(rng_key, x_test, ell, m, non_centered)


# --- Visualization ---
def plot_3d_data(grid, y, title="Synthetic Data"):
    """Plot 3D scatter plot for data."""
    x, y, z = grid[:, 0], grid[:, 1], y
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=z, cmap="viridis", alpha=0.6)
    plt.colorbar(sc)
    ax.set_title(title)
    plt.show()


def plot_posterior(idata):
    """Plot posterior distributions."""
    az.summary(idata, var_names=["alpha", "length", "noise"])
    az.plot_trace(idata, var_names=["alpha", "length", "noise"], compact=True)
    plt.gcf().suptitle("Posterior Distributions", fontsize=16, fontweight="bold")
    plt.show()


def plot_laplacian_eigenfunctions(grid, ell, m):
    """Plot Laplacian eigenfunctions for multidimensional data."""
    basis = eigenfunctions(x=grid, ell=ell, m=m)
    fig, ax = plt.subplots()
    ax.plot(grid[:, 0], basis)
    ax.set(xlabel="Input Dimension 1")
    ax.set_title("Laplacian Eigenfunctions", fontsize=16, fontweight="bold")
    plt.show()


def plot_spectral_density(idata, ell, m):
    """Plot spectral density for the posterior mean parameters."""
    alpha_mean = idata.posterior["alpha"].mean(dim=("chain", "draw")).item()
    length_mean = idata.posterior["length"].mean(dim=("chain", "draw")).item()

    fig, ax = plt.subplots()
    ax.set(xlabel="Index Eigenvalue (sorted)", ylabel="Spectral Density")

    # Compute spectral density for the first `m` eigenvalues
    diag_sd = diag_spectral_density_squared_exponential(
        alpha=alpha_mean, length=length_mean, ell=ell, m=m, dim=2  # dim=2 for 2D data
    )[
        :m
    ]  # Take only the first `m` values to match the x-axis

    ax.plot(
        range(1, m + 1), diag_sd, marker="o", color="black", linewidth=3, markersize=6
    )
    ax.set_title(
        r"Spectral Density on the First $m$ (square root) Eigenvalues",
        fontsize=16,
        fontweight="bold",
    )
    plt.show()


def plot_surface_scatter_with_predictive(
    grid,
    y,
    posterior_predictive,
    credible_interval=0.8,
    title="Surface Scatter with Predictive Samples",
):
    """
    Visualize the surface scatter and posterior predictive samples with credible intervals.

    Args:
        grid (jax.Array): The grid points for the input.
        y (jax.Array): The true function values at the grid points.
        posterior_predictive (dict): Posterior predictive samples.
        credible_interval (float): The width of the credible interval (default 80%).
        title (str): Title of the plot.
    """
    # Extract predictive samples of the latent function
    latent_samples = posterior_predictive["likelihood"]  # Shape: (samples, num_points)
    predictive_mean = latent_samples.mean(axis=0)
    lower_bound = jnp.quantile(latent_samples, (1 - credible_interval) / 2, axis=0)
    upper_bound = jnp.quantile(latent_samples, 1 - (1 - credible_interval) / 2, axis=0)

    # Check if grid is 2D
    num_points = int(jnp.sqrt(len(grid)))
    if num_points**2 != len(grid):
        raise ValueError("Grid points do not form a complete 2D grid.")

    # Prepare grid for 3D plotting
    x_grid = grid[:, 0].reshape(num_points, num_points)
    y_grid = grid[:, 1].reshape(num_points, num_points)
    z = y.reshape(num_points, num_points)
    z_pred_mean = predictive_mean.reshape(num_points, num_points)
    z_lower = lower_bound.reshape(num_points, num_points)
    z_upper = upper_bound.reshape(num_points, num_points)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the true surface
    surface = ax.plot_surface(
        x_grid, y_grid, z, cmap=cm.viridis, alpha=0.5, edgecolor="none"
    )
    plt.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label="True Surface")

    # Plot posterior predictive mean
    ax.plot_wireframe(
        x_grid,
        y_grid,
        z_pred_mean,
        color="tab:blue",
        linewidth=1,
        label="Posterior Predictive Mean",
    )

    # Plot credible intervals
    ax.plot_wireframe(
        x_grid,
        y_grid,
        z_lower,
        color="tab:green",
        linewidth=0.5,
        linestyle="--",
        alpha=0.7,
    )
    ax.plot_wireframe(
        x_grid,
        y_grid,
        z_upper,
        color="tab:green",
        linewidth=0.5,
        linestyle="--",
        alpha=0.7,
    )

    # Scatter the original points
    scatter = ax.scatter(
        grid[:, 0],
        grid[:, 1],
        y,
        c=y,
        cmap=cm.viridis,
        edgecolor="k",
        label="Data Points",
    )

    # Add titles and labels
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("X1", fontsize=12)
    ax.set_ylabel("X2", fontsize=12)
    ax.set_zlabel("Y", fontsize=12)
    ax.legend()
    plt.show()


def plot_calibration(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    fig_size: float = 5.0,
    label_size: float = 8.0,
    point_size: float = 10.0,
    x_label: str = "True",
    y_label: str = "Predicted",
    title: str = "Calibration Plot",
):
    """
    Create a calibration plot to compare true values against predictions.

    Args:
        y_true (jnp.ndarray): True latent values.
        y_pred (jnp.ndarray): Predicted values (posterior predictive mean).
        fig_size (float): Size of the figure.
        label_size (float): Font size for labels.
        point_size (float): Size of scatter points.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Scatter plot of true vs predicted
    ax.scatter(
        y_true, y_pred, alpha=0.6, s=point_size, c="tab:blue", label="Predictions"
    )

    # Diagonal line (perfect calibration)
    ax.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        color="tab:orange",
        linestyle="--",
        label="Perfect Calibration",
    )

    # Set limits
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])

    # Labels and title
    ax.set_xlabel(x_label, fontsize=label_size)
    ax.set_ylabel(y_label, fontsize=label_size)
    ax.set_title(title, fontsize=label_size + 2, fontweight="bold")
    ax.legend()

    plt.show()


# --- Main Script ---
if __name__ == "__main__":
    # Parameters
    rng_key = random.PRNGKey(0)
    grid_size = 20
    dims = 2
    scale = 0.1
    ell, m, non_centered = 0.8, 20, True

    # Generate Data
    grid, y = generate_multidimensional_data(rng_key, grid_size, dims, scale)
    x_train, y_train, x_test, y_test = train_test_split(grid, y)

    # Run Inference
    rng_key, rng_subkey = random.split(rng_key)
    idata, mcmc = run_inference(rng_subkey, x_train, y_train, ell, m, non_centered)

    # Predictive Checks (use the full grid for predictions)
    posterior_predictive = compute_predictive(
        mcmc, rng_subkey, grid, ell, m, non_centered
    )
    idata.extend(az.from_numpyro(posterior_predictive=posterior_predictive))

    # Visualization
    plot_3d_data(grid, y, title="Synthetic Data")
    plot_posterior(idata)
    plot_spectral_density(idata, ell, m)
    plot_laplacian_eigenfunctions(x_train, ell, m)

    # Visualize surface scatter with posterior predictive samples
    plot_surface_scatter_with_predictive(
        grid,
        y,
        posterior_predictive,
        credible_interval=0.8,
        title="Surface Scatter with Predictive Samples",
    )

    # Extract true and predicted values
    y_true = y  # True latent values from the full grid
    y_pred = posterior_predictive["likelihood"].mean(
        axis=0
    )  # Posterior predictive mean

    # Create the calibration plot
    plot_calibration(
        y_true=y_true,
        y_pred=y_pred,
        fig_size=6.0,
        label_size=10.0,
        title="Calibration Plot: True vs Predicted",
    )
