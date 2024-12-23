import matplotlib.pyplot as plt
import jax.numpy as jnp
from scipy.stats import norm


def plot_svi_loss(losses):
    """Plots the SVI loss curve."""
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Convergence of SVI")
    plt.show()


def plot_gmm_density(data, weights, locs, scale):
    """Plots the GMM density."""
    x = jnp.linspace(-3, 15, 500)
    density = sum(w * norm.pdf(x, loc, scale) for w, loc in zip(weights, locs))
    plt.plot(x, density, label="GMM Density")
    plt.scatter(data, jnp.zeros_like(data), alpha=0.5, label="Data")
    plt.legend()
    plt.show()


def plot_posterior_density(posterior_samples):
    """Plots the joint posterior density of loc parameters using contour plots."""
    X, Y = posterior_samples["locs"].T  # Extract loc samples
    h, x_edges, y_edges = jnp.histogram2d(X, Y, bins=[20, 20])
    plt.figure(figsize=(8, 8), dpi=100).set_facecolor("white")
    plt.hist2d(X, Y, bins=[x_edges, y_edges], cmap="Blues", density=True)
    plt.contour(
        jnp.log(h + 3).T,
        extent=[x_edges.min(), x_edges.max(), y_edges.min(), y_edges.max()],
        colors="white",
        alpha=0.8,
    )
    plt.title("Posterior density as estimated by collapsed NUTS")
    plt.xlabel("loc of component 0")
    plt.ylabel("loc of component 1")
    plt.tight_layout()
    plt.show()


def plot_trace(posterior_samples):
    """Plots the trace of loc parameters during NUTS inference."""
    X, Y = posterior_samples["locs"].T  # Extract loc samples
    plt.figure(figsize=(12, 4))
    plt.plot(X, label="loc[0]", color="red", alpha=0.8)
    plt.plot(Y, label="loc[1]", color="blue", alpha=0.8)
    plt.title("Trace plot of loc parameters during NUTS inference")
    plt.xlabel("NUTS step")
    plt.ylabel("loc value")
    plt.legend()
    plt.tight_layout()
    plt.show()
