import os
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

plt.style.use("bmh")


# --- Data Generation ---
def generate_synthetic_data(rng_key, start, stop, num, scale):
    x = jnp.linspace(start=start, stop=stop, num=num)
    y = jnp.sin(4 * jnp.pi * x) + jnp.sin(7 * jnp.pi * x)
    y_obs = y + scale * random.normal(rng_key, shape=(num,))
    return x, y, y_obs


def center_data(x_train, x_test):
    train_mean = x_train.mean()
    return x_train - train_mean, x_test - train_mean


# --- Model Definition ---
def model(x, ell, m, non_centered, y=None):
    alpha = numpyro.sample("alpha", dist.InverseGamma(concentration=12, rate=10))
    length = numpyro.sample("length", dist.InverseGamma(concentration=6, rate=1))
    noise = numpyro.sample("noise", dist.InverseGamma(concentration=12, rate=10))
    f = hsgp_squared_exponential(
        x=x, alpha=alpha, length=length, ell=ell, m=m, non_centered=non_centered
    )
    with numpyro.plate("data", x.shape[0]):
        numpyro.sample("likelihood", dist.Normal(loc=f, scale=noise), obs=y)


# --- Sampling and Inference ---
def run_inference(rng_key, x_train, y_train_obs, ell, m, non_centered):
    sampler = NUTS(model)
    mcmc = MCMC(sampler, num_warmup=1_000, num_samples=2_000, num_chains=2)
    mcmc.run(rng_key, x_train, ell, m, non_centered, y_train_obs)
    idata = az.from_numpyro(posterior=mcmc)
    return idata, mcmc


def compute_predictive(mcmc, rng_key, x_test, ell, m, non_centered):
    predictive = Predictive(model, mcmc.get_samples())
    return predictive(rng_key, x_test, ell, m, non_centered)


# --- Visualization ---
def plot_synthetic_data(x_train, y_train, y_train_obs, x_test, y_test_obs):
    fig, ax = plt.subplots()
    ax.scatter(x_train, y_train_obs, c="C0", label="observed (train)")
    ax.scatter(x_test, y_test_obs, c="C1", label="observed (test)")
    ax.plot(x_train, y_train, color="black", linewidth=3, label="mean (latent)")
    ax.axvline(x=0, color="C2", alpha=0.8, linestyle="--", linewidth=2)
    ax.axvline(
        x=1, color="C2", linestyle="--", alpha=0.8, linewidth=2, label="training range"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    ax.set(xlabel="x", ylabel="y")
    ax.set_title("Synthetic Data", fontsize=16, fontweight="bold")
    plt.show()


def plot_posterior(idata):
    az.summary(idata, var_names=["alpha", "length", "noise"])
    az.plot_trace(idata, var_names=["alpha", "length", "noise"], compact=True)
    plt.gcf().suptitle("Posterior Distributions", fontsize=16, fontweight="bold")
    plt.show()


def plot_predictive(idata, x_test, y_test_obs, y_train_obs, x_train, y_train):
    fig, ax = plt.subplots()
    az.plot_hdi(
        x_test,
        idata.posterior_predictive["likelihood"],
        hdi_prob=0.94,
        color="C1",
        fill_kwargs={"alpha": 0.1},
        ax=ax,
    )
    ax.scatter(x_train, y_train_obs, c="C0", label="observed (train)")
    ax.scatter(x_test, y_test_obs, c="C1", label="observed (test)")
    ax.plot(
        x_train, y_train, color="black", linewidth=3, alpha=0.7, label="mean (latent)"
    )
    ax.set_title("Posterior Predictive", fontsize=16, fontweight="bold")
    plt.show()


def plot_laplacian_eigenfunctions(x_train_centered, ell, m):
    basis = eigenfunctions(x=x_train_centered, ell=ell, m=m)
    fig, ax = plt.subplots()
    ax.plot(x_train_centered, basis)
    ax.set(xlabel="x_centered")
    ax.set_title("Laplacian Eigenfunctions", fontsize=16, fontweight="bold")
    plt.show()


def plot_spectral_density(idata, ell, m):
    # Extract posterior means of alpha and length
    alpha_mean = idata.posterior["alpha"].mean(dim=("chain", "draw")).item()
    length_mean = idata.posterior["length"].mean(dim=("chain", "draw")).item()

    fig, ax = plt.subplots()
    ax.set(xlabel="index eigenvalue (sorted)", ylabel="spectral density")

    # Plot spectral density for specific values of alpha and length
    for alpha_value in (1.0, 1.5):
        for length_value in (0.05, 0.1):
            diag_sd = diag_spectral_density_squared_exponential(
                alpha=alpha_value,
                length=length_value,
                ell=ell,
                m=m,
                dim=1,
            )
            ax.plot(
                range(1, m + 1),
                diag_sd,
                marker="o",
                linewidth=1.5,
                markersize=4,
                alpha=0.8,
                label=f"alpha = {alpha_value}, length = {length_value}",
            )

    # Plot spectral density using posterior means
    diag_sd = diag_spectral_density_squared_exponential(
        alpha=alpha_mean,
        length=length_mean,
        ell=ell,
        m=m,
        dim=1,
    )
    ax.plot(
        range(1, m + 1),
        diag_sd,
        marker="o",
        color="black",
        linewidth=3,
        markersize=6,
        label=f"posterior mean (alpha = {alpha_mean:.2f}, length = {length_mean:.2f})",
    )
    ax.xaxis.set_major_locator(MultipleLocator())
    ax.legend(loc="upper right", title="Hyperparameters")
    ax.set_title(
        r"Spectral Density on the First $m$ (square root) Eigenvalues",
        fontsize=16,
        fontweight="bold",
    )
    plt.show()


# --- Main Script ---
if __name__ == "__main__":
    rng_key = random.PRNGKey(0)
    n_train, n_test = 80, 100
    scale = 0.3

    rng_key, rng_subkey = random.split(rng_key)
    x_train, y_train, y_train_obs = generate_synthetic_data(
        rng_subkey, 0, 1, n_train, scale
    )

    rng_key, rng_subkey = random.split(rng_key)
    x_test, y_test, y_test_obs = generate_synthetic_data(
        rng_subkey, -0.2, 1.2, n_test, scale
    )

    x_train_centered, x_test_centered = center_data(x_train, x_test)

    # Run Model
    ell, m, non_centered = 0.8, 20, True
    rng_key, rng_subkey = random.split(rng_key)
    idata, mcmc = run_inference(
        rng_subkey, x_train_centered, y_train_obs, ell, m, non_centered
    )

    # Predictions
    posterior_predictive = compute_predictive(
        mcmc, rng_subkey, x_test_centered, ell, m, non_centered
    )
    idata.extend(az.from_numpyro(posterior_predictive=posterior_predictive))

    # Visualization
    plot_synthetic_data(x_train, y_train, y_train_obs, x_test, y_test_obs)
    plot_posterior(idata)
    plot_predictive(idata, x_test, y_test_obs, y_train_obs, x_train, y_train)
    plot_laplacian_eigenfunctions(x_train_centered, ell, m)
    plot_spectral_density(idata, ell, m)
