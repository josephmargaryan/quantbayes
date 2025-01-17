from collections import defaultdict
import os
import matplotlib.pyplot as plt
import scipy.stats

from jax import pure_callback, random
import jax.numpy as jnp
import optax

import numpyro
from numpyro import handlers
from numpyro.contrib.funsor import config_enumerate, infer_discrete
import numpyro.distributions as dist
from numpyro.infer import SVI, TraceEnum_ELBO, init_to_value
from numpyro.infer.autoguide import AutoDelta

# Define constants
K = 2  # Number of components
elbo = TraceEnum_ELBO()


def generate_synthetic_data(rng_key):
    """Generate synthetic data from two Gaussian components."""
    component_1 = random.normal(rng_key, (50,)) + 3  # Mean = 3
    component_2 = random.normal(rng_key, (50,)) + 8  # Mean = 8
    return jnp.concatenate([component_1, component_2])


def model(data):
    """Defines the Bayesian Gaussian Mixture Model."""
    weights = numpyro.sample("weights", dist.Dirichlet(0.5 * jnp.ones(K)))
    scale = numpyro.sample("scale", dist.LogNormal(0.0, 2.0))
    with numpyro.plate("components", K):
        locs = numpyro.sample("locs", dist.Normal(0.0, 10.0))

    with numpyro.plate("data", len(data)):
        assignment = numpyro.sample("assignment", dist.Categorical(weights))
        numpyro.sample("obs", dist.Normal(locs[assignment], scale), obs=data)


def initialize(data, seed):
    """Initialize the guide with the best parameters."""
    init_values = {
        "weights": jnp.ones(K) / K,
        "scale": jnp.sqrt(data.var() / 2),
        "locs": data[
            random.categorical(
                random.PRNGKey(seed), jnp.ones(len(data)) / len(data), shape=(K,)
            )
        ],
    }
    global_model = handlers.block(
        handlers.seed(model, random.PRNGKey(0)),
        hide_fn=lambda site: site["name"]
        not in ["weights", "scale", "locs", "components"],
    )
    global_guide = AutoDelta(
        global_model, init_loc_fn=init_to_value(values=init_values)
    )
    handlers.seed(global_guide, random.PRNGKey(0))(data)  # Warm up the guide
    return elbo.loss(random.PRNGKey(0), {}, model, global_guide, data), global_guide


def choose_best_initialization(data):
    """Choose the best initialization over multiple seeds."""
    loss, seed, guide = min(
        (initialize(data, seed)[0], seed, initialize(data, seed)[1])
        for seed in range(100)
    )
    print(f"seed = {seed}, initial_loss = {loss}")
    return guide


def hook_optax(optimizer):
    """Collect gradient norms during training."""
    gradient_norms = defaultdict(list)

    def append_grad(grad):
        for name, g in grad.items():
            gradient_norms[name].append(float(jnp.linalg.norm(g)))
        return grad

    def update_fn(grads, state, params=None):
        grads = pure_callback(append_grad, grads, grads)
        return optimizer.update(grads, state, params=params)

    return optax.GradientTransformation(optimizer.init, update_fn), gradient_norms


def visualize_convergence(losses):
    """Plot the convergence of the SVI loss."""
    plt.figure(figsize=(10, 3), dpi=100).set_facecolor("white")
    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Convergence of SVI")
    plt.show()


def visualize_density(data, weights, locs, scale):
    """Visualize the density of the two-component mixture model."""
    X = jnp.arange(-3, 15, 0.1)
    Y1 = weights[0] * scipy.stats.norm.pdf((X - locs[0]) / scale)
    Y2 = weights[1] * scipy.stats.norm.pdf((X - locs[1]) / scale)

    plt.figure(figsize=(10, 4), dpi=100).set_facecolor("white")
    plt.plot(X, Y1, "r-", label="Component 1")
    plt.plot(X, Y2, "b-", label="Component 2")
    plt.plot(X, Y1 + Y2, "k--", label="Mixture")
    plt.plot(data, jnp.zeros(len(data)), "k*", label="Data")
    plt.title("Density of Two-Component Mixture Model")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.show()


def main():
    rng_key = random.PRNGKey(0)
    data = generate_synthetic_data(rng_key)

    # Initialize the guide
    global_guide = choose_best_initialization(data)

    # Set up optimizer and SVI
    optim, gradient_norms = hook_optax(optax.adam(learning_rate=0.1, b1=0.8, b2=0.99))
    global_svi = SVI(model, global_guide, optim, loss=elbo)

    # Run SVI to fit the model
    global_svi_result = global_svi.run(random.PRNGKey(0), 200, data=data)

    # Visualize convergence
    visualize_convergence(global_svi_result.losses)

    # Extract MAP estimates
    map_estimates = global_svi_result.params
    weights = map_estimates["weights_auto_loc"]
    locs = map_estimates["locs_auto_loc"]
    scale = map_estimates["scale_auto_loc"]

    print(f"weights = {weights}")
    print(f"locs = {locs}")
    print(f"scale = {scale}")

    # Visualize density
    visualize_density(data, weights, locs, scale)


if __name__ == "__main__":
    main()
