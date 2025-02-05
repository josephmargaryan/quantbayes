import jax
import jax.numpy as jnp
from jax import random, tree_util
import equinox as eqx
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# Helper: Bayesianize an Equinox module
# ------------------------------------------------------------------------------
def bayesianize(module: eqx.Module, prior_fn):
    """
    Traverse the module's pytree and, for every array leaf, replace it with a
    NumPyro sample drawn from `prior_fn` (a function accepting a shape and
    returning a distribution). Each sample is tagged with a unique name.
    """
    leaves, treedef = tree_util.tree_flatten(module)
    new_leaves = []
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, jnp.ndarray):
            # Draw a sample for this parameter using the provided prior function.
            new_leaf = numpyro.sample(f"param_{i}", prior_fn(leaf.shape))
            new_leaves.append(new_leaf)
        else:
            new_leaves.append(leaf)
    return tree_util.tree_unflatten(treedef, new_leaves)


# ------------------------------------------------------------------------------
# Helper: Prior function defined as a normal distribution
# ------------------------------------------------------------------------------
def prior_fn(shape):
    return dist.Normal(0, 1).expand(shape).to_event(len(shape))


# ------------------------------------------------------------------------------
# Step 1: Define the Deterministic MLP Model with Equinox
# ------------------------------------------------------------------------------
class MLP(eqx.Module):
    layers: list

    def __init__(self, key, in_dim=1, hidden_dim=64, out_dim=1, depth=2):
        keys = random.split(key, num=depth + 1)
        self.layers = [eqx.nn.Linear(in_dim, hidden_dim, key=keys[0])]
        self.layers += [
            eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[i]) for i in range(1, depth)
        ]
        self.layers += [eqx.nn.Linear(hidden_dim, out_dim, key=keys[-1])]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)


# ------------------------------------------------------------------------------
# Step 2: Generate Synthetic Data
# ------------------------------------------------------------------------------
def generate_data(n_samples=100):
    key = random.PRNGKey(42)
    x = random.uniform(key, shape=(n_samples, 1)) * 10 - 5  # Range: [-5,5]
    y = jnp.sin(x) + 0.1 * random.normal(key, shape=x.shape)  # Noisy sine wave
    return x, y


# ------------------------------------------------------------------------------
# Step 3: Define the Probabilistic Model in NumPyro using our Bayesianization
# ------------------------------------------------------------------------------
def model(x, y=None):
    # Use a fixed PRNGKey here; in practice you might want to pass it in.
    key = random.PRNGKey(0)

    # Create a deterministic MLP
    mlp = MLP(key)

    # Use our helper to replace all array leaves (the weights) with Bayesian samples.
    bayesian_mlp = bayesianize(mlp, prior_fn)

    # Forward pass through the Bayesianized model using jax.vmap over the input examples.
    pred = jax.vmap(bayesian_mlp)(x)

    # Likelihood: also put a prior on sigma.
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    numpyro.sample("obs", dist.Normal(pred, sigma), obs=y)


# ------------------------------------------------------------------------------
# Step 4: Run Bayesian Inference (MCMC) and Visualize the Results
# ------------------------------------------------------------------------------
def main():
    # Generate synthetic data.
    x_train, y_train = generate_data(100)

    # Set up MCMC using NUTS.
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=200, num_chains=1)
    mcmc.run(random.PRNGKey(1), x_train, y_train)
    posterior_samples = mcmc.get_samples()
    print("Posterior sample keys and shapes:")
    for k, v in posterior_samples.items():
        print(f"{k}: {v.shape}")

    # Use Predictive to get posterior predictions.
    predictive = Predictive(model, posterior_samples, num_samples=1000)
    pred_dict = predictive(random.PRNGKey(2), x_train)
    posterior_preds = pred_dict["obs"]

    # Calculate mean and standard deviation across posterior draws.
    y_pred_mean = posterior_preds.mean(axis=0)
    y_pred_std = posterior_preds.std(axis=0)

    # Sort x_train for plotting.
    sorted_indices = jnp.argsort(x_train.flatten())
    x_train_sorted = x_train.flatten()[sorted_indices]
    y_pred_mean_sorted = y_pred_mean[sorted_indices].flatten()
    y_pred_std_sorted = y_pred_std[sorted_indices].flatten()

    # Plot the data and posterior predictions.
    plt.figure(figsize=(8, 5))
    plt.scatter(
        x_train_sorted,
        y_train[sorted_indices],
        label="Train Data",
        color="blue",
        alpha=0.6,
    )
    plt.plot(
        x_train_sorted,
        y_pred_mean_sorted,
        label="Posterior Mean",
        color="red",
        linewidth=2,
    )
    plt.fill_between(
        x_train_sorted,
        y_pred_mean_sorted - 2 * y_pred_std_sorted,
        y_pred_mean_sorted + 2 * y_pred_std_sorted,
        color="red",
        alpha=0.3,
        label="Uncertainty (±2σ)",
    )
    plt.xlabel("Feature (X)")
    plt.ylabel("Target (y)")
    plt.title("Bayesian Regression with Equinox & NumPyro")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
