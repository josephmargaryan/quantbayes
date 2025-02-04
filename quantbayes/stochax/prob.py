import jax
import jax.numpy as jnp
import equinox as eqx
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import matplotlib.pyplot as plt
from jax import random

eqx.nn.BatchNorm
eqx.nn.Linear
eqx.nn.Conv2d
eqx.nn.Conv1d
eqx.nn.LSTMCell
eqx.nn.MultiheadAttention
eqx.nn.Dropout

# === Step 1: Define the Bayesian Equinox MLP Model ===
class MLP(eqx.Module):
    layers: list

    def __init__(self, key, in_dim=1, hidden_dim=64, out_dim=1, depth=2):
        keys = random.split(key, num=depth+1)
        self.layers = [eqx.nn.Linear(in_dim, hidden_dim, key=keys[0])]
        self.layers += [eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[i]) for i in range(1, depth)]
        self.layers += [eqx.nn.Linear(hidden_dim, out_dim, key=keys[-1])]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)


# === Step 2: Generate Synthetic Data ===
def generate_data(n_samples=100):
    key = random.PRNGKey(42)
    x = jax.random.uniform(key, shape=(n_samples, 1)) * 10 - 5  # Range: [-5,5]
    y = jnp.sin(x) + 0.1 * jax.random.normal(key, shape=x.shape)  # Noisy sine wave
    return x, y


# === Step 3: Define the Probabilistic Model in NumPyro ===
def model(x, y=None):
    key = random.PRNGKey(0)

    # Define a normal prior for each weight matrix
    def prior(name, shape):
        return dist.Normal(0, 1).expand(shape).to_event(len(shape))

    # Create a deterministic MLP
    mlp = MLP(key)

    # Replace weights in the model with sampled priors
    sampled_weights = [
        numpyro.sample(f"w_{i}", prior(f"w_{i}", layer.weight.shape))
        for i, layer in enumerate(mlp.layers)
    ]

    # Apply sampled weights into an Equinox model
    bayesian_mlp = eqx.tree_at(
        lambda m: [m.layers[i].weight for i in range(len(m.layers))],
        mlp,
        replace=sampled_weights
    )

    # Forward pass through the Bayesian model
    pred = jax.vmap(bayesian_mlp)(x)

    # Likelihood
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    numpyro.sample("obs", dist.Normal(pred, sigma), obs=y)


# === Step 4: Run Bayesian Inference ===
x_train, y_train = generate_data(100)
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=200, num_chains=1)
mcmc.run(random.PRNGKey(1), x_train, y_train)
posterior_samples = mcmc.get_samples()


# === Step 5: Generate Predictions and Visualize Uncertainty ===
predictive = Predictive(model, posterior_samples, num_samples=1000)
posterior_preds = predictive(random.PRNGKey(2), x_train)["obs"]

# Mean and uncertainty bounds
y_pred_mean = posterior_preds.mean(axis=0)
y_pred_std = posterior_preds.std(axis=0)

# Sort x_train and rearrange predictions
sorted_indices = jnp.argsort(x_train.flatten())
x_train_sorted = x_train.flatten()[sorted_indices]
# Ensure predictions are 1D by flattening them
y_pred_mean_sorted = y_pred_mean[sorted_indices].flatten()
y_pred_std_sorted = y_pred_std[sorted_indices].flatten()

# Plot with sorted data
plt.figure(figsize=(8, 5))
plt.scatter(x_train_sorted, y_train[sorted_indices], label="Train Data", color="blue", alpha=0.6)
plt.plot(x_train_sorted, y_pred_mean_sorted, label="Posterior Mean", color="red", linewidth=2)
plt.fill_between(
    x_train_sorted,
    (y_pred_mean_sorted - 2 * y_pred_std_sorted),
    (y_pred_mean_sorted + 2 * y_pred_std_sorted),
    color="red",
    alpha=0.3,
    label="Uncertainty (±2σ)"
)
plt.legend()
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.title("Bayesian Regression with Equinox & NumPyro")
plt.show()
