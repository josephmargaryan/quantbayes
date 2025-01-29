import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import flax_module, random_flax_module
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from sklearn.metrics import mean_squared_error

plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"

# -----------------------------------
# Data Generation
# -----------------------------------
from quantbayes.fake_data import generate_regression_data

df = generate_regression_data()
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
scaler = MinMaxScaler()
y = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)


# Generate and preprocess data
df = generate_regression_data(n_samples=1000, n_features=1, random_seed=42)
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)


# -----------------------------------
# Neural Network Definition
# -----------------------------------
class Classification(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=16)(x)  # First hidden layer
        x = nn.relu(x)
        x = nn.Dense(features=16)(x)  # Second hidden layer
        x = nn.relu(x)
        logits = nn.Dense(features=self.num_classes)(
            x
        )  # Output logits for binary classification
        return logits


class MuNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=16)(x)  # First hidden layer
        x = nn.relu(x)
        x = nn.Dense(features=16)(x)  # Second hidden layer
        x = nn.relu(x)
        mu = nn.Dense(features=1)(x)  # Output layer for mean
        return mu


class SigmaNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=8)(x)  # First hidden layer
        x = nn.relu(x)
        x = nn.Dense(features=4)(x)  # Second hidden layer
        x = nn.relu(x)
        sigma = nn.softplus(
            nn.Dense(features=1)(x)
        )  # Output layer for sigma with softplus
        return sigma


# -----------------------------------
# Probabilistic Model
# -----------------------------------
def model(x, y=None):
    mu_nn = flax_module("mu_nn", MuNetwork(), input_shape=(1,))
    log_sigma_nn = flax_module("sigma_nn", SigmaNetwork(), input_shape=(1,))

    mu = numpyro.deterministic("mu", mu_nn(x).squeeze())
    sigma = numpyro.deterministic("sigma", jnp.exp(log_sigma_nn(x)).squeeze())

    with numpyro.plate("data", len(x)):
        numpyro.sample("y", dist.Normal(loc=mu, scale=sigma), obs=y)


# -----------------------------------
# Inference and Training
# -----------------------------------
guide = AutoNormal(model)
optimizer = numpyro.optim.Adam(step_size=0.01)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
n_samples = 5000

rng_key = random.PRNGKey(42)
svi_result = svi.run(rng_key, n_samples, X_train, y_train)

# -----------------------------------
# Posterior Predictive Checks
# -----------------------------------
params = svi_result.params
predictive = Predictive(model, guide=guide, params=params, num_samples=100)
posterior_predictive = predictive(rng_key, X_test)
print(f"All the posterior approximations: {posterior_predictive.keys()}")
posteriors = posterior_predictive["y"]


def visualize(X_test, y_test, posteriors, feature_index=None):
    """
    Visualize predictions with uncertainty bounds and true targets.
    """
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    mean_preds = np.array(posteriors.mean(axis=0))
    lower_bound = np.percentile(posteriors, 2.5, axis=0)
    upper_bound = np.percentile(posteriors, 97.5, axis=0)

    if (
        X_test.shape[1] == 1
        or feature_index is None
        or not (0 <= feature_index < X_test.shape[1])
    ):
        feature_index = 0

    feature = X_test[:, feature_index]
    sorted_indices = np.argsort(feature)
    feature = feature[sorted_indices]
    y_test = y_test[sorted_indices]
    mean_preds = mean_preds[sorted_indices]
    lower_bound = lower_bound[sorted_indices]
    upper_bound = upper_bound[sorted_indices]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(feature, y_test, color="blue", alpha=0.6, label="True Targets")
    plt.plot(feature, mean_preds, color="red", label="Mean Predictions", linestyle="-")
    plt.fill_between(
        feature,
        lower_bound,
        upper_bound,
        color="gray",
        alpha=0.3,
        label="Uncertainty Bounds",
    )
    plt.xlabel(f"Feature {feature_index + 1}")
    plt.ylabel("Target (y_test)")
    plt.title("Model Predictions with Uncertainty and True Targets")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.show()


visualize(X_test, y_test, posteriors, feature_index=0)

mean_preds = posteriors.mean(axis=0)
mse = mean_squared_error(np.array(y_test), np.array(mean_preds))
print(f"MSE: {mse}")


"""
For models with dropout we need o use the apply_rng

model_config = {
    'in_channels': 1,
    'img_size': 28,
    'patch_size': 4,
    'emb_dim': 768,
    'num_layers': 12,
    'num_heads': 12,
    'mlp_dim': 3072,
    'num_classes': 10,
    'dropout_rate': 0.1
}

vit = VisionTransformer(**model_config)
def model(X, y=None, **kwargs):
    assert X.shape[1:] == (28, 28, 1), f"Expected (28, 28, 1), got {X.shape}"
    test = flax_module(
        "vit", vit, input_shape=(1, 28, 28, 1), apply_rng=["dropout"]
    )
    logits = numpyro.deterministic("logits", test(X, rngs={"dropout": numpyro.prng_key()}))
    with numpyro.plate("data", len(X)):
        numpyro.sample("obs", dist.Categorical(logits=logits), obs=y)


dummy_input = jnp.ones((1, 28, 28, 1))  # Single grayscale image

# Initialize model parameters
rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
params = vit.init(rngs, dummy_input)

# Training mode (dropout enabled)
train_logits = vit.apply({'params': params['params']}, dummy_input, train=True, rngs={'dropout': random.PRNGKey(2)})
print(f"Logits (train): {train_logits.shape}")

# Evaluation mode (dropout disabled)
eval_logits = vit.apply({'params': params['params']}, dummy_input, train=False)
print(f"Logits (eval): {eval_logits.shape}")

"""
