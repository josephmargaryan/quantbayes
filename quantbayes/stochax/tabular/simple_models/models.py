import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Sequence
import jax.random as jr

# -------------------------------------------------------------------
# 1. Linear Regression Model
#
# A simple linear model for regression.
# -------------------------------------------------------------------
class LinearRegression(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, input_dim: int, *, key):
        # Map from input_dim to 1 output (the regression target).
        self.linear = eqx.nn.Linear(input_dim, 1, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x should be a vector (or batched, if vmap is applied externally).
        return self.linear(x)


# -------------------------------------------------------------------
# 2. Logistic Regression Model
#
# A single-layer model that outputs a probability for binary classification.
# -------------------------------------------------------------------
class LogisticRegression(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, input_dim: int, *, key):
        self.linear = eqx.nn.Linear(input_dim, 1, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Compute logits and apply sigmoid.
        logits = self.linear(x)
        return jax.nn.sigmoid(logits)


# -------------------------------------------------------------------
# 3. MLP Classifier for Multiclass Classification
#
# A feed-forward network with one or more hidden layers.
# -------------------------------------------------------------------
class MLPClassifier(eqx.Module):
    layers: Sequence[eqx.nn.Linear]
    activation: Callable = eqx.field(static=True)

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], num_classes: int, *, key):
        # Create a list of layers.
        # We use as many hidden layers as given in hidden_dims and a final layer mapping to num_classes.
        keys = jax.random.split(key, len(hidden_dims) + 1)
        layers = []
        current_dim = input_dim
        for h, k in zip(hidden_dims, keys[:-1]):
            layers.append(eqx.nn.Linear(current_dim, h, key=k))
            current_dim = h
        # Final layer:
        layers.append(eqx.nn.Linear(current_dim, num_classes, key=keys[-1]))
        self.layers = layers
        self.activation = jax.nn.relu

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: shape (..., input_dim)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


# -------------------------------------------------------------------
# Example usage: Testing the models
# -------------------------------------------------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    # For illustration, assume input features are 10-dimensional.
    input_dim = 10

    # Generate a random input (for a single sample).
    x = jax.random.normal(key, (input_dim,))

    # --- Linear Regression ---
    lr_key, key = jr.split(key)
    lin_reg = LinearRegression(input_dim, key=lr_key)
    y_reg = lin_reg(x)
    print("Linear Regression output shape:", y_reg.shape)  # Expected: (1,)

    # --- Logistic Regression ---
    logreg_key, key = jr.split(key)
    log_reg = LogisticRegression(input_dim, key=logreg_key)
    y_log = log_reg(x)
    print("Logistic Regression output (probability):", y_log)

    # --- MLP Classifier for Multiclass Classification ---
    # Suppose we have 5 classes.
    num_classes = 5
    hidden_dims = [32, 16]  # two hidden layers
    mlp_key, key = jr.split(key)
    mlp_model = MLPClassifier(input_dim, hidden_dims, num_classes, key=mlp_key)
    logits = mlp_model(x)
    print("MLP Classifier logits shape:", logits.shape)  # Expected: (5,)
