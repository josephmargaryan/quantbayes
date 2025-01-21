import jax
import jax.numpy as jnp
import bnn
import numpyro
import numpyro.distributions as dist
from fake_data import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from numpyro.infer import MCMC, NUTS, Predictive
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = generate_regression_data()
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
scaler = MinMaxScaler()
y = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)


class Module:
    """
    A base model class to provide shared functionality for all models.
    """

    def __call__(self, X: jnp.ndarray, y: jnp.ndarray = None):
        raise NotImplementedError("Subclasses must implement the `__call__` method.")


class SimpleRegressionModel(Module):
    """
    A simple shallow regression model with one dense layer.
    """

    def __init__(self, hidden_dim: int):

        self.hidden_dim = hidden_dim

    def __call__(self, X: jnp.ndarray, y: jnp.ndarray = None):

        input_dim = X.shape[1]

        # Hidden layer
        hidden_layer = bnn.FFTLinear(input_dim, name="hidden")
        hidden = jax.nn.tanh(hidden_layer(X))

        # Output layer
        output_layer = bnn.Linear(input_dim, 1, name="output")
        mean = numpyro.deterministic("mean", output_layer(hidden).squeeze())

        # Predictive standard deviation
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))

        # Observation likelihood
        numpyro.sample("y", dist.Normal(mean, sigma), obs=y)


hidden_dim = 10
model = SimpleRegressionModel(hidden_dim=hidden_dim)


# Step 1: Train the model using MCMC
def numpyro_model(X, y=None):
    return model(X, y)


# Define the inference algorithm
nuts_kernel = NUTS(numpyro_model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)

# Run MCMC
mcmc.run(jax.random.PRNGKey(0), X=X_train, y=y_train)

# Step 2: Make predictions on X_test
# Use the predictive object for posterior predictions
predictive = Predictive(numpyro_model, mcmc.get_samples())
predictions = predictive(jax.random.PRNGKey(1), X=X_test)

posteriors = predictions["mean"]

# Extract predicted means and other statistics
y_pred_mean = predictions["mean"].mean(axis=0)  # Mean prediction

loss = mean_squared_error(np.array(y_test), np.array(y_pred_mean))
print(f"mean squared error: {loss}")


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


visualize(X_test, y_test, posteriors, 0)
