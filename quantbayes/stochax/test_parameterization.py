import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Flax / NumPyro
import flax.linen as nn
from numpyro.contrib.module import flax_module
import numpyro
import numpyro.distributions as dist
from numpyro.optim import Adam
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoDiagonalNormal

# Utilities
from quantbayes.fake_data import generate_regression_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Load and Prepare the Data
# -----------------------------
df = generate_regression_data()  # Returns a pandas DataFrame
X_df = df.drop("target", axis=1)
y_df = df["target"]

# We will scale both X and y for demonstration
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(X_df)
y = scaler_y.fit_transform(y_df.to_numpy().reshape(-1, 1)).reshape(-1)

# Convert to JAX arrays
X = jnp.array(X)
y = jnp.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=24, test_size=0.2
)
print(f"Training Data: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Test Data: X_test {X_test.shape}, y_test {y_test.shape}")

# ------------------------------------
# 2. Define a Simple Flax MLP Network
# ------------------------------------
class MLP(nn.Module):
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        # x: (batch_size, input_dim)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        # final output dimension = 1 for regression
        x = nn.Dense(1)(x)
        # return shape = (batch_size,) for convenience
        return x[..., 0]

# ---------------------------------------------
# 3. Build a NumPyro Model that uses flax_module
# ---------------------------------------------
def regression_model(X, y=None):
    """
    Bayesian regression with a neural network for the mean function:
        mean = MLP(X)
    and an exponential prior on the noise scale sigma.
    """
    # Insert Flax MLP into the model as a param
    # We'll give an example shape of (1, n_features) to init MLP inside `flax_module`.
    net = flax_module(
        "nn_module",
        MLP(hidden_dim=64),
        input_shape=(1, X.shape[1])  # shape for a single "dummy" sample
    )

    # Noise scale prior
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))

    # Neural network prediction for each data point
    mu = net(X)  # shape: (batch_size,)

    # Likelihood
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

# ---------------------------------------------------------
# 4. Use AutoDiagonalNormal for a Variational Approximation
# ---------------------------------------------------------
guide = AutoDiagonalNormal(regression_model)

# ---------------------------
# 5. Set Up SVI and Optimizer
# ---------------------------
svi = SVI(
    model=regression_model,
    guide=guide,
    optim=Adam(1e-3),
    loss=Trace_ELBO()
)

rng_key = jax.random.PRNGKey(0)
# Initialize SVI
svi_state = svi.init(rng_key, X_train, y_train)

# ---------------
# 6. SVI Training
# ---------------
num_steps = 100
losses = []

for step in range(num_steps):
    svi_state, loss_val = svi.update(svi_state, X_train, y_train)
    losses.append(loss_val)
    if (step + 1) % 500 == 0:
        print(f"Step {step+1}, ELBO (negative loss) = {-loss_val:.4f}")

# --------------------------
# 7. Posterior Predictions
# --------------------------
# Extract learned variational parameters
params = svi.get_params(svi_state)
print(f"params keys: {params.keys()}")

# Draw posterior samples of all latent variables (including MLP params & sigma).
# shape of 'posterior_samples["sigma"]' => (n_draws,)
# shape of 'posterior_samples["nn_module$params"]' => ...
n_draws = 100
posterior_samples = guide.sample_posterior(
    jax.random.PRNGKey(1),
    params,
    sample_shape=(n_draws,)
)

# Build a Predictive wrapper to get posterior predictive distribution
predictive = Predictive(regression_model, posterior_samples)
print(f"Posterior samples dict keys: {predictive(jax.random.PRNGKey(2), X_test).keys()}")
predictions = predictive(jax.random.PRNGKey(2), X_test)["obs"]
# 'predictions' shape: (n_draws, n_test)

# Posterior predictive mean (per test data point)
mean_preds = np.mean(predictions, axis=0)
# 90% credible interval via percentile
lower_ci, upper_ci = np.percentile(predictions, [5, 95], axis=0)

# Invert scaling for interpretability
mean_preds_unscaled = scaler_y.inverse_transform(mean_preds.reshape(-1, 1)).flatten()
lower_ci_unscaled = scaler_y.inverse_transform(lower_ci.reshape(-1, 1)).flatten()
upper_ci_unscaled = scaler_y.inverse_transform(upper_ci.reshape(-1, 1)).flatten()
y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Evaluate MSE on the posterior predictive mean
mse = np.mean((mean_preds_unscaled - y_test_unscaled)**2)
print(f"\nPosterior Predictive Mean MSE (Test set): {mse:.4f}")

# -----------------------------------
# 8. Visualization of Results
# -----------------------------------
# (A) Plot SVI Loss (negative ELBO)
plt.figure(figsize=(7, 4))
plt.plot(-np.array(losses))  # negative of the loss is the ELBO
plt.title("ELBO Progress during SVI")
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.grid(True)
plt.show()

# (B) Compare true vs. predicted with credible intervals
plt.figure(figsize=(7, 6))
plt.scatter(y_test_unscaled, mean_preds_unscaled, alpha=0.6, label="Posterior Mean")
# Add error bars from the 5%-95% credible interval
plt.errorbar(
    y_test_unscaled,
    mean_preds_unscaled,
    yerr=[mean_preds_unscaled - lower_ci_unscaled,
          upper_ci_unscaled - mean_preds_unscaled],
    fmt="none",
    ecolor="gray",
    alpha=0.5
)
# Ideal diagonal
min_val = min(y_test_unscaled.min(), mean_preds_unscaled.min())
max_val = max(y_test_unscaled.max(), mean_preds_unscaled.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
plt.xlabel("True y")
plt.ylabel("Predicted y (Posterior Mean)")
plt.legend()
plt.grid(True)
plt.title("Posterior Predictive Mean and 90% CI")
plt.show()
