import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.contrib.control_flow import scan
import matplotlib.pyplot as plt


# --- Utility Functions ---
def spectral_density(w, alpha, length):
    """Squared exponential spectral density."""
    c = alpha * jnp.sqrt(2 * jnp.pi) * length
    e = jnp.exp(-0.5 * (length**2) * (w**2))
    return c * e


def diag_spectral_density(alpha, length, L, M):
    """Diagonal spectral density for the kernel."""
    sqrt_eigenvalues = jnp.arange(1, 1 + M) * jnp.pi / (2 * L)
    return spectral_density(sqrt_eigenvalues, alpha, length)


def eigenfunctions(x, L, M):
    """Laplacian eigenfunctions for the squared exponential kernel."""
    m1 = (jnp.pi / (2 * L)) * jnp.tile(L + x[:, None], M)
    m2 = jnp.diag(jnp.linspace(1, M, num=M))
    num = jnp.sin(m1 @ m2)
    den = jnp.sqrt(L)
    return num / den


def approx_se_ncp(x, alpha, length, L, M):
    """Hilbert space approximation for the squared exponential kernel."""
    phi = eigenfunctions(x, L, M)
    spd = jnp.sqrt(diag_spectral_density(alpha, length, L, M))
    with numpyro.plate("basis", M):
        beta = numpyro.sample("beta", dist.Normal(0, 1))
    f = jnp.dot(phi, spd * beta)
    return f


# --- Model Definition ---
def model(x, L, M, y=None):
    """
    Generalized probabilistic model with Hilbert space approximation.

    :param x: Input features, shape (n,).
    :param L: Domain length for eigenfunctions.
    :param M: Number of eigenfunctions.
    :param y: Observed outputs, shape (n,). Default is None for prediction.
    """
    alpha = numpyro.sample("alpha", dist.HalfNormal(1.0))
    length = numpyro.sample("length", dist.InverseGamma(2.0, 1.0))
    noise = numpyro.sample("noise", dist.HalfNormal(1.0))
    f = approx_se_ncp(x, alpha, length, L, M)
    numpyro.sample("likelihood", dist.Normal(f, noise), obs=y)


# --- Inference Function ---
def run_inference(rng_key, x, y, L, M, num_warmup=1000, num_samples=2000, num_chains=1):
    """
    Perform inference using MCMC.

    :param rng_key: JAX random key.
    :param x: Input features, shape (n,).
    :param y: Observed outputs, shape (n,).
    :param L: Domain length for eigenfunctions.
    :param M: Number of eigenfunctions.
    :param num_warmup: Number of warmup steps for MCMC.
    :param num_samples: Number of MCMC samples.
    :param num_chains: Number of MCMC chains.
    :return: MCMC object.
    """
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains
    )
    mcmc.run(rng_key, x=x, L=L, M=M, y=y)
    return mcmc


# --- Prediction Function ---
def predict(rng_key, mcmc, x_new, L, M):
    """
    Predict using the posterior samples from MCMC.

    :param rng_key: JAX random key.
    :param mcmc: MCMC object containing posterior samples.
    :param x_new: New input features for prediction, shape (n_new,).
    :param L: Domain length for eigenfunctions.
    :param M: Number of eigenfunctions.
    :return: Predictive samples for new inputs.
    """
    predictive = Predictive(model, mcmc.get_samples())
    predictions = predictive(rng_key, x=x_new, L=L, M=M)
    return predictions["likelihood"]


def predict_future_trajectories(
    rng_key, mcmc, x_train, n_future_steps, L, M, n_samples=100
):
    """
    Generate future trajectories using the trained model.

    :param rng_key: JAX random key.
    :param mcmc: Trained MCMC object containing posterior samples.
    :param x_train: Training input data, shape (n_train,).
    :param n_future_steps: Number of steps into the future to predict.
    :param L: Domain length for eigenfunctions.
    :param M: Number of eigenfunctions.
    :param n_samples: Number of samples for uncertainty visualization.
    :return: Future trajectories (mean, std, and sampled trajectories).
    """
    # Define future input range
    x_future = jnp.linspace(
        x_train[-1] + 1, x_train[-1] + n_future_steps, n_future_steps
    )

    # Combine training and future input data
    x_combined = jnp.concatenate([x_train, x_future])

    # Subsample posterior samples to match n_samples
    posterior_samples = mcmc.get_samples()
    subsampled_posterior = {
        key: value[:n_samples] for key, value in posterior_samples.items()
    }

    # Predict for combined data
    predictive = Predictive(model, subsampled_posterior)
    predictions = predictive(rng_key, x=x_combined, L=L, M=M)["likelihood"]

    # Separate future predictions
    future_predictions = predictions[:, len(x_train) :]
    future_mean = future_predictions.mean(axis=0)
    future_std = future_predictions.std(axis=0)

    return x_future, future_mean, future_std, future_predictions


def visualize_model_performance(
    x_train, y_train, x_val, y_val, predictions_mean, predictions_std
):
    """
    Visualize the model's performance on training and validation data.

    :param x_train: Training input data, shape (n_train,).
    :param y_train: Training observed data, shape (n_train,).
    :param x_val: Validation input data, shape (n_val,).
    :param y_val: Validation observed data, shape (n_val,).
    :param predictions_mean: Predictive mean for validation data, shape (n_val,).
    :param predictions_std: Predictive std for validation data, shape (n_val,).
    """
    plt.figure(figsize=(12, 6))

    # Plot training data
    plt.scatter(x_train, y_train, color="blue", alpha=0.6, label="Training Data")

    # Plot validation data
    plt.scatter(x_val, y_val, color="green", alpha=0.6, label="Validation Data")

    # Plot predictive mean
    plt.plot(x_val, predictions_mean, color="red", label="Predictive Mean")

    # Plot predictive uncertainty (shaded region)
    plt.fill_between(
        x_val,
        predictions_mean - 2 * predictions_std,
        predictions_mean + 2 * predictions_std,
        color="red",
        alpha=0.2,
        label="Predictive ± 2 Std Dev (Approx. 95% CI)",
    )

    plt.title("Model Performance on Training and Validation Data", fontsize=16)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True)
    plt.show()


def visualize_future_trajectories(
    x_train, y_train, x_future, future_mean, future_std, future_samples
):
    """
    Visualize the model's future trajectories.

    :param x_train: Training input data, shape (n_train,).
    :param y_train: Training observed data, shape (n_train,).
    :param x_future: Future input data, shape (n_future,).
    :param future_mean: Predictive mean for future data, shape (n_future,).
    :param future_std: Predictive std for future data, shape (n_future,).
    :param future_samples: Sampled future trajectories, shape (n_samples, n_future).
    """
    plt.figure(figsize=(12, 6))

    # Plot training data
    plt.plot(x_train, y_train, color="blue", alpha=0.6, label="Training Data")

    # Plot future mean trajectory
    plt.plot(x_future, future_mean, color="red", label="Predictive Mean")

    # Plot individual sample trajectories
    for i in range(min(len(future_samples), 10)):  # Plot up to 10 trajectories
        plt.plot(x_future, future_samples[i], alpha=0.3, color="orange", linestyle="--")

    # Plot predictive uncertainty (shaded region)
    plt.fill_between(
        x_future,
        future_mean - 2 * future_std,
        future_mean + 2 * future_std,
        color="red",
        alpha=0.2,
        label="Predictive ± 2 Std Dev (Approx. 95% CI)",
    )

    plt.title("Model Future Trajectories", fontsize=16)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True)
    plt.show()


# --- Demonstration ---
def demonstrate_future_trajectory():
    # Generate synthetic data
    rng_key = random.PRNGKey(0)
    x = jnp.linspace(0, 10, 100)
    y_true = jnp.sin(x) + 0.1 * random.normal(rng_key, shape=x.shape)
    rng_key, subkey = random.split(rng_key)
    y_obs = y_true + 0.2 * random.normal(subkey, shape=x.shape)

    # Split data into training and validation
    x_train, y_train = x[:80], y_obs[:80]
    x_val, y_val = x[80:], y_obs[80:]

    # Define model parameters
    L = x.max() - x.min()
    M = 30

    # Run inference
    rng_key, subkey = random.split(rng_key)
    mcmc = run_inference(subkey, x_train, y_train, L, M)

    # Predict on validation data
    rng_key, subkey = random.split(rng_key)
    predictions = Predictive(model, mcmc.get_samples())(subkey, x=x_val, L=L, M=M)
    predictions_mean = predictions["likelihood"].mean(axis=0)
    predictions_std = predictions["likelihood"].std(axis=0)

    # Visualize model performance
    visualize_model_performance(
        x_train, y_train, x_val, y_val, predictions_mean, predictions_std
    )

    # Predict future trajectories
    rng_key, subkey = random.split(rng_key)
    x_future, future_mean, future_std, future_samples = predict_future_trajectories(
        subkey, mcmc, x_train, 20, L, M
    )

    # Visualize future trajectories
    visualize_future_trajectories(
        x_train, y_train, x_future, future_mean, future_std, future_samples
    )


if __name__ == "__main__":
    demonstrate_future_trajectory()
