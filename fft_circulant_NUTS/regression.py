import numpy as np
import jax.numpy as jnp
import jax
from jax import random, vmap
import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
from matplotlib import pyplot as plt


def circulant_matrix_multiply(first_row, X):
    """
    Perform circulant matrix multiplication using FFT.
    """
    first_row_fft = jnp.fft.fft(first_row, axis=-1)
    X_fft = jnp.fft.fft(X, axis=-1)
    result_fft = first_row_fft[None, :] * X_fft
    result = jnp.fft.ifft(result_fft, axis=-1).real
    return result


def bnn_circulant(X, y=None):
    """
    Bayesian Neural Network with Circulant Matrix Layer.
    """
    input_size = X.shape[1]

    # Priors for circulant matrix and bias
    first_row = numpyro.sample("first_row", dist.Normal(0, 1).expand([input_size]))
    bias_circulant = numpyro.sample("bias_circulant", dist.Normal(0, 1))

    # Circulant layer
    hidden = circulant_matrix_multiply(first_row, X) + bias_circulant
    hidden = jax.nn.relu(hidden)

    # Output layer
    weights_out = numpyro.sample(
        "weights_out", dist.Normal(0, 1).expand([input_size, 1])
    )
    bias_out = numpyro.sample("bias_out", dist.Normal(0, 1))

    predictions = jnp.matmul(hidden, weights_out).squeeze() + bias_out

    # Likelihood
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    numpyro.sample("obs", dist.Normal(predictions, sigma), obs=y)


def generate_synthetic_data(n_samples=100, n_features=8, noise_std=0.1):
    """
    Generate synthetic regression data.
    """
    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    true_bias = np.random.randn(1)
    y = X @ true_weights + true_bias + noise_std * np.random.randn(n_samples)
    return jnp.array(X), jnp.array(y)


def run_inference(model, rng_key, X, y, num_samples=1000, num_warmup=500):
    """
    Run MCMC using NUTS.
    """
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, X=X, y=y)
    mcmc.print_summary()
    return mcmc.get_samples()


def predict(model, rng_key, samples, X):
    """
    Generate predictions using posterior samples.
    """
    model = numpyro.handlers.substitute(numpyro.handlers.seed(model, rng_key), samples)
    trace = numpyro.handlers.trace(model).get_trace(X=X, y=None)
    return trace["obs"]["value"]


def compute_mse(y_true, y_pred):
    """
    Compute Mean Squared Error.
    """
    return jnp.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
    # Generate synthetic data
    n_samples, n_features = 100, 8
    X, y = generate_synthetic_data(n_samples, n_features)

    # Run Bayesian inference
    rng_key = random.PRNGKey(0)
    samples = run_inference(
        bnn_circulant, rng_key, X, y, num_samples=1000, num_warmup=500
    )

    # Generate predictions
    rng_keys = random.split(rng_key, samples["bias_out"].shape[0])
    predictions = vmap(lambda sample, key: predict(bnn_circulant, key, sample, X))(
        {k: v for k, v in samples.items()}, rng_keys
    )

    # Compute mean predictions and confidence intervals
    mean_prediction = jnp.mean(predictions, axis=0)
    lower, upper = np.percentile(predictions, [5, 95], axis=0)

    # Compute Mean Squared Error
    mse = compute_mse(y, mean_prediction)
    print(f"Mean Squared Error: {mse:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y, label="True Data", color="blue", alpha=0.5)
    plt.plot(X[:, 0], mean_prediction, label="Mean Prediction", color="red")
    plt.fill_between(X[:, 0], lower, upper, color="pink", alpha=0.3, label="90% CI")
    plt.xlabel("X[0]")
    plt.ylabel("y")
    plt.title(f"Bayesian Neural Network with Circulant Layer\nMSE: {mse:.4f}")
    plt.legend()
    plt.show()
