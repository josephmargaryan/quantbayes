import numpy as np
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
import arviz as az


# Generate synthetic data
def generate_synthetic_data(n_samples=100):
    X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
    y = 2.0 * X.squeeze() + 1.0 + np.random.normal(0, 0.5, size=n_samples)
    return X, y


# Dense Bayesian Neural Network
def dense_bnn(X, y=None, hidden_dim=10):
    input_dim = X.shape[1]

    w_hidden = numpyro.sample(
        "w_hidden", dist.Normal(0, 1).expand([input_dim, hidden_dim])
    )
    b_hidden = numpyro.sample("b_hidden", dist.Normal(0, 1).expand([hidden_dim]))

    w_out = numpyro.sample("w_out", dist.Normal(0, 1).expand([hidden_dim, 1]))
    b_out = numpyro.sample("b_out", dist.Normal(0, 1).expand([1]))

    hidden = jax.nn.relu(jnp.dot(X, w_hidden) + b_hidden)
    mean = numpyro.deterministic("mean", jnp.dot(hidden, w_out).squeeze() + b_out)

    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    numpyro.sample("y", dist.Normal(mean, sigma), obs=y)


# Circulant Bayesian Neural Network
def circulant_matrix_multiply(first_row, X):
    first_row_fft = jnp.fft.fft(first_row, axis=-1)
    X_fft = jnp.fft.fft(X, axis=-1)
    result_fft = first_row_fft[None, :] * X_fft
    result = jnp.fft.ifft(result_fft, axis=-1).real
    return result


def circulant_bnn(X, y=None):
    input_dim = X.shape[1]

    first_row = numpyro.sample("first_row", dist.Normal(0, 1).expand([input_dim]))
    bias_circulant = numpyro.sample(
        "bias_circulant", dist.Normal(0, 1).expand([input_dim])
    )

    hidden = circulant_matrix_multiply(first_row, X) + bias_circulant
    hidden = jax.nn.relu(hidden)

    weights_out = numpyro.sample(
        "w_out", dist.Normal(0, 1).expand([hidden.shape[1], 1])
    )
    bias_out = numpyro.sample("b_out", dist.Normal(0, 1))

    predictions = jnp.matmul(hidden, weights_out).squeeze() + bias_out

    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    numpyro.sample("y", dist.Normal(predictions, sigma), obs=y)


# Run MCMC and plot posterior distributions
def run_and_plot(X, y, model, model_name, num_samples=1000, num_warmup=500):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=True
    )
    rng_key = PRNGKey(0)
    mcmc.run(rng_key, X, y)

    posterior_samples = mcmc.get_samples()
    az_data = az.from_numpyro(mcmc)

    # Plot posterior distributions
    az.plot_trace(az_data, var_names=["w_out", "sigma"])
    plt.suptitle(f"{model_name} Posterior Distributions", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()


# Main script
if __name__ == "__main__":
    # Generate data
    X, y = generate_synthetic_data()

    # Run Dense BNN
    print("Running Dense BNN...")
    run_and_plot(X, y, dense_bnn, "Dense BNN")

    # Run Circulant BNN
    print("Running Circulant BNN...")
    run_and_plot(X, y, circulant_bnn, "Circulant BNN")
