import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
import jax


# --- Kernel Function --- #
def kernel(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
    deltaXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k


# --- GP Model --- #
def model(X, Y):
    var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 1.0))
    noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 1.0))
    length = numpyro.sample("kernel_length", dist.LogNormal(0.0, 1.0))

    k = kernel(X, X, var, length, noise)
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=k),
        obs=Y,
    )


# --- Run Inference --- #
def run_inference(rng_key, X, Y):
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=5000, num_samples=10000, num_chains=1)
    mcmc.run(rng_key, X, Y)
    mcmc.print_summary()
    return mcmc


# --- Prediction --- #
def predict(rng_key, X, Y, X_test, var, length, noise, use_cholesky=True):
    k_pp = kernel(X_test, X_test, var, length, noise, include_noise=True)
    k_pX = kernel(X_test, X, var, length, noise, include_noise=False)
    k_XX = kernel(X, X, var, length, noise, include_noise=True)

    if use_cholesky:
        K_xx_cho = jax.scipy.linalg.cho_factor(k_XX)
        K = k_pp - jnp.matmul(k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, k_pX.T))
        mean = jnp.matmul(k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, Y))
    else:
        K_xx_inv = jnp.linalg.inv(k_XX)
        K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, k_pX.T))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y))

    std_dev = jnp.sqrt(jnp.clip(jnp.diag(K), 0.0))
    return mean, std_dev


# --- Predict Future Trajectories --- #
def predict_future_trajectories(rng_key, X, Y, X_future, posterior_samples):
    """
    Predict future trajectories using the Gaussian Process model.

    This function uses the posterior samples from the MCMC to compute the
    predictive mean and uncertainty (standard deviation) for future points.

    :param rng_key: JAX random key for reproducibility.
    :param X: Training input data, shape (n_train,).
    :param Y: Training target data, shape (n_train,).
    :param X_future: Future input data, shape (n_future,).
    :param posterior_samples: Posterior samples of kernel hyperparameters,
                              containing "kernel_var", "kernel_length", and "kernel_noise".

    :return:
        - mean_future (jnp.ndarray): Mean predictions for future trajectories, shape (n_future,).
        - std_future (jnp.ndarray): Standard deviation (uncertainty) of predictions, shape (n_future,).
    """
    mean_trajectories = []
    std_trajectories = []

    # Iterate over posterior samples to compute predictions
    for i in range(len(posterior_samples["kernel_var"])):
        mean, std = predict(
            rng_key,
            X,
            Y,
            X_future,
            posterior_samples["kernel_var"][i],
            posterior_samples["kernel_length"][i],
            posterior_samples["kernel_noise"][i],
        )
        mean_trajectories.append(mean)
        std_trajectories.append(std)

    # Compute mean and std across all trajectories
    mean_future = jnp.mean(jnp.array(mean_trajectories), axis=0)
    std_future = jnp.std(jnp.array(std_trajectories), axis=0)

    return mean_future, std_future


# --- Visualization --- #
def visualize_predictions(X_train, Y_train, X_test, Y_test, mean, std):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, Y_train, label="Training Data", color="blue")
    plt.scatter(X_test, Y_test, label="Test Data", color="green")
    plt.plot(X_test, mean, label="Predictive Mean", color="red")
    plt.fill_between(
        X_test.flatten(),
        mean - 2 * std,
        mean + 2 * std,
        color="red",
        alpha=0.3,
        label="Predictive Uncertainty (95%)",
    )
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("GP Predictions on Test Data")
    plt.grid(True)
    plt.show()


def visualize_future_predictions(X_train, Y_train, X_future, mean_future, std_future):
    """
    Visualize training data and future trajectories predicted by the Gaussian Process.

    :param X_train: Training input data, shape (n_train,).
    :param Y_train: Training target data, shape (n_train,).
    :param X_future: Future input data, shape (n_future,).
    :param mean_future: Predictive mean for future data, shape (n_future,).
    :param std_future: Predictive standard deviation for future data, shape (n_future,).
    """
    plt.figure(figsize=(10, 6))

    # Plot training data
    plt.scatter(X_train, Y_train, color="blue", label="Training Data", zorder=2)

    # Plot predictive mean
    plt.plot(
        X_future, mean_future, color="red", label="Predictive Mean (Future)", zorder=3
    )

    # Plot predictive uncertainty
    plt.fill_between(
        X_future,
        mean_future - 2 * std_future,
        mean_future + 2 * std_future,
        color="red",
        alpha=0.2,
        label="Predictive Uncertainty (95%)",
        zorder=1,
    )

    # Add titles and labels
    plt.title("GP Future Predictions", fontsize=16)
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


# --- Main --- #
def main():
    rng_key = random.PRNGKey(0)

    # Generate synthetic data
    X_train = jnp.linspace(0, 5, 20)
    Y_train = jnp.sin(X_train) + 0.1 * random.normal(rng_key, shape=X_train.shape)

    X_test = jnp.linspace(5, 6, 10)
    Y_test = jnp.sin(X_test) + 0.1 * random.normal(rng_key, shape=X_test.shape)

    # Run inference
    rng_key, subkey = random.split(rng_key)
    mcmc = run_inference(subkey, X_train, Y_train)

    # Predictions on test data
    posterior_samples = mcmc.get_samples()
    mean, std = predict(
        rng_key,
        X_train,
        Y_train,
        X_test,
        posterior_samples["kernel_var"].mean(),
        posterior_samples["kernel_length"].mean(),
        posterior_samples["kernel_noise"].mean(),
    )
    visualize_predictions(X_train, Y_train, X_test, Y_test, mean, std)

    # Future predictions
    X_future = jnp.linspace(6, 8, 20)
    future_mean, future_std = predict_future_trajectories(
        rng_key, X_train, Y_train, X_future, posterior_samples
    )
    visualize_future_predictions(X_train, Y_train, X_future, future_mean, future_std)


if __name__ == "__main__":
    main()
