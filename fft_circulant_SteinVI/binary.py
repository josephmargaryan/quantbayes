import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpyro
from numpyro import sample
from numpyro.contrib.einstein import SteinVI, RBFKernel, MixtureGuidePredictive
from numpyro.distributions import Normal, Bernoulli, Gamma
from numpyro.optim import Adam
from numpyro.infer.autoguide import AutoNormal
from sklearn.metrics import accuracy_score


# Step 1: Generate Synthetic Data for Binary Classification
def generate_binary_classification_data(n_samples=500, noise=0.2):
    np.random.seed(42)
    x1 = np.random.uniform(-3, 3, n_samples)
    x2 = np.random.uniform(-3, 3, n_samples)
    y = (x1**2 + x2**2 > 4).astype(int)  # Nonlinear decision boundary
    x = np.stack([x1, x2], axis=1)
    x += noise * np.random.randn(*x.shape)  # Add noise
    return jnp.array(x), jnp.array(y)


# Step 2: FFT Circulant Matrix Multiplication
def circulant_matrix_multiply(first_row, X):
    n_features = first_row.shape[0]
    if X.shape[1] < n_features:
        X = jnp.pad(X, ((0, 0), (0, n_features - X.shape[1])))
    elif X.shape[1] > n_features:
        X = X[:, :n_features]
    first_row_fft = jnp.fft.fft(first_row, axis=-1)
    X_fft = jnp.fft.fft(X, axis=-1)
    result_fft = first_row_fft[None, :] * X_fft
    result = jnp.fft.ifft(result_fft, axis=-1).real
    return result


# Step 3: Bayesian Neural Network Model for Binary Classification
def bnn_model_fft(x, y=None, hidden_dim=10):
    input_dim = x.shape[1]
    first_row_proj = sample("first_row_proj", Normal(0, 1).expand([hidden_dim]))
    bias_proj = sample("bias_proj", Normal(0, 1).expand([hidden_dim]))
    x_proj = circulant_matrix_multiply(first_row_proj, x) + bias_proj
    first_row_1 = sample("first_row_1", Normal(0, 1).expand([hidden_dim]))
    bias_circulant_1 = sample("bias_circulant_1", Normal(0, 1).expand([hidden_dim]))
    first_row_2 = sample("first_row_2", Normal(0, 1).expand([hidden_dim]))
    bias_circulant_2 = sample("bias_circulant_2", Normal(0, 1))
    hidden = circulant_matrix_multiply(first_row_1, x_proj) + bias_circulant_1
    hidden = jnp.maximum(hidden, 0)
    logits = (
        circulant_matrix_multiply(first_row_2, hidden).sum(axis=-1) + bias_circulant_2
    )
    sample("y", Bernoulli(logits=logits), obs=y)


# Step 4: Train the Model Using SteinVI
def train_steinvi(x_train, y_train, hidden_dim=10):
    rng_key = random.PRNGKey(0)
    guide = AutoNormal(bnn_model_fft)
    optimizer = Adam(step_size=0.01)
    stein = SteinVI(
        model=bnn_model_fft,
        guide=guide,
        optim=optimizer,
        kernel_fn=RBFKernel(),
        num_stein_particles=10,
        num_elbo_particles=10,
    )
    stein_result = stein.run(
        rng_key, num_steps=2000, x=x_train, y=y_train, hidden_dim=hidden_dim
    )
    return stein, stein_result


# Step 5: Make Predictions
def make_predictions(stein, stein_result, x_test, hidden_dim=10):
    predictive = MixtureGuidePredictive(
        bnn_model_fft,
        stein.guide,
        params=stein.get_params(stein_result.state),
        guide_sites=stein.guide_sites,
        num_samples=100,
    )
    rng_key = random.PRNGKey(1)
    pred_samples = predictive(rng_key, x=x_test, hidden_dim=hidden_dim)["y"]
    probs = pred_samples.mean(axis=0)
    return probs


# Step 6: Plot Decision Boundary
def plot_decision_boundary(stein, stein_result, x_train, y_train, hidden_dim=10):
    x_min, x_max = x_train[:, 0].min() - 0.5, x_train[:, 0].max() + 0.5
    y_min, y_max = x_train[:, 1].min() - 0.5, x_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = jnp.c_[xx.ravel(), yy.ravel()]
    probs = make_predictions(stein, stein_result, grid, hidden_dim=hidden_dim)
    zz = probs.reshape(xx.shape)
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, zz, levels=50, cmap="RdBu", alpha=0.8)
    plt.scatter(
        x_train[:, 0],
        x_train[:, 1],
        c=y_train,
        edgecolor="k",
        cmap="RdBu",
        label="Train",
    )
    plt.colorbar(label="Predicted Probability")
    plt.title("Binary Classification Decision Boundary with SteinVI")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


# Step 7: Compute Accuracy
def compute_accuracy(y_true, probs):
    y_pred = (probs > 0.5).astype(int)
    return accuracy_score(y_true, y_pred)


# Main script
if __name__ == "__main__":
    x_train, y_train = generate_binary_classification_data(n_samples=500)
    x_test, y_test = generate_binary_classification_data(n_samples=200)
    stein, stein_result = train_steinvi(x_train, y_train, hidden_dim=10)
    train_probs = make_predictions(stein, stein_result, x_train, hidden_dim=10)
    test_probs = make_predictions(stein, stein_result, x_test, hidden_dim=10)
    train_accuracy = compute_accuracy(y_train, train_probs)
    test_accuracy = compute_accuracy(y_test, test_probs)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    plot_decision_boundary(stein, stein_result, x_train, y_train, hidden_dim=10)
