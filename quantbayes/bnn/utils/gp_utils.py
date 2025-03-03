import matplotlib.pyplot as plt
import jax
import numpy as np
import jax.numpy as jnp
from numpyro.infer import Predictive
from jax.nn import sigmoid

__all__ = [
    "visualize_gp_kernel",
    "sample_gp_prior",
    "predict_gp",
    "predict_gp_binary",
    "visualize_predictions",
    "visualize_predictions_binary",
]


def visualize_gp_kernel(gp_layer, X):
    """
    Computes and visualizes the kernel matrix from the GP layer.

    Parameters:
      gp_layer: GaussianProcessLayer instance.
      X: jnp.ndarray of shape (num_points, input_dim)

    Returns:
      fig: Matplotlib figure.

    Example Usage:
    preds = predict_gp(model, X_train, y_train, X_test)
    mean_pred, var_pred = preds
    fig = visualize_predictions(X_test, mean_pred, var_pred)
    visualize_gp_kernel(model.gp_layer, X_test)
    sample_gp_prior(model.gp_layer, X_train, num_samples=5)
    """
    # Compute the kernel matrix
    kernel_matrix = gp_layer(X)
    kernel_matrix_np = jax.device_get(kernel_matrix)  # Convert to NumPy array

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(kernel_matrix_np, cmap="viridis")
    ax.set_title("GP Kernel (Covariance Matrix)")
    ax.set_xlabel("Data Point Index")
    ax.set_ylabel("Data Point Index")
    fig.colorbar(cax)
    plt.tight_layout()
    plt.show()
    return fig


def sample_gp_prior(gp_layer, X, num_samples=5):
    """
    Draw samples from the GP prior and visualize them.

    Parameters:
      gp_layer: GaussianProcessLayer instance.
      X: jnp.ndarray of shape (num_points, input_dim)
      num_samples: int, number of GP samples to draw.

    Returns:
      fig: Matplotlib figure.
    """
    import numpy as np

    kernel_matrix = gp_layer(X)
    kernel_np = jax.device_get(kernel_matrix)
    # Ensure the kernel is symmetric positive-definite:
    L = np.linalg.cholesky(kernel_np + 1e-6 * np.eye(kernel_np.shape[0]))

    samples = []
    for i in range(num_samples):
        # Draw a sample from standard normal and scale it with the Cholesky factor
        sample = L @ np.random.randn(kernel_np.shape[0])
        samples.append(sample)

    samples = np.array(samples)  # Shape: (num_samples, num_points)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, sample in enumerate(samples):
        ax.plot(sample, label=f"Sample {i+1}")
    ax.set_title("Samples from the GP Prior")
    ax.set_xlabel("Data Point Index")
    ax.set_ylabel("Function Value")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig


def predict_gp(model, X_train, y_train, X_test):
    try:
        # Compute the training covariance matrix (with noise)
        K_train = jax.device_get(model.gp_layer(X_train))
    except:
        raise ValueError("Model must have self.gp_layer")

    # Compute the cross-covariance between test and train data
    K_cross = jax.device_get(model.gp_layer(X_test, X_train))

    # Compute the test covariance (for predictive variance)
    K_test = jax.device_get(model.gp_layer(X_test))

    # Retrieve the noise parameter (as a concrete number) from the gp_layer attribute
    noise = jax.device_get(model.gp_layer.noise)

    # Add noise to training covariance matrix
    # (Note: Depending on your intended model, you might add noise once already in the kernel;
    # here we're following your original code which squares the noise value.)
    K_train_noise = K_train + (noise**2) * np.eye(K_train.shape[0])
    L = np.linalg.cholesky(K_train_noise + 1e-6 * np.eye(K_train.shape[0]))

    # Solve for alpha: K_train_noise⁻¹ y_train = L⁻T (L⁻¹ y_train)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, np.array(y_train)))

    # Predictive mean: K_cross @ alpha
    mean_pred = K_cross.dot(alpha)

    # Predictive variance: diag(K_test) - sum(v**2, axis=0) where v = L⁻¹ K_cross^T
    v = np.linalg.solve(L, K_cross.T)
    var_pred = np.diag(K_test) - np.sum(v**2, axis=0)

    return mean_pred, var_pred


def predict_gp_binary(model, X_train, y_train, X_test):
    # Compute the training covariance matrix (with noise) using the GP layer from the fitted model
    K_train = jax.device_get(model.gp_layer(X_train))
    K_cross = jax.device_get(model.gp_layer(X_test, X_train))
    K_test = jax.device_get(model.gp_layer(X_test))
    noise = jax.device_get(model.gp_layer.noise)
    
    # Add noise to training covariance matrix and compute Cholesky decomposition
    K_train_noise = K_train + (noise**2) * np.eye(K_train.shape[0])
    L = np.linalg.cholesky(K_train_noise + 1e-6 * np.eye(K_train.shape[0]))
    
    # Solve for latent function at training points (posterior mean approximation, for instance)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, np.array(y_train)))
    f_mean = K_cross.dot(alpha)
    
    # Transform the latent mean predictions to probabilities using the sigmoid function
    p_mean = jax.nn.sigmoid(f_mean)
    
    # In practice, you might also want to compute uncertainty in p_mean via sampling
    return p_mean



def visualize_predictions(X_test, mean_pred, var_pred):
    # Convert X_test to a NumPy array
    X_arr = np.array(X_test)

    # If multi-dimensional (more than 1 feature), reduce to 1D with PCA
    if X_arr.ndim > 1 and X_arr.shape[1] > 1:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=1)
        X_reduced = pca.fit_transform(X_arr).flatten()
    else:
        # If univariate, just flatten it.
        X_reduced = X_arr.flatten()

    # Sort for a nice plot
    order = np.argsort(X_reduced)
    X_plot = X_reduced[order]
    mean_pred = np.array(mean_pred)[order]
    std_pred = np.sqrt(np.array(var_pred))[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X_plot, mean_pred, "b-", label="Predictive Mean")
    ax.fill_between(
        X_plot,
        mean_pred - 2 * std_pred,
        mean_pred + 2 * std_pred,
        color="blue",
        alpha=0.3,
        label="Uncertainty (±2 std)",
    )
    ax.set_title("GP Predictive Posterior")
    ax.set_xlabel(
        "Input" if X_arr.ndim == 1 or X_arr.shape[1] == 1 else "PCA Component 1"
    )
    ax.set_ylabel("Output")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig


def visualize_predictions_binary(X_test, pred_probs, threshold=0.5):
    """
    Visualizes the predicted probabilities for binary classification from a Gaussian Process model.

    Parameters:
      X_test: jnp.ndarray or np.ndarray of shape (num_points, input_dim)
        Test inputs.
      pred_probs: array-like of shape (num_points,)
        Predicted probabilities for class 1.
      threshold: float (default: 0.5)
        Decision threshold for classification.

    Returns:
      fig: Matplotlib figure.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Convert X_test to a NumPy array
    X_arr = np.array(X_test)

    # If inputs are multidimensional (>1 feature), reduce to 1D with PCA for visualization.
    if X_arr.ndim > 1 and X_arr.shape[1] > 1:
        pca = PCA(n_components=1)
        X_reduced = pca.fit_transform(X_arr).flatten()
    else:
        X_reduced = X_arr.flatten()

    # Sort the inputs for a cleaner plot.
    order = np.argsort(X_reduced)
    X_sorted = X_reduced[order]
    probs_sorted = np.array(pred_probs)[order]

    # Plot the predicted probabilities.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X_sorted, probs_sorted, "b-", label="Probability of Class 1")
    ax.scatter(X_sorted, probs_sorted, color="blue", s=20)
    ax.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")
    ax.set_title("GP Binary Classification Predictions")
    xlabel = "Input" if X_arr.ndim == 1 or X_arr.shape[1] == 1 else "PCA Component 1"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability of Class 1")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return fig
