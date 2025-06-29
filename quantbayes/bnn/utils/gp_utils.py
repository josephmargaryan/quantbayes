import jax
import matplotlib.pyplot as plt
import numpy as np

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
        # Compute the training covariance matrix (noise already added inside gp_layer)
        K_train = jax.device_get(model.gp_layer(X_train))
    except Exception as e:
        raise ValueError("Model must have self.gp_layer") from e

    # Compute the cross-covariance between test and train data
    K_cross = jax.device_get(model.gp_layer(X_test, X_train))

    # Compute the test covariance (for predictive variance)
    K_test = jax.device_get(model.gp_layer(X_test))

    # Do not add noise here since it is already included in K_train
    L = np.linalg.cholesky(K_train)

    # Solve for alpha: K_train⁻¹ y_train = L⁻T (L⁻¹ y_train)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, np.array(y_train)))

    # Predictive mean: K_cross @ alpha
    mean_pred = K_cross.dot(alpha)

    # Predictive variance: diag(K_test) - sum(v², axis=0) where v = L⁻¹ K_cross^T
    v = np.linalg.solve(L, K_cross.T)
    var_pred = np.diag(K_test) - np.sum(v**2, axis=0)

    return mean_pred, var_pred


def predict_gp_binary(model, X_train, y_train, X_test, num_samples=100):
    """
    Computes the predictive distribution for a GP binary classifier by drawing multiple samples
    from the approximate posterior of the latent function and then applying the sigmoid transformation.
    """
    # Compute covariance matrices using the fitted GP layer (noise is already added)
    K_train = jax.device_get(model.gp_layer(X_train))
    K_cross = jax.device_get(model.gp_layer(X_test, X_train))
    K_test = jax.device_get(model.gp_layer(X_test))

    # Use the training covariance matrix as-is (noise already included)
    L = np.linalg.cholesky(K_train)

    # Compute a "regression-style" mean for f at test points.
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, np.array(y_train)))
    f_mean = K_cross.dot(alpha)

    # Approximate the posterior covariance for f at test points.
    V = K_test - K_cross.dot(np.linalg.solve(K_train, K_cross.T))
    V += 1e-6 * np.eye(V.shape[0])  # Ensure numerical stability
    L_V = np.linalg.cholesky(V)

    # Draw samples from the approximate Gaussian posterior of f at test points.
    all_probs = []
    for i in range(num_samples):
        z = np.random.randn(V.shape[0])
        f_sample = f_mean + L_V.dot(z)
        probs = jax.nn.sigmoid(f_sample)
        all_probs.append(probs)
    all_probs = np.array(all_probs)  # shape: (num_samples, n_test)

    # Compute mean and standard deviation of the predicted probabilities.
    mean_prob = all_probs.mean(axis=0)
    std_prob = all_probs.std(axis=0)
    return mean_prob, std_prob, all_probs


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


def visualize_predictions_binary(X_test, mean_prob, std_prob, threshold=0.5):
    """
    Visualizes the predicted probabilities for binary classification along with uncertainty bands.

    Parameters:
      X_test: array-like, shape (num_points, input_dim)
        Test inputs.
      mean_prob: array-like, shape (num_points,)
        Mean predicted probability for class 1.
      std_prob: array-like, shape (num_points,)
        Standard deviation of predicted probabilities.
      threshold: float (default: 0.5)
        Decision threshold for classification.

    Returns:
      fig: Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA

    # Convert X_test to a NumPy array.
    X_arr = np.array(X_test)

    # Reduce dimensionality for visualization if necessary.
    if X_arr.ndim > 1 and X_arr.shape[1] > 1:
        pca = PCA(n_components=1)
        X_reduced = pca.fit_transform(X_arr).flatten()
    else:
        X_reduced = X_arr.flatten()

    # Sort inputs for a smooth plot.
    order = np.argsort(X_reduced)
    X_sorted = X_reduced[order]
    mean_sorted = np.array(mean_prob)[order]
    std_sorted = np.array(std_prob)[order]

    # Create the plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X_sorted, mean_sorted, "b-", label="Mean Probability")
    # Plot uncertainty bands (e.g., ±2 standard deviations).
    ax.fill_between(
        X_sorted,
        mean_sorted - 2 * std_sorted,
        mean_sorted + 2 * std_sorted,
        color="blue",
        alpha=0.3,
        label="Uncertainty (±2 std)",
    )
    ax.axhline(
        y=threshold, color="red", linestyle="--", label=f"Threshold = {threshold}"
    )
    xlabel = "Input" if X_arr.ndim == 1 or X_arr.shape[1] == 1 else "PCA Component 1"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability of Class 1")
    ax.set_title("GP Binary Classification Predictions with Uncertainty")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig
