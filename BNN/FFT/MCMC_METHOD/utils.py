from numpyro.infer import NUTS, MCMC
from numpyro.infer import Predictive
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import random
from matplotlib.colors import ListedColormap

def run_inference(model, rng_key, X, y, num_samples=1000, num_warmup=500):
    """
    Run MCMC using NUTS.
    """
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, X=X, y=y)
    mcmc.print_summary()
    return mcmc


def predict_regression(mcmc, X_test, model):
    posterior_samples = mcmc.get_samples()
    predictive = Predictive(model, posterior_samples)
    preds = predictive(rng_key=jax.random.PRNGKey(1), X=X_test)
    return preds["obs"]

def predict_binary(mcmc, X_test, model):
    """
    Predict probabilities for a binary classification model.

    Parameters:
    - mcmc: The MCMC object after sampling.
    - X_test: The test set input features (JAX array).
    - model: The probabilistic model used for inference.

    Returns:
    - probabilities: A 2D array of shape (num_samples, len(X_test)) containing predicted probabilities.
    """
    posterior_samples = mcmc.get_samples()
    predictive = Predictive(model, posterior_samples)
    preds = predictive(rng_key=jax.random.PRNGKey(1), X=X_test)
    predictions = preds["obs"]  
    return predictions

def predict_multiclass(mcmc, X_test, model, n_classes=None):
    """
    Predict probabilities for a multiclass classification model.

    Parameters:
    - mcmc: The MCMC object after sampling.
    - X_test: The test set input features (JAX array).
    - model: The probabilistic model used for inference.
    - n_classes: The expected number of classes (optional, for validation).

    Returns:
    - probabilities: A 3D array of shape (num_samples, len(X_test), n_classes) containing predicted probabilities.
    """
    posterior_samples = mcmc.get_samples()
    predictive = Predictive(model, posterior_samples)
    preds = predictive(rng_key=jax.random.PRNGKey(1), X=X_test)
    predictions = preds["logits"]  # Extract logits
    
    if n_classes is not None and predictions.shape[-1] != n_classes:
        raise ValueError(
            f"Mismatch in number of classes: logits have {predictions.shape[-1]} classes, expected {n_classes}."
        )
    
    return predictions


def visualize_regression(
    X_test, y_test, mean_preds, lower_bound, upper_bound, feature_index=None
):
    """
    Visualize predictions with uncertainty bounds and true targets.

    Args:
        X_test (jnp.ndarray): Test features.
        y_test (jnp.ndarray): Test target values.
        mean_preds (jnp.ndarray): Mean predictions from the model.
        lower_bound (jnp.ndarray): Lower uncertainty bound.
        upper_bound (jnp.ndarray): Upper uncertainty bound.
        feature_index (int): Index of the feature to plot against y_test. If None or invalid, uses default.

    Returns:
        None. Displays the plot.
    """
    X_test = np.array(X_test)  # Convert to NumPy for easier handling
    y_test = np.array(y_test)
    mean_preds = np.array(mean_preds)
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)

    # Handle feature selection logic
    if (
        X_test.shape[1] == 1
        or feature_index is None
        or not (0 <= feature_index < X_test.shape[1])
    ):
        feature_index = 0  # Default to the only feature if invalid index is provided

    feature = X_test[:, feature_index]

    # Sort the data for proper visualization
    sorted_indices = np.argsort(feature)
    feature = feature[sorted_indices]
    y_test = y_test[sorted_indices]
    mean_preds = mean_preds[sorted_indices]
    lower_bound = lower_bound[sorted_indices]
    upper_bound = upper_bound[sorted_indices]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(feature, y_test, color="blue", alpha=0.6, label="True Targets")
    plt.plot(
        feature,
        mean_preds,
        color="red",
        label="Mean Predictions",
        linestyle="-",
    )
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


def visualize_binary(
    X, y, mcmc, predict, binary_model, feature_indices=(0, 1), grid_resolution=100
):
    """
    Visualize binary classification decision boundary with uncertainty.

    Args:
        X (jnp.ndarray): Input features.
        y (jnp.ndarray): Target labels (binary).
        mcmc: numpyro.infer.MCMC
            Trained MCMC object containing posterior samples.
        predict: Callable
            Prediction function for binary classification.
        binary_model: Callable
            The binary classification model to use for predictions.
        feature_indices (tuple): Indices of the two features to visualize (x and y axes).
        grid_resolution (int): Number of points for each grid axis (higher means finer grid).

    Returns:
        None. Displays the plot.
    """
    X = np.array(X)
    y = np.array(y)

    # Extract the features to plot
    feature1_idx, feature2_idx = feature_indices
    feature1, feature2 = X[:, feature1_idx], X[:, feature2_idx]

    # Create a grid of points for the selected features
    x_min, x_max = feature1.min() - 1, feature1.max() + 1
    y_min, y_max = feature2.min() - 1, feature2.max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Create input for prediction by setting other features to their mean
    X_for_grid = np.zeros((grid.shape[0], X.shape[1]))
    X_for_grid[:, feature1_idx] = grid[:, 0]
    X_for_grid[:, feature2_idx] = grid[:, 1]
    for i in range(X.shape[1]):
        if i not in feature_indices:
            X_for_grid[:, i] = X[:, i].mean()

    # Predict probabilities for the grid
    grid_preds = predict(mcmc, jnp.array(X_for_grid), binary_model)
    grid_mean = grid_preds.mean(axis=0).reshape(xx.shape)
    grid_uncertainty = grid_preds.var(axis=0).reshape(xx.shape)

    # Plot decision boundary with uncertainty
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, grid_mean, levels=20, cmap="RdBu", alpha=0.8, vmin=0, vmax=1)
    plt.colorbar(label="Predicted Probability (Mean)")

    # Overlay uncertainty as brightness effect
    plt.imshow(
        grid_uncertainty,
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap="binary",
        alpha=0.3,
        aspect="auto",
    )

    # Plot decision boundary
    plt.contour(xx, yy, grid_mean, levels=[0.5], colors="black", linestyles="--")

    # Scatter plot for true labels
    plt.scatter(
        feature1[y == 0],
        feature2[y == 0],
        color="blue",
        label="Class 0",
        edgecolor="k",
        alpha=0.6,
    )
    plt.scatter(
        feature1[y == 1],
        feature2[y == 1],
        color="red",
        label="Class 1",
        edgecolor="k",
        alpha=0.6,
    )

    plt.xlabel(f"Feature {feature1_idx + 1}")
    plt.ylabel(f"Feature {feature2_idx + 1}")
    plt.title("Binary Decision Boundary with Uncertainty")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()


def visualize_multiclass(
    X, y, mcmc, predict, multiclass_model, feature_indices=(0, 1), grid_resolution=100
):
    """
    Visualize multiclass classification decision boundary with uncertainty.

    Args:
        X (jnp.ndarray): Input features.
        y (jnp.ndarray): Target labels (integer class labels).
        mcmc: numpyro.infer.MCMC
            Trained MCMC object containing posterior samples.
        predict: Callable
            Prediction function for multiclass classification.
        multiclass_model: Callable
            The multiclass classification model to use for predictions.
        feature_indices (tuple): Indices of the two features to visualize (x and y axes).
        grid_resolution (int): Number of points for each grid axis (higher means finer grid).

    Returns:
        None. Displays the plot.
    """
    X = np.array(X)
    y = np.array(y)

    # Automatically infer n_classes from the target labels
    n_classes = len(np.unique(y))

    # Extract the features to plot
    feature1_idx, feature2_idx = feature_indices
    feature1, feature2 = X[:, feature1_idx], X[:, feature2_idx]

    # Create a grid of points for the selected features
    x_min, x_max = feature1.min() - 1, feature1.max() + 1
    y_min, y_max = feature2.min() - 1, feature2.max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Create input for prediction by setting other features to their mean
    X_for_grid = np.zeros((grid.shape[0], X.shape[1]))
    X_for_grid[:, feature1_idx] = grid[:, 0]
    X_for_grid[:, feature2_idx] = grid[:, 1]
    for i in range(X.shape[1]):
        if i not in feature_indices:
            X_for_grid[:, i] = X[:, i].mean()

    # Predict probabilities for the grid
    grid_preds = predict(
        mcmc, jnp.array(X_for_grid), multiclass_model, n_classes=n_classes
    )

    # Ensure consistency in the number of classes
    if grid_preds.shape[-1] != n_classes:
        raise ValueError(
            f"Mismatch in number of classes: grid_preds has {grid_preds.shape[-1]} classes, expected {n_classes}."
        )

    # Reshape grid_preds to match grid resolution
    grid_mean = grid_preds.mean(axis=0).reshape(
        grid_resolution, grid_resolution, n_classes
    )
    grid_uncertainty = grid_preds.var(axis=0).reshape(
        grid_resolution, grid_resolution, n_classes
    )

    # Plot decision boundaries and uncertainties for each class
    plt.figure(figsize=(10, 6))
    cmap = ListedColormap(plt.cm.tab10.colors[:n_classes])

    # Plot predicted class regions
    predicted_classes = grid_mean.argmax(axis=2)
    plt.contourf(
        xx,
        yy,
        predicted_classes,
        alpha=0.5,
        cmap=cmap,
        levels=np.arange(n_classes + 1) - 0.5,
    )
    plt.colorbar(ticks=np.arange(n_classes), label="Predicted Class")

    # Overlay uncertainty for each class as brightness effect
    for class_idx in range(n_classes):
        plt.imshow(
            grid_uncertainty[:, :, class_idx],
            extent=(x_min, x_max, y_min, y_max),
            origin="lower",
            cmap="binary",
            alpha=0.3,
            aspect="auto",
        )

    # Scatter plot for true labels
    for class_idx in range(n_classes):
        plt.scatter(
            feature1[y == class_idx],
            feature2[y == class_idx],
            label=f"Class {class_idx}",
            edgecolor="k",
            alpha=0.6,
        )

    plt.xlabel(f"Feature {feature1_idx + 1}")
    plt.ylabel(f"Feature {feature2_idx + 1}")
    plt.title("Multiclass Decision Boundary with Uncertainty")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()
