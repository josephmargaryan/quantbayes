from numpyro.contrib.einstein import SteinVI, RBFKernel, MixtureGuidePredictive
import jax
import jax.numpy as jnp
from jax import random
from numpyro.optim import Adam, Adagrad
from numpyro.infer.autoguide import AutoNormal
import matplotlib.pyplot as plt
import numpy as np


def train_regressor(bnn_model, X_train, y_train, num_steps=1000):
    """
    Train a Bayesian Neural Network for regression using SteinVI.
    """
    guide = AutoNormal(bnn_model)

    stein = SteinVI(
        model=bnn_model,
        guide=guide,
        optim=Adagrad(0.5),
        kernel_fn=RBFKernel(),
        repulsion_temperature=1.0,
        num_stein_particles=5,
        num_elbo_particles=50,
    )

    rng_key = random.PRNGKey(0)
    stein_result = stein.run(rng_key, num_steps, X_train, y_train, progress_bar=True)

    return stein, stein_result


def train_multiclass(
    bnn_model, X_train, y_train, num_classes, num_steps=1000, init_strategy=None
):
    """
    Train a Bayesian Neural Network for multiclass classification using SteinVI.
    """
    if init_strategy:
        guide = AutoNormal(bnn_model, init_loc_fn=init_strategy)
    else:
        guide = AutoNormal(bnn_model)
    stein = SteinVI(
        model=bnn_model,
        guide=guide,
        optim=Adagrad(0.5),
        kernel_fn=RBFKernel(),
        repulsion_temperature=1.0,
        num_stein_particles=5,
        num_elbo_particles=50,
    )

    rng_key = random.PRNGKey(0)
    stein_result = stein.run(
        rng_key, num_steps, X_train, y_train, num_classes=num_classes, progress_bar=True
    )

    return stein, stein_result


def train_binary(bnn_model, X_train, y_train, num_steps=1000):
    """
    Train a Bayesian Neural Network for binary classification using SteinVI.
    """
    guide = AutoNormal(bnn_model)

    stein = SteinVI(
        model=bnn_model,
        guide=guide,
        optim=Adagrad(0.5),
        kernel_fn=RBFKernel(),
        repulsion_temperature=1.0,
        num_stein_particles=5,
        num_elbo_particles=50,
    )

    rng_key = random.PRNGKey(0)
    stein_result = stein.run(rng_key, num_steps, X_train, y_train, progress_bar=True)

    return stein, stein_result


def predict_regressor(stein, bnn_model, stein_result, X_test):
    """
    Generate predictions for regression using a trained Bayesian Neural Network.
    """
    predictive = MixtureGuidePredictive(
        bnn_model,
        stein.guide,
        params=stein.get_params(stein_result.state),
        num_samples=100,
        guide_sites=stein.guide_sites,
    )

    rng_key = random.PRNGKey(1)
    predictions = predictive(rng_key, X_test)["obs"]

    return predictions


def predict_multiclass(
    stein, bnn_model, stein_result, X_test, num_classes=3, sample_from="obs"
):
    """
    Generate predictions for multiclass classification using a trained Bayesian Neural Network.
    """

    def bnn_model_multiclass(X, y=None):
        return bnn_model(X, y=y, num_classes=num_classes)

    predictive = MixtureGuidePredictive(
        bnn_model_multiclass,
        stein.guide,
        params=stein.get_params(stein_result.state),
        num_samples=100,
        guide_sites=stein.guide_sites,
    )

    rng_key = random.PRNGKey(1)
    pred_samples = predictive(rng_key, X_test)

    probs_samples = pred_samples[sample_from]

    return probs_samples


def predict_binary(stein, bnn_model, stein_result, X_test, sample_from="obs"):
    """
    Generate predictions for binary classification using a trained Bayesian Neural Network.
    """

    predictive = MixtureGuidePredictive(
        bnn_model,
        stein.guide,
        params=stein.get_params(stein_result.state),
        num_samples=100,
        guide_sites=stein.guide_sites,
    )

    rng_key = random.PRNGKey(1)
    pred_samples = predictive(rng_key, X_test)[sample_from]
    return pred_samples


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
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    mean_preds = np.array(mean_preds)
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)

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

    plt.figure(figsize=(10, 6))
    plt.scatter(feature, y_test, color="blue", alpha=0.6, label="True Targets")
    plt.plot(
        feature,
        mean_preds,
        color="red",
        label="Mean Predictions",
        linestyle="-",
        linewidth=2,
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
    model, X, y, stein, stein_results, resolution=100, features=(0, 1)
):
    """
    Plot decision boundaries with uncertainty for binary classification, selecting specific features for visualization.
    Parameters:
        model: Trained Bayesian binary classification model.
        X: Input data (jnp array).
        y: True labels for the input data.
        stein: Trained SteinVI object.
        stein_results: Results from SteinVI training.
        resolution: Grid resolution for visualization.
        features: Tuple of indices specifying which two features to visualize (e.g., (0, 1)).
    """
    feature_1, feature_2 = features
    X_selected = X[:, [feature_1, feature_2]]

    x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
    y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )
    grid_points = jnp.array(np.c_[xx.ravel(), yy.ravel()])
    grid_points_full = jnp.zeros((grid_points.shape[0], X.shape[1]))
    grid_points_full = grid_points_full.at[:, feature_1].set(grid_points[:, 0])
    grid_points_full = grid_points_full.at[:, feature_2].set(grid_points[:, 1])

    pred_samples = predict_binary(
        stein, model, stein_results, grid_points_full, sample_from="logits"
    )
    pred_samples = jax.nn.sigmoid(pred_samples)
    mean_probs = pred_samples.mean(axis=0)
    uncertainty = pred_samples.std(axis=0)
    mean_probs = mean_probs.reshape(xx.shape)
    uncertainty = uncertainty.reshape(xx.shape)

    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, mean_probs, levels=100, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.colorbar(label="Predicted Probability (Class 1)")

    plt.contourf(xx, yy, uncertainty, levels=20, cmap="gray", alpha=0.4)
    plt.colorbar(label="Uncertainty (Standard Deviation)")

    plt.scatter(
        X_selected[:, 0], X_selected[:, 1], c=y, edgecolor="k", cmap=plt.cm.RdYlBu
    )
    plt.title(
        f"Binary Decision Boundaries with Uncertainty (Features {features[0]} and {features[1]})"
    )
    plt.xlabel(f"Feature {feature_1 + 1}")
    plt.ylabel(f"Feature {feature_2 + 1}")
    plt.show()


def visualize_multiclass(
    model, X, y, stein, stein_results, num_classes=3, resolution=100, features=(0, 1)
):
    """
    Plot decision boundaries with uncertainty for multiclass classification, selecting specific features for visualization.
    Parameters:
        model: Trained Bayesian multiclass classification model.
        X: Input data (jnp array).
        y: True labels for the input data.
        stein: Trained SteinVI object.
        stein_results: Results from SteinVI training.
        num_classes: Number of classes for multiclass classification.
        resolution: Grid resolution for visualization.
        features: Tuple of indices specifying which two features to visualize (e.g., (0, 1)).
    """
    feature_1, feature_2 = features
    X_selected = X[:, [feature_1, feature_2]]
    x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
    y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )
    grid_points = jnp.array(np.c_[xx.ravel(), yy.ravel()])

    grid_points_full = jnp.zeros((grid_points.shape[0], X.shape[1]))
    grid_points_full = grid_points_full.at[:, feature_1].set(grid_points[:, 0])
    grid_points_full = grid_points_full.at[:, feature_2].set(grid_points[:, 1])

    pred_samples = predict_multiclass(
        stein, model, stein_results, grid_points_full, num_classes, sample_from="logits"
    )
    mean_probs = pred_samples.mean(axis=0)
    probabilities = jax.nn.softmax(mean_probs, axis=-1)
    uncertainty = -jnp.sum(
        probabilities * jnp.log(probabilities + 1e-9), axis=1
    )  # Entropy as uncertainty metric

    mean_class = jnp.argmax(probabilities, axis=1).reshape(xx.shape)
    uncertainty = uncertainty.reshape(xx.shape)

    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, mean_class, alpha=0.6, cmap=plt.cm.RdYlBu)
    plt.colorbar(label="Predicted Class")

    plt.contourf(xx, yy, uncertainty, alpha=0.3, cmap="gray", levels=15)
    plt.colorbar(label="Uncertainty (Entropy)")

    plt.scatter(
        X_selected[:, 0], X_selected[:, 1], c=y, edgecolor="k", cmap=plt.cm.RdYlBu
    )
    plt.title(
        f"Multiclass Decision Boundaries with Uncertainty (Features {features[0]} and {features[1]})"
    )
    plt.xlabel(f"Feature {feature_1 + 1}")
    plt.ylabel(f"Feature {feature_2 + 1}")
    plt.show()
