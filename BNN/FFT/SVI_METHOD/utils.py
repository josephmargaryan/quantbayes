from numpyro.optim import Adam
from numpyro.infer import SVI, Trace_ELBO, Predictive, autoguide
from jax import random
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax


def train(
    X_train, y_train, bnn_model, num_steps=1000, num_classes=None, track_loss=False
):
    """
    Unified training function for Bayesian Neural Networks using SVI.
    """
    guide = autoguide.AutoNormal(bnn_model)
    optimizer = Adam(0.01)

    svi = SVI(
        bnn_model,
        guide,
        optimizer,
        loss=Trace_ELBO(),
    )

    rng_key = random.PRNGKey(0)
    init_args = {}
    if num_classes is not None and "n_classes" in bnn_model.__code__.co_varnames:
        init_args["n_classes"] = num_classes

    svi_state = svi.init(rng_key, X_train, y_train, **init_args)

    loss_progression = [] if track_loss else None

    for step in range(num_steps):
        update_args = {}
        if num_classes is not None and "n_classes" in bnn_model.__code__.co_varnames:
            update_args["n_classes"] = num_classes

        svi_state, loss = svi.update(svi_state, X_train, y_train, **update_args)

        if track_loss:
            loss_progression.append(loss)
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

    params = svi.get_params(svi_state)
    if track_loss:
        return svi, params, loss_progression
    return svi, params


def inference(svi, params, X_test, sample_from="y", num_classes=None):
    """
    Generate predictions using the trained SVI model.
    """
    predictive_args = {}
    if num_classes is not None and "n_classes" in svi.model.__code__.co_varnames:
        predictive_args["n_classes"] = num_classes

    predictive = Predictive(svi.model, guide=svi.guide, params=params, num_samples=100)
    rng_key = random.PRNGKey(1)
    pred_samples = predictive(rng_key, X=X_test, **predictive_args)
    return pred_samples[sample_from]


def visualize_regression(
    X_test, y_test, mean_predictions, lower_bound, upper_bound, feature_index=0
):
    """
    Visualizes predictions on X_test with uncertainties and true y_test values.

    Args:
        X_test: Test input data (shape: (N, D)).
        y_test: True target values (shape: (N,)).
        mean_predictions: Predicted mean values (shape: (N,)).
        lower_bound: Lower bound of the confidence interval (shape: (N,)).
        upper_bound: Upper bound of the confidence interval (shape: (N,)).
        feature_index: Index of the feature to visualize (default: 0).
    """
    sorted_indices = jnp.argsort(X_test[:, feature_index])
    X_test_sorted = X_test[sorted_indices, feature_index]
    y_test_sorted = y_test[sorted_indices]
    mean_predictions_sorted = mean_predictions[sorted_indices]
    lower_bound_sorted = lower_bound[sorted_indices]
    upper_bound_sorted = upper_bound[sorted_indices]

    plt.figure(figsize=(12, 6))
    plt.fill_between(
        X_test_sorted,
        lower_bound_sorted,
        upper_bound_sorted,
        color="gray",
        alpha=0.3,
        label="95% Confidence Interval",
    )
    plt.plot(
        X_test_sorted, mean_predictions_sorted, color="blue", label="Predicted Mean"
    )
    plt.scatter(
        X_test_sorted, y_test_sorted, color="red", label="True Values", alpha=0.8, s=20
    )
    plt.xlabel(f"X_test (Feature {feature_index})")
    plt.ylabel("Predictions / True Values")
    plt.title("Predictions with Uncertainties")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_binary(X, y, svi, params, features=(0, 1), resolution=100):
    """
    Visualizes the binary decision boundaries with uncertainty for a binary classification model.

    Args:
        X: Input data (shape: (N, D)).
        y: True binary labels (shape: (N,)).
        mean_predictions: Predicted mean probabilities (shape: (N,)).
        uncertainty: Predicted uncertainty (shape: (N,)).
        features: Tuple specifying the indices of the two features to visualize.
        resolution: Grid resolution for decision boundary visualization.
    """
    feature_1, feature_2 = features
    X_selected = X[:, [feature_1, feature_2]]

    x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
    y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1

    xx, yy = jnp.meshgrid(
        jnp.linspace(x_min, x_max, resolution), jnp.linspace(y_min, y_max, resolution)
    )
    grid_points = jnp.c_[xx.ravel(), yy.ravel()]

    grid_points_full = jnp.zeros((grid_points.shape[0], X.shape[1]))
    grid_points_full = grid_points_full.at[:, feature_1].set(grid_points[:, 0])
    grid_points_full = grid_points_full.at[:, feature_2].set(grid_points[:, 1])

    grid_predictions = inference(svi, params, grid_points_full, sample_from="obs")
    grid_mean_predictions = grid_predictions.mean(axis=0).reshape(xx.shape)
    grid_uncertainty = grid_predictions.std(axis=0).reshape(xx.shape)

    plt.figure(figsize=(12, 8))
    plt.contourf(
        xx, yy, grid_mean_predictions, levels=100, cmap=plt.cm.RdYlBu, alpha=0.8
    )
    plt.colorbar(label="Predicted Probability (Class 1)")

    plt.contourf(xx, yy, grid_uncertainty, levels=20, cmap="gray", alpha=0.3)
    plt.colorbar(label="Uncertainty (Standard Deviation)")

    plt.scatter(
        X_selected[:, 0], X_selected[:, 1], c=y, edgecolor="k", cmap=plt.cm.RdYlBu, s=20
    )
    plt.title(
        f"Binary Decision Boundaries with Uncertainty (Features {features[0]} and {features[1]})"
    )
    plt.xlabel(f"Feature {feature_1 + 1}")
    plt.ylabel(f"Feature {feature_2 + 1}")
    plt.grid(True)
    plt.show()


def visualize_multiclass(
    X, y, svi, params, num_classes, features=(0, 1), resolution=100
):
    """
    Visualizes multiclass decision boundaries with uncertainty.

    Args:
        X: Input data (shape: (N, D)).
        y: True class labels (shape: (N,)).
        mean_predictions: Predicted probabilities (shape: (N, num_classes)).
        uncertainty: Uncertainty (entropy) for each prediction (shape: (N,)).
        num_classes: Number of classes.
        features: Tuple specifying the indices of the two features to visualize.
        resolution: Grid resolution for decision boundary visualization.
    """
    feature_1, feature_2 = features
    X_selected = X[:, [feature_1, feature_2]]

    x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
    y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1

    xx, yy = jnp.meshgrid(
        jnp.linspace(x_min, x_max, resolution), jnp.linspace(y_min, y_max, resolution)
    )
    grid_points = jnp.c_[xx.ravel(), yy.ravel()]

    grid_points_full = jnp.zeros((grid_points.shape[0], X.shape[1]))
    grid_points_full = grid_points_full.at[:, feature_1].set(grid_points[:, 0])
    grid_points_full = grid_points_full.at[:, feature_2].set(grid_points[:, 1])

    grid_predictions = inference(
        svi, params, grid_points_full, sample_from="logits", num_classes=num_classes
    )
    grid_mean_predictions = jax.nn.softmax(grid_predictions.mean(axis=0), axis=-1)
    grid_uncertainty = -jnp.sum(
        grid_mean_predictions * jnp.log(grid_mean_predictions + 1e-9), axis=1
    )  # Entropy
    grid_classes = jnp.argmax(grid_mean_predictions, axis=1).reshape(xx.shape)
    grid_uncertainty = grid_uncertainty.reshape(xx.shape)

    plt.figure(figsize=(12, 8))
    plt.contourf(
        xx, yy, grid_classes, levels=num_classes, cmap=plt.cm.RdYlBu, alpha=0.6
    )
    plt.colorbar(label="Predicted Class")

    plt.contourf(xx, yy, grid_uncertainty, levels=20, cmap="gray", alpha=0.4)
    plt.colorbar(label="Uncertainty (Entropy)")

    plt.scatter(
        X_selected[:, 0], X_selected[:, 1], c=y, edgecolor="k", cmap=plt.cm.RdYlBu, s=20
    )
    plt.title(
        f"Multiclass Decision Boundaries with Uncertainty (Features {features[0]} and {features[1]})"
    )
    plt.xlabel(f"Feature {feature_1 + 1}")
    plt.ylabel(f"Feature {feature_2 + 1}")
    plt.grid(True)
    plt.show()
