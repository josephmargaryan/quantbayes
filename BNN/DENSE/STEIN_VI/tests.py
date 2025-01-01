from BNN.DENSE.STEIN_VI.models import (
    regression_model,
    binary_model,
    multiclass_model,
    hierarchical_multiclass,
    hierarchical_binary,
    hierarchical_regressor,
)
from BNN.DENSE.STEIN_VI.fake_data import (
    generate_simple_regression_data,
    generate_binary_classification_data,
    generate_multiclass_classification_data,
)
from jax import random
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from BNN.DENSE.STEIN_VI.utils import (
    train_binary,
    train_multiclass,
    train_regressor,
    predict_binary,
    predict_multiclass,
    predict_regressor,
    visualize_regression,
    visualize_binary,
    visualize_multiclass,
)
from sklearn.metrics import root_mean_squared_error, accuracy_score, log_loss
import numpy as np
import jax


def test_regression():
    n_samples = 500
    n_features = 8
    random_seed = 42
    rng_key = random.key(0)

    simple_data = generate_simple_regression_data(
        n_samples, n_features, random_seed=random_seed
    )

    X, y = simple_data.drop(columns=["target"], axis=1), simple_data["target"]
    X, y = jnp.array(X), jnp.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=24, test_size=0.2
    )

    stein, stein_result = train_regressor(
        hierarchical_regressor, X_train, y_train, 1000
    )
    predictions = predict_regressor(stein, hierarchical_regressor, stein_result, X_test)
    mean_preds = predictions.mean(axis=0)
    std_preds = predictions.std(axis=0)
    lower_bound = mean_preds - 1.96 * std_preds
    upper_bound = mean_preds + 1.96 * std_preds
    RMSE = root_mean_squared_error(np.array(y_test), np.array(mean_preds))
    print(f"RMSE: {RMSE}")
    visualize_regression(X_test, y_test, mean_preds, lower_bound, upper_bound, 0)


def test_binary():
    from BNN.model_calibration import plot_calibration_curve, plot_roc_curve

    df = generate_binary_classification_data()
    X, y = df.drop(columns=["target"], axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=34
    )

    stein, stein_results = train_binary(hierarchical_binary, X_train, y_train, 2000)

    pred_samples = predict_binary(
        stein, hierarchical_binary, stein_results, X_test, sample_from="logits"
    )
    probs = jax.nn.sigmoid(pred_samples)
    mean_predictions = probs.mean(axis=0)
    loss = log_loss(np.array(y_test), np.array(mean_predictions))
    print(f"Loss: {loss}")
    visualize_binary(binary_model, X_test, y_test, stein, stein_results, 100, (2, 3))
    plot_roc_curve(np.array(y_test), np.array(mean_predictions))
    plot_calibration_curve(np.array(y_test), np.array(mean_predictions))


def test_multiclass():
    df = generate_multiclass_classification_data()
    X, y = df.drop(columns=["target"], axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24
    )
    num_classes = len(jnp.unique(y_train))
    stein, stein_results = train_multiclass(
        hierarchical_multiclass, X_train, y_train, num_classes
    )
    pred_samples = predict_multiclass(
        stein,
        hierarchical_multiclass,
        stein_results,
        X_test,
        num_classes,
        sample_from="logits",
    )
    mean_predictions = pred_samples.mean(axis=0)
    probabilities = jax.nn.softmax(mean_predictions, axis=-1)
    loss = log_loss(np.array(y_test), np.array(probabilities))
    print(f"Loss: {loss}")
    visualize_multiclass(
        multiclass_model,
        X_test,
        y_test,
        stein,
        stein_results,
        num_classes=X_train.shape[1],
        resolution=100,
        features=(0, 1),
    )


if __name__ == "__main__":
    print("Testing Binary")
    test_binary()
    """    
    print("Testing Regressor")
    test_regression()
    print("Testing Multiclass")
    test_multiclass()"""
