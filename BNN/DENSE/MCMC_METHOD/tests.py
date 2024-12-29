from BNN.DENSE.MCMC_METHOD.models import (
    regression_model,
    binary_model,
    multiclass_model,
)
from BNN.DENSE.MCMC_METHOD.utils import (
    run_inference,
    visualize_regression,
    predict_binary,
    predict_multiclass,
    predict_regressor,
    visualize_binary,
    visualize_multiclass,
)
from BNN.DENSE.MCMC_METHOD.fake_data import (
    generate_simple_regression_data,
    generate_binary_classification_data,
    generate_multiclass_classification_data,
)
from sklearn.model_selection import train_test_split
import numpy as np
import jax.numpy as jnp
from jax import random
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    recall_score,
    precision_score,
    log_loss,
)
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

    mcmc = run_inference(
        regression_model, rng_key, X_train, y_train, num_samples=1000, num_warmup=500
    )

    predictions = predict_regressor(mcmc, X_test, regression_model)

    mean_preds = predictions.mean(axis=0)
    std_preds = predictions.std(axis=0)
    lower_bound = mean_preds - 1.96 * std_preds
    upper_bound = mean_preds + 1.96 * std_preds

    MSE = mean_squared_error(y_test, mean_preds)
    RMSE = root_mean_squared_error(y_test, mean_preds)
    MAE = mean_absolute_error(y_test, mean_preds)
    print(f"MSE: {MSE}\nRMSE: {RMSE}\nMAE: {MAE}")

    visualize_regression(X_test, y_test, mean_preds, lower_bound, upper_bound, 0)


def test_binary():
    rng_key = random.key(1)
    df = generate_binary_classification_data()
    X, y = df.drop(columns=["target"], axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24
    )

    mcmc = run_inference(
        binary_model, rng_key, X_train, y_train, num_samples=1000, num_warmup=500
    )

    predictions = predict_binary(mcmc, X_test, binary_model, sample_from="logits")
    mean_preds = predictions.mean(axis=0)
    probabilities = jax.nn.sigmoid(mean_preds)
    std_preds = predictions.std(axis=0)
    binary_preds = np.array((probabilities >= 0.5).astype(int))
    y_preds = np.array(y_test)
    accuracy = accuracy_score(y_preds, binary_preds)
    precision = precision_score(y_preds, binary_preds)
    recall = recall_score(y_preds, binary_preds)
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}")

    visualize_binary(
        X_test,
        y_test,
        mcmc,
        predict_binary,
        binary_model,
        feature_indices=(0, 1),
        grid_resolution=200,
    )


def test_multiclass():
    rng_key = random.key(2)
    df = generate_multiclass_classification_data()
    X, y = df.drop(columns=["target"], axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=45, test_size=0.2
    )

    mcmc = run_inference(multiclass_model, rng_key, X_train, y_train, 100, 50)
    predictions = predict_multiclass(
        mcmc, X_test, multiclass_model, sample_from="logits"
    )
    mean_preds = predictions.mean(axis=0)
    probabilities = jax.nn.softmax(mean_preds, axis=-1)
    std_preds = predictions.std(axis=0)

    loss = log_loss(np.array(y_test), np.array(probabilities))
    print(f"Loss: {loss}")

    visualize_multiclass(
        X_test,
        y_test,
        mcmc,
        predict_multiclass,
        multiclass_model,
        feature_indices=(0, 1),  # Features to visualize
        grid_resolution=200,  # Resolution for decision boundary
    )


##################################
def multiclass_log_loss(pred_probs, true_labels):
    pred_probs = jnp.clip(pred_probs, 1e-7, 1 - 1e-7)  # Avoid log(0)
    log_probs = jnp.log(pred_probs)
    return -jnp.mean(log_probs[jnp.arange(len(true_labels)), true_labels])


def compute_empirical_risk(predictions, y_true, loss_fn):
    # Alternatively, aggregate risks across samples for robustness
    risks = jnp.array([loss_fn(pred, y_true) for pred in predictions])
    empirical_risk = risks.mean()
    return empirical_risk


def compute_kl_divergence(mean_posterior, std_posterior, mean_prior=0, std_prior=1):
    """
    Compute the KL divergence between a Gaussian posterior and a Gaussian prior.

    Parameters:
    - mean_posterior: Mean of the posterior distribution (array).
    - std_posterior: Standard deviation of the posterior distribution (array).
    - mean_prior: Mean of the prior distribution (default=0).
    - std_prior: Standard deviation of the prior distribution (default=1).

    Returns:
    - kl_divergence: KL divergence between posterior and prior.
    """
    kl_divergence = (
        0.5
        * (
            (std_posterior / std_prior) ** 2
            + ((mean_posterior - mean_prior) / std_prior) ** 2
            - 1
            + 2 * jnp.log(std_prior / std_posterior)
        ).sum()
    )
    return kl_divergence


def compute_confidence_term(kl_divergence, num_samples, delta=0.05):
    """
    Compute the confidence term in the PAC-Bayesian bound.

    Parameters:
    - kl_divergence: KL divergence between posterior and prior.
    - num_samples: Number of training samples.
    - delta: Confidence level (default=0.05 for 95% confidence).

    Returns:
    - confidence_term: Square root term in the PAC-Bayesian bound.
    """
    confidence_term = jnp.sqrt((kl_divergence + jnp.log(1 / delta)) / (2 * num_samples))
    return confidence_term


def pac_bayesian_bound(
    predictions, y_true, mean_posterior, std_posterior, num_samples, loss_fn, delta=0.05
):
    """
    Compute the PAC-Bayesian bound for a model.

    Parameters:
    - predictions: Posterior samples of shape (num_samples, num_data_points).
    - y_true: True labels or targets (array of shape (num_data_points,)).
    - mean_posterior: Mean of the posterior distribution.
    - std_posterior: Standard deviation of the posterior distribution.
    - num_samples: Number of training samples.
    - loss_fn: Loss function (e.g., mean squared error, log loss).
    - delta: Confidence level (default=0.05).

    Returns:
    - bound: PAC-Bayesian bound for the model.
    """
    empirical_risk = compute_empirical_risk(predictions, y_true, loss_fn)
    kl_divergence = compute_kl_divergence(mean_posterior, std_posterior)
    confidence_term = compute_confidence_term(kl_divergence, num_samples, delta)

    bound = empirical_risk + confidence_term
    return bound


def test_multiclass2():
    rng_key = random.key(2)
    df = generate_multiclass_classification_data()
    X, y = df.drop(columns=["target"], axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=45, test_size=0.2
    )

    mcmc = run_inference(multiclass_model, rng_key, X_train, y_train, 100, 50)
    predictions = predict_multiclass(
        mcmc, X_test, multiclass_model, sample_from="logits"
    )
    pred = jax.nn.softmax(predictions, axis=-1)
    risks = jnp.array([log_loss(pred, y_test) for pred in predictions])
    empirical_risk = risks.mean()
    mean_preds = predictions.mean(axis=0)
    probabilities = jax.nn.softmax(mean_preds, axis=-1)
    std_preds = predictions.std(axis=0)

    loss = log_loss(np.array(y_test), np.array(probabilities))
    print(f"Loss: {loss}")
    # Example for multiclass classification
    empirical_risk = risks
    kl_divergence = compute_kl_divergence(mean_preds, std_preds)
    confidence_term = compute_confidence_term(kl_divergence, len(X_train))

    pac_bound = empirical_risk + confidence_term
    print(f"PAC-Bayesian Bound: {pac_bound}")


if __name__ == "__main__":
    """
    print("Testing Binary")
    test_binary()
    print("Testing Regressor")
    test_regression()
    print("Testing Multiclass")"""
    test_multiclass2()
