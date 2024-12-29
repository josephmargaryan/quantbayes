import jax.numpy as jnp


def multiclass_log_loss(pred_probs, true_labels):
    pred_probs = jnp.clip(pred_probs, 1e-7, 1 - 1e-7)  # Avoid log(0)
    log_probs = jnp.log(pred_probs)
    return -jnp.mean(log_probs[jnp.arange(len(true_labels)), true_labels])


def binary_log_loss(pred_probs, true_labels):
    """
    Binary log loss (cross-entropy) for binary classification.

    Parameters:
    - pred_probs: Predicted probabilities (array of shape (n_samples,)).
    - true_labels: True binary labels (array of shape (n_samples,), values in {0, 1}).

    Returns:
    - Binary log loss (scalar).
    """
    pred_probs = jnp.clip(pred_probs, 1e-7, 1 - 1e-7)  # Avoid log(0)
    return -jnp.mean(
        true_labels * jnp.log(pred_probs) + (1 - true_labels) * jnp.log(1 - pred_probs)
    )


def regression_mse(pred_values, true_values):
    """
    Mean squared error for regression.

    Parameters:
    - pred_values: Predicted values (array of shape (n_samples,)).
    - true_values: True target values (array of shape (n_samples,)).

    Returns:
    - Mean squared error (scalar).
    """
    return jnp.mean((pred_values - true_values) ** 2)


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


def compute_kl_divergence_hierarchical(
    mean_posterior,
    std_posterior,
    mean_prior=0,
    std_prior=1,
    precision_posterior_mean=None,
    precision_posterior_std=None,
    alpha_prior=1.0,
    beta_prior=0.1,
):
    """
    Compute the KL divergence for hierarchical priors with Gamma-distributed precision.

    Parameters:
    - mean_posterior: Mean of the posterior distribution for weights/biases (array).
    - std_posterior: Standard deviation of the posterior distribution for weights/biases (array).
    - mean_prior: Mean of the Gaussian prior for weights/biases (default=0).
    - std_prior: Standard deviation of the Gaussian prior for weights/biases (default=1).
    - precision_posterior_mean: Mean of the posterior distribution for precision.
    - precision_posterior_std: Standard deviation of the posterior distribution for precision.
    - alpha_prior: Shape parameter of the Gamma prior for precision.
    - beta_prior: Rate parameter of the Gamma prior for precision.

    Returns:
    - kl_divergence: Total KL divergence for hierarchical model.
    """
    # KL divergence for weights/biases
    kl_weights = (
        0.5
        * (
            (std_posterior / std_prior) ** 2
            + ((mean_posterior - mean_prior) / std_prior) ** 2
            - 1
            + 2 * jnp.log(std_prior / std_posterior)
        ).sum()
    )

    # KL divergence for Gamma-distributed precision
    if precision_posterior_mean is not None and precision_posterior_std is not None:
        # Gamma prior: KL term for precision
        alpha_post = (precision_posterior_mean / precision_posterior_std) ** 2
        beta_post = precision_posterior_mean / (precision_posterior_std**2)
        kl_precision = (
            alpha_post * jnp.log(beta_post / beta_prior)
            + alpha_prior * (jnp.log(beta_prior / beta_post) - 1)
            + jnp.log(jnp.exp(beta_prior) / jnp.exp(beta_post))
            + jnp.exp(beta_post - beta_prior)
            - alpha_post
        ).sum()
    else:
        kl_precision = 0

    # Total KL divergence
    kl_divergence = kl_weights + kl_precision
    return kl_divergence


def extract_posteriors(posterior_samples):
    """
    Extract posterior mean and standard deviation for weights, biases, and precision.
    """
    w1_mean = posterior_samples["w_hidden"].mean(axis=0)
    w1_std = posterior_samples["w_hidden"].std(axis=0)

    b1_mean = posterior_samples["b_hidden"].mean(axis=0)
    b1_std = posterior_samples["b_hidden"].std(axis=0)

    w2_mean = posterior_samples["w_out"].mean(axis=0)
    w2_std = posterior_samples["w_out"].std(axis=0)

    b2_mean = posterior_samples["b_out"].mean(axis=0)
    b2_std = posterior_samples["b_out"].std(axis=0)

    prec_mean = (
        posterior_samples["precision"].mean()
        if "precision" in posterior_samples
        else None
    )
    prec_std = (
        posterior_samples["precision"].std()
        if "precision" in posterior_samples
        else None
    )

    mean_posterior = jnp.concatenate(
        [w1_mean.flatten(), b1_mean, w2_mean.flatten(), b2_mean]
    )
    std_posterior = jnp.concatenate(
        [w1_std.flatten(), b1_std, w2_std.flatten(), b2_std]
    )

    return mean_posterior, std_posterior, prec_mean, prec_std


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
