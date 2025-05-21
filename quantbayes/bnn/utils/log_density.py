import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import expit, logsumexp
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm, bernoulli
from numpyro.infer import Predictive
from jaxtyping import PRNGKeyArray

"""
Example Usage:
param_keys = {
    "weights": "layer_w",
    "bias": "layer_b"
}
density = PredictiveLogDensityCalculator(model.get_samples, mode="multiclass", param_keys=param_keys)
log_density = density.compute_predictive_log_density(X_test, y_test)
"""


class Lppd:
    def __init__(
        self, mcmc_samples, mode="binary", param_keys=None, custom_likelihood_func=None
    ):
        """
        A class for computing the Predictive Log Density (Log Predictive Likelihood)
        using MCMC samples for Bayesian models. Accepts both a list of sample dictionaries
        and a dictionary of arrays (e.g., from model.get_samples()).

        It computes:
            lppd = sum_{i=1}^N log((1/S) * sum_{s=1}^S p(y_i | x_i, θ_s))
        where S is the number of MCMC samples.

        Parameters:
            mcmc_samples (iterable or dict or callable): Either:
                - An iterable (e.g., list) of parameter sample dictionaries, or
                - A dictionary of NumPy arrays (e.g., output from model.get_samples()).
                  In the latter case, each key's array should have the first dimension as the number of samples.
                - A callable returning one of the above.
            mode (str): "binary" or "multiclass", specifying the type of classification.
            param_keys (dict): A mapping of parameter names. For example:
                {"weights": "layer_w", "bias": "layer_b"}
                If None, defaults to {"weights": "weights", "bias": "bias"}.
            custom_likelihood_func (callable): Optional. A function that accepts (sample, X, y) and returns
                the likelihood (not the log) for each data point. If provided, it overrides the default computation.
        """
        if mode not in ["binary", "multiclass"]:
            raise ValueError("mode must be either 'binary' or 'multiclass'")
        self.mode = mode

        # If mcmc_samples is callable, call it.
        if callable(mcmc_samples):
            mcmc_samples = mcmc_samples()

        # If mcmc_samples is a dictionary of arrays, convert it to a list of dictionaries.
        if isinstance(mcmc_samples, dict):
            # Assume each value is a numpy array where the first dimension is sample index.
            n_samples = None
            for key, array in mcmc_samples.items():
                if n_samples is None:
                    n_samples = array.shape[0]
                else:
                    if array.shape[0] != n_samples:
                        raise ValueError(
                            "All arrays in the mcmc_samples dictionary must have the same first dimension."
                        )
            samples_list = []
            for i in range(n_samples):
                sample = {key: array[i] for key, array in mcmc_samples.items()}
                samples_list.append(sample)
            self.mcmc_samples = samples_list
        else:
            self.mcmc_samples = mcmc_samples

        self.param_keys = (
            param_keys
            if param_keys is not None
            else {"weights": "weights", "bias": "bias"}
        )
        self.custom_likelihood_func = custom_likelihood_func

    def compute_likelihood_sample(self, sample, X, y):
        """
        Compute the likelihood for each data point given a single MCMC sample.

        Parameters:
            sample (dict): A single MCMC sample.
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): True labels.
                For binary: shape (n_samples,) with values 0 or 1.
                For multiclass: shape (n_samples,) with integer class labels.

        Returns:
            np.ndarray: Likelihood values for each data point.
        """
        if self.custom_likelihood_func is not None:
            return self.custom_likelihood_func(sample, X, y)

        try:
            weights = sample[self.param_keys["weights"]]
            bias = sample[self.param_keys["bias"]]
        except (TypeError, KeyError) as e:
            raise ValueError(
                "Each sample should be a dictionary with keys specified in 'param_keys'."
            ) from e

        if self.mode == "binary":
            logits = np.dot(X, weights) + bias  # shape: (n_samples,)
            prob = expit(logits)
            # Compute likelihood p(y|x,θ) = p^y * (1-p)^(1-y) for each data point.
            likelihood = (prob**y) * ((1 - prob) ** (1 - y))
            return likelihood

        elif self.mode == "multiclass":
            logits = np.dot(X, weights) + bias  # shape: (n_samples, n_classes)
            log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
            # Likelihood is the probability assigned to the true class.
            likelihood = np.exp(log_probs[np.arange(len(y)), y])
            return likelihood

    def compute_predictive_log_density(self, X, y):
        """
        Compute the Predictive Log Density (lppd) for the dataset by averaging the likelihoods
        over all MCMC samples and then taking the logarithm.

        lppd = sum_{i=1}^N log((1/S) * sum_{s=1}^S p(y_i | x_i, θ_s))

        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): True labels.

        Returns:
            float: The predictive log density.
        """
        n_samples = X.shape[0]
        n_mcmc = len(self.mcmc_samples)
        likelihood_matrix = np.zeros((n_samples, n_mcmc))

        for s, sample in enumerate(self.mcmc_samples):
            likelihood_matrix[:, s] = self.compute_likelihood_sample(sample, X, y)

        # Average likelihood over samples for each data point; add eps to prevent log(0)
        log_likelihoods = np.log(likelihood_matrix + 1e-9)  # Prevent log(0)
        lppd = np.sum(logsumexp(log_likelihoods, axis=1) - np.log(n_mcmc))
        return lppd


def compute_nll(
    model,
    guide,
    params: dict,
    X: jnp.ndarray,
    y: jnp.ndarray,
    *,
    key: PRNGKeyArray,
    num_samples: int = 200,
    mode: str = "multiclass",  # "multiclass", "binary" or "regression"
) -> tuple[float, float, float]:
    """
    Monte Carlo estimate of log predictive density (lppd) and perplexity/RMSE.

    Args:
        model, guide, params: as in SVI.
        X: [N, D] features; y: [N] labels or targets.
        key: PRNGKey.
        num_samples: posterior predictive draws.
        mode: "multiclass", "binary", or "regression".
    Returns:
        key, lppd, avg_lppd, perplexity (or RMSE for regression)
    """
    assert mode in ("multiclass", "binary", "regression")
    key, subkey = jax.random.split(key)

    # pick sites
    sites = ("logits",) if mode in ("multiclass", "binary") else ("mu", "sigma")
    predictive = Predictive(
        model, guide, params, num_samples=num_samples, return_sites=sites
    )
    samples = predictive(subkey, X)  # respects your model’s plates

    if mode == "multiclass":
        # [S,N,C] → log‐probs → pick y
        logits = samples["logits"]
        logp = logits - logsumexp(logits, axis=-1, keepdims=True)  # (S,N,C)
        S, N, C = logp.shape
        yb = jnp.broadcast_to(y, (S, N))
        L = logp[jnp.arange(S)[:, None], jnp.arange(N)[None, :], yb]

    elif mode == "binary":
        # [S,N,1] or [S,N] → Bernoulli(logits)
        logits = samples["logits"]
        # squeeze to (S,N)
        logits = logits.squeeze(-1)
        # bernoulli.logpmf works with prob not logits, so convert:
        p = jax.nn.sigmoid(logits)
        L = bernoulli.logpmf(y, p)  # (S,N)

    else:  # regression
        mu = samples["mu"]  # (S,N)
        sigma = samples["sigma"].reshape((num_samples, 1))
        L = norm.logpdf(y, loc=mu, scale=sigma)  # (S,N)

    # Monte Carlo lppd
    lppd_per_example = logsumexp(L, axis=0) - jnp.log(num_samples)
    lppd = jnp.sum(lppd_per_example)
    avg_lppd = float(lppd) / y.shape[0]

    if mode == "regression":
        # also return RMSE for regression
        # point‐predict via mean mu
        mu_mean = jnp.mean(samples["mu"], axis=0)
        rmse = float(jnp.sqrt(jnp.mean((mu_mean - y) ** 2)))
        return key, float(lppd), avg_lppd, rmse

    perplexity = float(jnp.exp(-avg_lppd))
    return float(lppd), avg_lppd, perplexity
