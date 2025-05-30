import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from numpyro.infer import Predictive
from scipy.special import logsumexp

__all__ = ["NLL"]

"""
# Regression + NUTS
avg_nll, rmse = NLL(
    model=BNN,
    inference="nuts",
    seed=1,
    X=X_test,
    y=y_test,
    mcmc=mcmc,
    mode="regression",
)

# Classification + SVI
avg_nll, perplexity = NLL(
    model=BNN,
    inference="svi",
    seed=1,
    X=X_test,
    y=y_test,
    guide=guide,
    params=svi_params,
    num_samples=500,
    mode="binary",
)

"""


def get_predictive_samples(
    model,
    inference: str,
    *,
    seed: int,
    X: jnp.ndarray,
    mcmc=None,
    guide=None,
    params=None,
    num_samples: int = 200,
    return_sites=("mu", "sigma"),
):
    key = PRNGKey(seed)

    if inference.lower() == "nuts":
        if mcmc is None:
            raise ValueError("For NUTS, `mcmc` must be provided")
        samples = mcmc.get_samples(flatten_chains=True)
        predictive = Predictive(model, samples, return_sites=return_sites)
        return predictive(key, X)

    elif inference.lower() == "svi":
        if guide is None or params is None:
            raise ValueError("For SVI, both `guide` and `params` must be provided")
        predictive = Predictive(
            model,
            guide=guide,
            params=params,
            num_samples=num_samples,
            return_sites=return_sites,
        )
        return predictive(key, X)

    else:
        raise ValueError("`inference` must be 'nuts' or 'svi'")


def compute_nll_and_metric(preds: dict, y: np.ndarray, mode: str):
    # Determine if regression or classification by returned sites
    if "mu" in preds and "sigma" in preds:
        # Regression
        mu = np.array(preds["mu"])  # (S, N)
        sigma = np.array(preds["sigma"]).reshape(mu.shape)
        S, N = mu.shape

        # Gaussian log-probs
        logp = -0.5 * ((y - mu) ** 2 / (sigma**2) + np.log(2 * np.pi * sigma**2))

        # NLL
        lppd_i = logsumexp(logp, axis=0) - np.log(S)
        avg_nll = -np.mean(lppd_i)

        # RMSE of posterior mean
        mu_mean = mu.mean(axis=0)
        rmse = np.sqrt(np.mean((mu_mean - y) ** 2))
        return avg_nll, rmse

    else:
        # Classification
        logits = np.array(preds["logits"])  # (S, N) or (S, N, C)
        S = logits.shape[0]

        if logits.ndim == 2:
            # Binary
            probs = 1 / (1 + np.exp(-logits))
            logp = y * np.log(probs) + (1 - y) * np.log(1 - probs)
        else:
            # Multiclass
            logp_all = logits - logsumexp(logits, axis=-1, keepdims=True)
            N = y.shape[0]
            logp = logp_all[np.arange(N), y]

        # NLL
        lppd_i = logsumexp(logp, axis=0) - np.log(S)
        avg_nll = -np.mean(lppd_i)

        # Perplexity
        perplexity = np.exp(-np.mean(lppd_i))
        return avg_nll, perplexity


def NLL(
    model,
    inference: str,
    seed: int,
    X: jnp.ndarray,
    y: np.ndarray,
    *,
    mcmc=None,
    guide=None,
    params=None,
    num_samples: int = 200,
    mode: str = "regression",
):
    """
    Compute average NLL and RMSE or perplexity for regression or classification.

    Args:
      model:       NumPyro model function.
      inference:   "nuts" or "svi".
      seed:        int, random seed.
      X:           JAX array of features, shape (N, D).
      y:           NumPy array of targets/labels, shape (N,).
      mcmc:        MCMC object (if inference="nuts").
      guide:       guide function (if inference="svi").
      params:      guide parameters (if inference="svi").
      num_samples: number of predictive draws for SVI.
      mode:        "regression", "binary", or "multiclass".

    Returns:
      avg_nll: average negative log-likelihood per example
      metric:  RMSE (if regression) or perplexity (if classification)
    """
    # choose which sites to return
    return_sites = ("mu", "sigma") if mode == "regression" else ("logits",)

    # get predictive samples dict
    preds = get_predictive_samples(
        model,
        inference,
        seed=seed,
        X=X,
        mcmc=mcmc,
        guide=guide,
        params=params,
        num_samples=num_samples,
        return_sites=return_sites,
    )

    # compute and return NLL and the relevant metric
    return compute_nll_and_metric(preds, y, mode=mode)
