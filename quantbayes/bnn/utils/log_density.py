import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from numpyro.infer import Predictive
from scipy.special import logsumexp

__all__ = ["NLL"]


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
    """
    Draw from the posterior predictive using either NUTS or SVI.

    Returns a dict of samples at the requested return_sites.
    """
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
    """
    Given predictive samples and true targets, compute:
      - regression: avg NLL and RMSE
      - classification: avg NLL and perplexity
    """
    # Regression branch
    if "mu" in preds and "sigma" in preds:
        mu = np.array(preds["mu"])  # (S, N)
        sigma = np.array(preds["sigma"])  # may come (S, N, 1) or (S, N)
        sigma = sigma.reshape(mu.shape)
        S, N = mu.shape

        # Gaussian log-probs per sample, per data point
        logp = -0.5 * ((y - mu) ** 2 / sigma**2 + np.log(2 * np.pi * sigma**2))

        # Log point-wise predictive density
        lppd_i = logsumexp(logp, axis=0) - np.log(S)
        avg_nll = -np.mean(lppd_i)

        # RMSE of posterior mean
        mu_mean = mu.mean(axis=0)
        rmse = np.sqrt(np.mean((mu_mean - y) ** 2))

        return avg_nll, rmse

    # Classification branch (binary or multiclass)
    else:
        logits = np.array(
            preds["logits"]
        )  # (S, N) for binary, (S, N, C) for multiclass
        S = logits.shape[0]

        if logits.ndim == 2:
            # Binary: treat logits as log-odds
            # Expand to shape (S, N, 2) if desired, but simpler to compute log-prob directly
            # logp[s,i] = y[i] * log σ(l) + (1-y[i]) * log (1-σ(l))
            probs = 1 / (1 + np.exp(-logits))
            logp = y * np.log(probs) + (1 - y) * np.log(1 - probs)
            # shape (S, N)

        else:
            # Multiclass
            # logits shape = (S, N, C)
            _, N, C = logits.shape
            # normalize to log‐probabilities
            logp_all = logits - logsumexp(logits, axis=-1, keepdims=True)
            # pick out log-prob of true class for each sample/data
            logp = logp_all[:, np.arange(N), y]  # shape (S, N)

        # now for either binary or multiclass, logp.shape == (S, N)
        lppd_i = logsumexp(logp, axis=0) - np.log(S)
        avg_nll = -np.mean(lppd_i)
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
    Compute avg negative log‐likelihood and a secondary metric.

    Args:
      model:     NumPyro model function that emits either
                   - sites "mu","sigma" (regression), or
                   - site "logits"         (classification)
      inference: "nuts" or "svi"
      seed:      RNG seed
      X:         array, shape (N, ...)
      y:         array, shape (N,)
      mcmc:      MCMC object (if inference="nuts")
      guide:     guide function (if inference="svi")
      params:    guide params (if inference="svi")
      num_samples: number of SVI draws
      mode:      one of {"regression","binary","multiclass"}

    Returns:
      (avg_nll, rmse)       if regression
      (avg_nll, perplexity) if classification
    """
    # choose which sites to sample
    return_sites = ("mu", "sigma") if mode == "regression" else ("logits",)

    # gather predictive samples
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

    # compute metrics
    return compute_nll_and_metric(preds, y, mode=mode)
