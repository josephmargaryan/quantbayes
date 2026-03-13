import warnings

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np

from quantbayes import bnn


def count_params_mcmc(samples):
    """
    Count the total number of parameters from an MCMC object.

    This function assumes that mcmc.get_samples() returns a dictionary
    where each value is an array with shape (num_samples, ...), and counts
    the total number of parameters per sample.
    """
    flat_samples, _ = jax.tree_util.tree_flatten(samples)
    total_params = 0
    for arr in flat_samples:
        # Ignore the first dimension (num_samples) and count the rest
        param_count = int(jnp.prod(jnp.array(arr.shape[1:])))
        total_params += param_count
    return total_params


def evaluate_mcmc(model) -> dict:
    """
    Evaluate an MCMC model by computing common diagnostics and return a summary dictionary.

    The returned dictionary includes WAIC, LOO, average Rhat, average ESS, and total parameter count.
    Values are formatted to two decimal places where appropriate.

    :param model: A model instance (subclass of Module) that has run inference.
    :raises ValueError: If the model does not have inference results.
    :return: A dictionary containing the diagnostics.
    """
    if not hasattr(model, "inference") or model.inference is None:
        raise ValueError(
            "The model does not have inference results. Please run inference first!"
        )

    idata = az.from_numpyro(model.inference)

    try:
        waic_result = az.waic(idata)
    except Exception as e:
        warnings.warn("WAIC computation failed: " + str(e))
        waic_result = None

    try:
        loo_result = az.loo(idata)
    except Exception as e:
        warnings.warn("LOO computation failed: " + str(e))
        loo_result = None

    try:
        rhat_result = az.rhat(idata)
        avg_rhat = float(rhat_result.to_array().mean())
    except Exception as e:
        warnings.warn("Rhat computation failed: " + str(e))
        avg_rhat = np.nan

    try:
        ess_result = az.ess(idata)
        avg_ess = float(ess_result.to_array().mean())
    except Exception as e:
        warnings.warn("ESS computation failed: " + str(e))
        avg_ess = np.nan

    # For WAIC standard error
    if waic_result is not None:
        if hasattr(waic_result, "waic_se"):
            waic_se = waic_result.waic_se
        else:
            waic_i = jnp.array(waic_result.waic_i)
            waic_se = float(waic_i.std() / jnp.sqrt(waic_i.shape[0]))
    else:
        waic_se = np.nan

    # For LOO standard error
    if loo_result is not None:
        if hasattr(loo_result, "loo_se"):
            loo_se = loo_result.loo_se
        else:
            loo_i = jnp.array(loo_result.loo_i)
            loo_se = float(loo_i.std() / jnp.sqrt(loo_i.shape[0]))
    else:
        loo_se = np.nan

    # Count total parameters from the MCMC samples.
    samples = model.inference.get_samples()
    total_params = count_params_mcmc(samples)

    summary = {
        "elpd_waic": (
            f"{waic_result.elpd_waic:.2f}" if waic_result is not None else "NaN"
        ),
        "p_waic": f"{waic_result.p_waic:.2f}" if waic_result is not None else "NaN",
        "waic_se": f"{waic_se:.2f}",
        "elpd_loo": f"{loo_result.elpd_loo:.2f}" if loo_result is not None else "NaN",
        "p_loo": f"{loo_result.p_loo:.2f}" if loo_result is not None else "NaN",
        "loo_se": f"{loo_se:.2f}",
        "avg_rhat": f"{avg_rhat:.2f}" if not np.isnan(avg_rhat) else "NaN",
        "avg_ess": f"{avg_ess:.2f}",
        "total_params": total_params,
    }
    return summary
