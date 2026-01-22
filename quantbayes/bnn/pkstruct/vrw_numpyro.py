from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.random as jr
import numpy as np

import numpyro
import numpyro.distributions as dist
import numpyro.handlers as handlers
from numpyro.infer import MCMC, NUTS

from numpyro.infer.reparam import CircularReparam


from quantbayes.pkstruct.toy.vrw import vrw_r, stephens_logpdf_r, StephensConfig
from quantbayes.pkstruct.utils.stats import log_scaled_beta_pdf


@dataclass(frozen=True)
class VRWNUTSConfig:
    N: int = 5
    mu: float = 0.0
    kappa: float = 10.0
    alpha: float = 10.0
    beta: float = 10.0
    num_warmup: int = 1000
    num_samples: int = 1000
    use_reference: bool = True
    use_circular_reparam: bool = True  # NEW: turn off only if debugging


def pk_vrw_numpyro_model(cfg: VRWNUTSConfig):
    """
    Paper-faithful VRW PK model with best-practice circular reparameterization for HMC/NUTS.
    """
    with handlers.reparam(config={"theta": CircularReparam()}):
        theta = numpyro.sample(
            "theta",
            dist.VonMises(cfg.mu, cfg.kappa).expand([cfg.N]).to_event(1),
        )

    r = vrw_r(theta)
    numpyro.deterministic("r", r)

    log_e = log_scaled_beta_pdf(r, cfg.alpha, cfg.beta, cfg.N)
    if cfg.use_reference:
        log_ref = stephens_logpdf_r(r, cfg=StephensConfig(kappa=cfg.kappa, N=cfg.N))
    else:
        log_ref = 0.0

    numpyro.factor("pk_ratio", log_e - log_ref)


def run_nuts_vrw_pk(key: jax.Array, cfg: VRWNUTSConfig) -> dict[str, np.ndarray]:
    """
    Run NUTS on the VRW PK model and return samples as numpy arrays.
    """
    kernel = NUTS(lambda: pk_vrw_numpyro_model(cfg))
    mcmc = MCMC(kernel, num_warmup=cfg.num_warmup, num_samples=cfg.num_samples)
    mcmc.run(key)
    samples = mcmc.get_samples()
    return {k: np.array(v) for k, v in samples.items()}
