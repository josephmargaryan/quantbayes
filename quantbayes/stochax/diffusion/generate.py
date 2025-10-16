# quantbayes/stochax/diffusion/generate.py
from __future__ import annotations
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.diffusion.sde import single_sample_fn
from quantbayes.stochax.diffusion.samplers.edm_heun import sample_edm_heun
from quantbayes.stochax.diffusion.samplers.dpm_solver_pp import (
    sample_dpmpp_2m,
    sample_dpmpp_3m,
)


def generate(  # PF-ODE sampler (unchanged)
    model,
    int_beta_fn,
    sample_shape,
    dt0,
    t1,
    sample_key,
    num_samples: int,
    data_stats: dict | None = None,
):
    model = eqx.tree_inference(model, value=True)
    keys = jr.split(sample_key, num_samples)

    def sample_fn(k):
        return single_sample_fn(model, int_beta_fn, sample_shape, dt0, t1, k)

    samples = jax.vmap(sample_fn)(keys)
    if data_stats is not None:
        mu = data_stats.get("mean", 0.0)
        std = data_stats.get("std", 1.0)
        lo = data_stats.get("min", None)
        hi = data_stats.get("max", None)
        samples = samples * std + mu
        if lo is not None and hi is not None:
            samples = jnp.clip(samples, lo, hi)
    return samples


def _normalize_name(s: str) -> str:
    return s.lower().replace("-", "").replace("_", "").strip()


def generate_with_sampler(
    denoise_fn,  # callable(log_sigma, x) -> D  (EDM head)
    sampler: str,
    sample_shape: tuple,
    *,
    key,
    num_samples: int,
    sampler_kwargs: dict | None = None,
    data_stats: dict | None = None,
):
    """
    Unified generator wrapper around multiple samplers.

    Supported samplers:
      - "edm_heun"  (aliases: "heun", "edmheun")
      - "dpmpp_2m"  (aliases: "dpmpp2m", "dpmsolverpp2m")
      - "dpmpp_3m"  (aliases: "dpmpp3m", "dpmsolverpp3m")
      - "unipc"     (2nd order, log-sigma PC)
      - "ipndm"     (AB2 multistep, log-sigma)
      - "ipndm4"    (AB4 multistep, log-sigma)   [NEW]
      - "dpmv3"     (single-step order-2/3)      [NEW]
    """
    sampler_kwargs = dict(sampler_kwargs or {})
    s = _normalize_name(sampler)

    if s in ("edmheun", "heun"):

        def fn(k):
            return sample_edm_heun(denoise_fn, sample_shape, key=k, **sampler_kwargs)

    elif s in ("dpmpp2m", "dpmsolverpp2m"):

        def fn(k):
            return sample_dpmpp_2m(denoise_fn, sample_shape, key=k, **sampler_kwargs)

    elif s in ("dpmpp3m", "dpmsolverpp3m"):

        def fn(k):
            return sample_dpmpp_3m(denoise_fn, sample_shape, key=k, **sampler_kwargs)

    elif s in ("unipc", "unifiedpc"):
        from quantbayes.stochax.diffusion.samplers.unipc import sample_unipc

        def fn(k):
            return sample_unipc(denoise_fn, sample_shape, key=k, **sampler_kwargs)

    elif s == "ipndm":
        from quantbayes.stochax.diffusion.samplers.ipndm import sample_ipndm

        def fn(k):
            return sample_ipndm(denoise_fn, sample_shape, key=k, **sampler_kwargs)

    elif s in ("ipndm4", "ipndm_4"):
        from quantbayes.stochax.diffusion.samplers.ipndm import sample_ipndm4

        def fn(k):
            return sample_ipndm4(denoise_fn, sample_shape, key=k, **sampler_kwargs)

    elif s in ("dpmv3", "dpmsolverv3"):
        from quantbayes.stochax.diffusion.samplers.dpm_solver_v3 import sample_dpmv3

        def fn(k):
            return sample_dpmv3(denoise_fn, sample_shape, key=k, **sampler_kwargs)

    else:
        raise ValueError(
            f"Unknown sampler '{sampler}'. "
            "Valid: edm_heun, dpmpp_2m, dpmpp_3m, unipc, ipndm, ipndm4, dpmv3."
        )

    keys = jr.split(key, num_samples)
    samples = jax.vmap(fn)(keys)

    if data_stats is not None:
        mu = data_stats.get("mean", 0.0)
        std = data_stats.get("std", 1.0)
        lo = data_stats.get("min", None)
        hi = data_stats.get("max", None)
        samples = samples * std + mu
        if lo is not None and hi is not None:
            samples = jnp.clip(samples, lo, hi)

    return samples
