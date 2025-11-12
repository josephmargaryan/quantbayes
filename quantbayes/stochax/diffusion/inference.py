# quantbayes/stochax/diffusion/inference.py
from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .generate import generate_with_sampler
from .samplers.dpm_solver_pp import sample_dpmpp_3m


def _inference_copy(model):
    """Return a copy of `model` with dropout/etc. set to inference mode.

    Works with Equinox versions where `inference_mode` returns either:
      - a model copy, or
      - a context manager that yields a model copy.
    """
    maybe = eqx.nn.inference_mode(model)
    enter = getattr(maybe, "__enter__", None)
    exit_ = getattr(maybe, "__exit__", None)
    if callable(enter) and callable(exit_):
        try:
            m = enter()
            return m
        finally:
            exit_(None, None, None)
    return maybe


def sample_edm(
    ema_model,
    num_samples: int,
    sample_shape: tuple,
    key: jr.PRNGKey,
    *,
    steps: int = 30,
    sigma_min: float = 0.002,
    sigma_max: float = 1.0,
    sigma_data: float = 0.5,
    rho: float = 7.0,
    sampler: str = "dpmpp_3m",
):
    """Unconditional EDM sampling; disables dropout, etc., during inference."""
    ema_eval = _inference_copy(ema_model)

    def denoise_fn(log_sigma, x):
        return ema_eval(log_sigma, x, key=None, train=False)

    return generate_with_sampler(
        denoise_fn,
        sampler=sampler,
        sample_shape=sample_shape,
        key=key,
        num_samples=num_samples,
        sampler_kwargs=dict(
            steps=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            rho=rho,
        ),
    )


def sample_edm_conditional(
    ema_model,
    label: jnp.ndarray | int,
    cfg_scale: float,
    num_samples: int,
    sample_shape: tuple,
    key: jr.PRNGKey,
    *,
    steps: int = 30,
    sigma_min: float = 0.002,
    sigma_max: float = 1.0,
    sigma_data: float = 0.5,
    rho: float = 7.0,
    sampler: str = "dpmpp_3m",
):
    """Conditional EDM sampling with CFG; `ema_model` must be a DiTWrapper.

    Important: each denoising call handles a **single** sample, so we vmap over
    (keys, labels) and build a per-sample denoise_fn that captures a scalar label.
    """
    if sampler != "dpmpp_3m":
        raise ValueError("Only 'dpmpp_3m' sampler is supported in conditional path.")

    ema_eval = _inference_copy(ema_model)

    # Normalize labels to shape (num_samples,)
    label = jnp.asarray(label, dtype=jnp.int32)
    if label.ndim == 0:
        label = jnp.full((num_samples,), int(label), dtype=jnp.int32)
    elif label.shape[0] != num_samples:
        label = jnp.broadcast_to(label, (num_samples,))

    keys = jr.split(key, num_samples)

    def one_sample(k, lbl_scalar):
        # Build a per-sample denoise function that captures a SCALAR label.
        def denoise_single(log_sigma, x):
            return ema_eval(
                log_sigma,
                x,
                key=None,
                train=False,
                label=lbl_scalar,
                cfg_scale=cfg_scale,
            )

        return sample_dpmpp_3m(
            denoise_single,
            sample_shape,
            key=k,
            steps=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            rho=rho,
        )

    # Map over (keys, labels) so each sample uses its own scalar label.
    samples = jax.vmap(one_sample)(keys, label)
    return samples
