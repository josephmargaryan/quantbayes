# quantbayes/stochax/diffusion/pk/sampling.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.diffusion.edm import edm_precond_scalars
from quantbayes.stochax.diffusion.parameterizations import edm_denoise_to_x0
from quantbayes.stochax.diffusion.schedules.karras import get_sigmas_karras
from .guidance import PKMode
from .observables import CoarseObservable


def _inference_copy(model):
    """
    Return a copy of `model` with dropout/etc. set to inference mode.
    Compatible with Equinox versions where inference_mode returns either:
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


def make_preconditioned_edm_denoise_fn(
    ema_model,
    *,
    sigma_data: float,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Build an EDM denoise_fn(log_sigma, x)->D that matches your EDM training:
      model sees x_in = c_in * x
    """
    ema_eval = _inference_copy(ema_model)
    sd = float(sigma_data)

    def denoise_fn(log_sigma: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        sigma = jnp.exp(log_sigma)
        c_in, _, _ = edm_precond_scalars(sigma, sd)
        return ema_eval(log_sigma, x * c_in, key=None, train=False)

    return denoise_fn


@dataclass(frozen=True)
class EDMHeunConfig:
    steps: int = 40
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    sigma_data: float = 0.5


def make_edm_heun_sampler(
    denoise_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    *,
    sample_shape: tuple[int, ...],
    cfg: EDMHeunConfig,
) -> Callable[[jr.PRNGKey, int], jnp.ndarray]:
    """
    JIT-friendly batched EDM-Heun sampler (matches your notebook logic),
    but takes any EDM denoise_fn(log_sigma,x)->D (optionally guided).
    """

    sigmas = get_sigmas_karras(
        cfg.steps,
        sigma_min=cfg.sigma_min,
        sigma_max=cfg.sigma_max,
        rho=cfg.rho,
        include_zero=True,
    )
    sigma_data = float(cfg.sigma_data)

    def _x0_and_v(x_state: jnp.ndarray, sigma: jnp.ndarray):
        log_sigma = jnp.log(jnp.maximum(sigma, 1e-12))
        D = denoise_fn(log_sigma, x_state)
        x0 = edm_denoise_to_x0(x_state, D, sigma, sigma_data=sigma_data)
        v = (x_state - x0) / jnp.maximum(sigma, 1e-12)
        return x0, v

    @eqx.filter_jit
    def sample(key: jr.PRNGKey, num_samples: int) -> jnp.ndarray:
        x = jr.normal(key, (num_samples, *sample_shape)) * sigmas[0]
        last = sigmas.shape[0] - 2  # final transition is to appended sigma=0

        def step_fn(i, x_state):
            s = sigmas[i]
            sn = sigmas[i + 1]
            x0, v = _x0_and_v(x_state, s)

            def do_final(_):
                return x0

            def do_heun(_):
                x_e = x_state + (sn - s) * v  # Euler
                x0n, vn = _x0_and_v(x_e, sn)
                return x_state + 0.5 * (sn - s) * (v + vn)  # Heun

            return jax.lax.cond(i == last, do_final, do_heun, operand=None)

        x = jax.lax.fori_loop(0, sigmas.shape[0] - 1, step_fn, x)
        return x

    return sample


def make_edm_heun_gaussian_pk_sampler(
    denoise_fn: Callable[
        [jnp.ndarray, jnp.ndarray], jnp.ndarray
    ],  # (log_sigma, x) -> D
    *,
    observable: CoarseObservable,
    score_net_z: Callable[[jnp.ndarray], jnp.ndarray],  # s_pi(z) (B,)
    sample_shape: tuple[int, ...],  # e.g. (1,28,28)
    cfg: EDMHeunConfig,
    num_samples: int,
    mode: PKMode,  # "none" | "evidence" | "pk"
    eps: float = 1e-12,
) -> Callable:
    """
    Parametric EDM-Heun sampler for the "perfect experiment":

    Inputs (runtime scalars; pass as jnp.float32 arrays to avoid recompiles):
      mean_d, std_d:    prior pushforward stats of d
      mu_z, tau_z:      evidence Gaussian in z-space
      guide_strength:   λ
      guide_sigma_max:  apply guidance only when σ <= guide_sigma_max
      max_guide_norm:   per-sample norm clip
      w_gamma:          σ-weight exponent γ in:
                          w(σ) = (σ_data^2 / (σ^2 + σ_data^2))^γ

    Returns:
      x_final:   (num_samples, *sample_shape)
      clip_avg:  average clip rate over guided evaluations (like your notebook)
    """
    if mode not in ("none", "evidence", "pk"):
        raise ValueError(f"mode must be 'none'|'evidence'|'pk'. Got {mode!r}")

    sigmas = get_sigmas_karras(
        cfg.steps, cfg.sigma_min, cfg.sigma_max, rho=cfg.rho, include_zero=True
    )
    sigma_data = float(cfg.sigma_data)

    def score_p_z(z: jnp.ndarray, mu_z: jnp.ndarray, tau_z: jnp.ndarray) -> jnp.ndarray:
        # score of N(mu_z, tau_z^2) in z-space
        return -(z - mu_z) / (tau_z**2 + eps)

    @eqx.filter_jit
    def sample(
        key: jr.PRNGKey,
        mean_d: jnp.ndarray,
        std_d: jnp.ndarray,
        mu_z: jnp.ndarray,
        tau_z: jnp.ndarray,
        guide_strength: jnp.ndarray,
        guide_sigma_max: jnp.ndarray,
        max_guide_norm: jnp.ndarray,
        w_gamma: jnp.ndarray,
    ):
        # normalize dtypes (keeps compile stable)
        mean_d = jnp.asarray(mean_d, dtype=jnp.float32)
        std_d = jnp.asarray(std_d, dtype=jnp.float32)
        mu_z = jnp.asarray(mu_z, dtype=jnp.float32)
        tau_z = jnp.asarray(tau_z, dtype=jnp.float32)
        guide_strength = jnp.asarray(guide_strength, dtype=jnp.float32)
        guide_sigma_max = jnp.asarray(guide_sigma_max, dtype=jnp.float32)
        max_guide_norm = jnp.asarray(max_guide_norm, dtype=jnp.float32)
        w_gamma = jnp.asarray(w_gamma, dtype=jnp.float32)

        x = jr.normal(key, (num_samples, *sample_shape)) * sigmas[0]

        def x0_and_v(x_state: jnp.ndarray, sigma: jnp.ndarray):
            log_sigma = jnp.log(jnp.maximum(sigma, eps))
            D = denoise_fn(log_sigma, x_state)
            x0 = edm_denoise_to_x0(x_state, D, sigma, sigma_data=sigma_data)

            clip_rate = jnp.asarray(0.0, dtype=jnp.float32)
            guide_ct = jnp.asarray(0.0, dtype=jnp.float32)

            if mode != "none":
                do_guide = sigma <= guide_sigma_max
                guide_ct = do_guide.astype(jnp.float32)

                # observable and grad on clipped x0
                x0c = jnp.clip(x0, -1.0, 1.0)
                d, grad_d = observable.value_and_grad(x0c)  # d:(B,) grad:(B,...)

                z = (d - mean_d) / (std_d + eps)
                s_p = score_p_z(z, mu_z, tau_z)

                if mode == "evidence":
                    pk_score_z = s_p
                else:
                    s_pi = score_net_z(z)  # (B,)
                    pk_score_z = s_p - s_pi

                pk_score_d = pk_score_z / (std_d + eps)
                pk_score_d = pk_score_d.reshape(
                    (pk_score_d.shape[0],) + (1,) * (x0.ndim - 1)
                )

                g = pk_score_d * grad_d  # (B,...)

                # clip
                axes = tuple(range(1, g.ndim))
                g_norm = jnp.sqrt(jnp.sum(g * g, axis=axes, keepdims=True) + eps)

                # average fraction clipped in this eval, but only when guiding is active
                clip_rate = (
                    jnp.mean((g_norm > max_guide_norm).astype(jnp.float32)) * guide_ct
                )

                clip = jnp.minimum(1.0, max_guide_norm / g_norm)
                g = g * clip

                # σ-weight exponent γ (middle ground dial)
                base = (sigma_data**2) / (sigma**2 + sigma_data**2)  # in (0,1]
                w_sigma = jnp.power(base, w_gamma)  # γ=0 -> 1 ; γ=1 -> base
                g = g * w_sigma

                # λ + indicator
                g = g * (guide_strength * do_guide.astype(x0.dtype))

                # x0 guidance
                x0 = x0 + (sigma**2) * g

            v = (x_state - x0) / jnp.maximum(sigma, eps)
            return x0, v, clip_rate, guide_ct

        last = sigmas.shape[0] - 2

        def step_carry(i, carry):
            x_state, clip_sum, guide_sum = carry
            s = sigmas[i]
            sn = sigmas[i + 1]

            x0, v, cr, gc = x0_and_v(x_state, s)

            def do_final(_):
                return (x0, clip_sum + cr, guide_sum + gc)

            def do_heun(_):
                x_e = x_state + (sn - s) * v
                x0n, vn, crn, gcn = x0_and_v(x_e, sn)
                x_next = x_state + 0.5 * (sn - s) * (v + vn)
                return (x_next, clip_sum + cr + crn, guide_sum + gc + gcn)

            return jax.lax.cond(i == last, do_final, do_heun, operand=None)

        x_fin, clip_sum, guide_sum = jax.lax.fori_loop(
            0,
            sigmas.shape[0] - 1,
            step_carry,
            (x, jnp.asarray(0.0, jnp.float32), jnp.asarray(0.0, jnp.float32)),
        )

        x_fin, clip_sum, guide_sum = jax.lax.fori_loop(
            0,
            sigmas.shape[0] - 1,
            step_carry,
            (x, jnp.asarray(0.0, jnp.float32), jnp.asarray(0.0, jnp.float32)),
        )

        clip_avg = clip_sum / (guide_sum + eps)
        return x_fin, clip_avg

    return sample
