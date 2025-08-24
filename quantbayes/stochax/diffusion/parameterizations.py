# quantbayes/stochax/diffusion/parameterizations.py
from __future__ import annotations
import jax.numpy as jnp

EPS = 1e-12


def vp_alpha_sigma(int_beta_fn, t):
    ib = int_beta_fn(t)
    alpha = jnp.exp(-0.5 * ib)
    sigma = jnp.sqrt(jnp.maximum(1.0 - jnp.exp(-ib), EPS))
    return alpha, sigma


# --- conversions (x = alpha*x0 + sigma*eps) ---
def eps_to_score(eps, sigma):
    return -eps / jnp.maximum(sigma, EPS)


def score_to_eps(score, sigma):
    return -sigma * score


def eps_to_x0(eps, x, alpha, sigma):
    return (x - sigma * eps) / jnp.maximum(alpha, EPS)


def x0_to_eps(x0, x, alpha, sigma):
    return (x - alpha * x0) / jnp.maximum(sigma, EPS)


# v-parameterization helpers
def eps_x0_to_v(eps, x0, alpha, sigma):
    return alpha * eps - sigma * x0


def v_to_eps(v, x, alpha, sigma):
    # given x = alpha*x0 + sigma*eps
    denom = jnp.maximum(alpha**2 + sigma**2, EPS)
    return (sigma * x + alpha * v) / denom


# EDM denoiser adapters (D is raw net output)
def edm_denoise_to_x0(x, D, sigma, sigma_data=0.5):
    denom = jnp.sqrt(sigma**2 + sigma_data**2)
    c_skip = (sigma_data**2) / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / denom
    return c_skip * x + c_out * D
