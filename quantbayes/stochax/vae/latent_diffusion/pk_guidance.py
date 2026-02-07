# quantbayes/stochax/vae/latent_diffusion/pk_guidance.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from quantbayes.stochax.diffusion.edm import edm_precond_scalars
from quantbayes.stochax.diffusion.parameterizations import edm_denoise_to_x0

# Reuse your 1D DSM score-net from the diffusion PK module you added earlier
from quantbayes.stochax.diffusion.pk.reference_score import (
    ScoreNet1D,
    ScoreNet1DConfig,
    train_or_load_score_net_dsm,
)

from .coarse import ink_fraction_and_grad_01


def _inference_copy(model):
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


@dataclass(frozen=True)
class DecodedInkPKConfig:
    # Ink observable
    ink_thr: float = 0.35
    ink_temp: float = 0.08

    # PK guidance knobs
    guide_strength: float = 8.0  # λ
    guide_sigma_max: float = 1.5  # 𝟙[σ <= ...]
    max_guide_norm: float = 50.0  # clip on ||grad_z||

    # sigma weight exponent
    sigma_data: float = 0.5
    w_gamma: float = 0.5

    # numerical
    eps: float = 1e-12


@dataclass(frozen=True)
class InkEvidence:
    # evidence is Gaussian in standardized space z_d = (d-mean_d)/std_d
    mean_d: float
    std_d: float
    mu_z: float
    tau_z: float


def compute_ink_evidence_from_real_data(
    x_real_01: jnp.ndarray,
    *,
    digit_labels: jnp.ndarray,
    target_digit: int,
    fat_quantile: float = 0.80,
    sharpen: float = 0.60,
    ink_thr: float = 0.35,
    ink_temp: float = 0.08,
) -> Tuple[float, float]:
    """
    Returns (mu_d, tau_d) in d-space based on class-k real images.
    """
    mask = digit_labels == int(target_digit)
    xk = x_real_01[mask]
    xk = xk[:5000] if xk.shape[0] > 5000 else xk

    d, _ = ink_fraction_and_grad_01(xk, thr=ink_thr, temp=ink_temp)
    mu_d = float(jnp.quantile(d, fat_quantile))
    tau_d_data = float(jnp.std(d) + 1e-6)
    tau_d = float(max(1e-4, tau_d_data * sharpen))
    return mu_d, tau_d


def train_reference_score_for_decoded_ink(
    *,
    vae,
    decode_logits_fn: Callable[[jnp.ndarray], jnp.ndarray],  # z(B,D)->logits(B,1,H,W)
    z_samples: jnp.ndarray,  # (N,D) from latent prior sampling
    score_path: Path,
    cfg: ScoreNet1DConfig,
    ink_thr: float,
    ink_temp: float,
) -> Tuple[ScoreNet1D, float, float]:
    """
    Build π(d) reference by decoding prior samples -> d -> standardize -> DSM score net s_pi(z_d).
    Returns (score_net, mean_d, std_d).
    """
    logits = decode_logits_fn(z_samples)  # (N,1,H,W)
    x01 = jax.nn.sigmoid(logits)
    d, _ = ink_fraction_and_grad_01(x01, thr=ink_thr, temp=ink_temp)

    mean_d = float(jnp.mean(d))
    std_d = float(jnp.std(d) + 1e-6)
    z_ref = ((d - mean_d) / std_d).reshape(-1, 1)

    score_net = train_or_load_score_net_dsm(z_ref, Path(score_path), cfg=cfg)
    return score_net, mean_d, std_d


class DecodedInkPKGuidance:
    """
    Computes delta in x0(z) space:
      x0_z <- x0_z + sigma^2 * guidance
    where guidance is proportional to (score_p(d)-score_pi(d)) * ∇_z d(decode(x0_z)).
    """

    def __init__(
        self,
        *,
        vae_decoder,  # module with .__call__(z)->logits or callable
        evidence: InkEvidence,
        ref_score_net: ScoreNet1D,
        cfg: DecodedInkPKConfig,
        mode: str = "pk",  # "pk" or "evidence"
    ):
        self.decoder = _inference_copy(vae_decoder)
        self.evidence = evidence
        self.ref_score = ref_score_net
        self.cfg = cfg
        self.mode = mode.lower().strip()
        if self.mode not in ("pk", "evidence"):
            raise ValueError("mode must be 'pk' or 'evidence'.")

    def _decode_logits(self, z: jnp.ndarray) -> jnp.ndarray:
        # decoder in your VAEs expects z(B,D) and returns logits(B,1,H,W)
        return self.decoder(z, rng=None, train=False)

    def _grad_z_of_d(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns:
          d: (B,)
          grad_z: (B,D)
        """
        logits = self._decode_logits(z)  # (B,1,H,W)
        x01 = jax.nn.sigmoid(logits)
        d, grad_x = ink_fraction_and_grad_01(
            x01, thr=self.cfg.ink_thr, temp=self.cfg.ink_temp
        )

        # d is function of x01; x01 = sigmoid(logits)
        # grad wrt logits: grad_x * sigmoid'(logits)
        sigp = x01 * (1.0 - x01)
        grad_logits = grad_x * sigp  # (B,1,H,W)

        # Per-sample VJP through decoder
        def one(z_i, g_i):
            def f(zz):
                # returns (1,H,W) with channel dim
                return self._decode_logits(zz[None, :])[0]

            _, vjp = jax.vjp(f, z_i)
            gz = vjp(g_i)[0]
            return gz

        grad_z = jax.vmap(one)(z, grad_logits)
        return d, grad_z

    def delta_x0(self, x0_z: jnp.ndarray, *, sigma: jnp.ndarray) -> jnp.ndarray:
        """
        x0_z: (B,D)
        sigma: scalar
        """
        sigma = jnp.asarray(sigma).reshape(())
        do_guide = (sigma <= float(self.cfg.guide_sigma_max)).astype(x0_z.dtype)

        # early exit (keeps it cheap)
        if float(self.cfg.guide_strength) == 0.0:
            return jnp.zeros_like(x0_z)

        d, grad_z = self._grad_z_of_d(x0_z)  # (B,), (B,D)

        # standardize d -> z_d
        mean_d = float(self.evidence.mean_d)
        std_d = float(self.evidence.std_d)
        z_d = (d - mean_d) / (std_d + self.cfg.eps)

        # evidence score in z_d space
        mu_z = float(self.evidence.mu_z)
        tau_z = float(self.evidence.tau_z)
        s_p = -(z_d - mu_z) / (tau_z**2 + self.cfg.eps)

        if self.mode == "evidence":
            pk_score_z = s_p
        else:
            s_pi = self.ref_score(z_d)  # (B,)
            pk_score_z = s_p - s_pi

        # to d-space
        pk_score_d = pk_score_z / (std_d + self.cfg.eps)  # (B,)
        g = pk_score_d[:, None] * grad_z  # (B,D)

        # clip per sample
        n = jnp.sqrt(jnp.sum(g * g, axis=-1, keepdims=True) + self.cfg.eps)
        clip = jnp.minimum(1.0, float(self.cfg.max_guide_norm) / n)
        g = g * clip

        # sigma weight
        base = (float(self.cfg.sigma_data) ** 2) / (
            sigma**2 + float(self.cfg.sigma_data) ** 2
        )
        w_sigma = jnp.power(base, float(self.cfg.w_gamma)).astype(x0_z.dtype)

        scale = float(self.cfg.guide_strength) * do_guide * w_sigma

        # x0 guidance uses sigma^2
        return (sigma**2) * scale * g


def wrap_denoise_fn_with_x0_guidance(
    base_denoise_fn: Callable[
        [jnp.ndarray, jnp.ndarray], jnp.ndarray
    ],  # (log_sigma, z)->D
    *,
    sigma_data: float,
    guidance: DecodedInkPKGuidance,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    EDM x0-guidance wrapper in latent space:
      D' = D + delta_x0 / c_out
    """
    sd = float(sigma_data)

    def guided(log_sigma: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        sigma = jnp.exp(log_sigma)
        D = base_denoise_fn(log_sigma, z)

        x0 = edm_denoise_to_x0(z, D, sigma, sigma_data=sd)
        delta = guidance.delta_x0(x0, sigma=sigma)

        _, _, c_out = edm_precond_scalars(sigma, sd)
        c_out = jnp.maximum(c_out, 1e-12)

        return D + (delta / c_out)

    return guided
