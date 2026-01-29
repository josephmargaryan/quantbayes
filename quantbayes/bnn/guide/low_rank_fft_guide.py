# quantbayes/bnn/layers/custom_guides.py
# -----------------------------------------------------------------------------
# Research-grade NumPyro guides for spectral layers
#   - FFT/circulant 1D/2D (stationary & adaptive)
#   - SVD-based Dense/Conv2d (stationary & adaptive)
#   - Mean-field fallbacks
#
# All guides are memory-safe by default (low-rank reparameterization).
# Set mode="dense" if you want a true MVN for small problems.
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Tuple, Optional
import math

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoGuide

__all__ = [
    # FFT (you already had these; kept and hardened)
    "LowRankFFTGuide",
    "LowRankAdaptiveFFTGuide",
    "LowRankFFTGuide2d",
    "LowRankAdaptiveFFTGuide2d",
    "MeanFieldFFTGuide",
    "MeanFieldFFTGuide2d",
    # NEW: SVD guides for SpectralDense / SpectralConv2d
    "LowRankSVDGuide",
    "LowRankAdaptiveSVDGuide",
    "MeanFieldSVDGuide",
]

# =============================== utils =======================================


def _softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.softplus(x)


def _rng(shape, scale=0.05):
    return scale * jax.random.normal(numpyro.prng_key(), shape)


def halfplane_to_full_multi(
    real_hp: jnp.ndarray, imag_hp: jnp.ndarray, H_pad: int, W_pad: int
):
    """RFFT2 half-plane -> full Hermitian spectrum (no amplitude loss)."""
    W_half = W_pad // 2 + 1
    hp = real_hp + 1j * imag_hp  # (..., H, W_half)
    full = jnp.zeros(hp.shape[:-1] + (W_pad,), hp.dtype)
    full = full.at[..., :W_half].set(hp)  # left half = half-plane
    # mirror right half by conjugate flip
    mirror = jnp.flip(jnp.conj(full), axis=(-2, -1))
    full = full.at[..., W_half:].set(mirror[..., W_half:])
    # enforce real lines
    full = full.at[..., 0, 0].set(jnp.real(full[..., 0, 0]))
    if H_pad % 2 == 0:
        full = full.at[..., H_pad // 2, :].set(jnp.real(full[..., H_pad // 2, :]))
    if W_pad % 2 == 0:
        full = full.at[..., :, W_pad // 2].set(jnp.real(full[..., :, W_pad // 2]))
    return full.real, full.imag


def _check_rank(rank: int, dim: int) -> int:
    return int(max(1, min(rank, dim)))


# ====================== low-rank reparameterized joint =======================


class _LowRankJointSampler:
    """
    Sample z ~ N(loc, V V^T + diag(d)) either by:
      - mode="reparam":  z = loc + V @ eps1 + sqrt_diag(d) * eps2
                         eps1 ~ N(0, I_r), eps2 ~ N(0, I_D). No dense cov.
      - mode="dense":    exact MultivariateNormal (good for small D).
    Parameters are registered as learnable guide params.
    """

    def __init__(
        self, joint_name: str, dim: int, rank: int, jitter: float, mode: str = "reparam"
    ):
        assert mode in ("reparam", "dense")
        self.joint = joint_name
        self.dim = int(dim)
        self.rank = _check_rank(rank, dim)
        self.jitter = float(jitter)
        self.mode = mode

    def params(self):
        loc = numpyro.param(f"{self.joint}_loc", jnp.zeros(self.dim))
        V = numpyro.param(
            f"{self.joint}_V", _rng((self.dim, self.rank)) / math.sqrt(self.dim)
        )
        d_raw = numpyro.param(f"{self.joint}_d_raw", -3.0 * jnp.ones(self.dim))
        log_tau = numpyro.param(f"{self.joint}_log_tau", jnp.array(0.0))
        d = _softplus(d_raw) * jnp.exp(log_tau) + self.jitter
        return loc, V, d

    def sample(self) -> jnp.ndarray:
        loc, V, d = self.params()
        if self.mode == "dense":
            cov = V @ V.T + jnp.diag(d**2)
            return numpyro.sample(
                self.joint,
                dist.MultivariateNormal(loc, covariance_matrix=cov),
                infer={"is_auxiliary": True},
            )
        eps1 = numpyro.sample(
            f"{self.joint}_eps1",
            dist.Normal(0.0, 1.0).expand([self.rank]).to_event(1),
            infer={"is_auxiliary": True},
        )
        eps2 = numpyro.sample(
            f"{self.joint}_eps2",
            dist.Normal(0.0, 1.0).expand([self.dim]).to_event(1),
            infer={"is_auxiliary": True},
        )
        z = loc + V @ eps1 + d * eps2
        numpyro.deterministic(f"{self.joint}_z", z)
        return z

    def sample_posterior(self, rng_key, params, sample_shape=()):
        loc = params[f"{self.joint}_loc"]
        V = params[f"{self.joint}_V"]
        d = (
            _softplus(params[f"{self.joint}_d_raw"])
            * jnp.exp(params[f"{self.joint}_log_tau"])
            + self.jitter
        )
        if self.mode == "dense":
            cov = V @ V.T + jnp.diag(d**2)
            mvn = dist.MultivariateNormal(loc, covariance_matrix=cov)
            return mvn.sample(rng_key, sample_shape)
        key1, key2 = jax.random.split(rng_key)
        eps1 = jax.random.normal(key1, sample_shape + (self.rank,))
        eps2 = jax.random.normal(key2, sample_shape + (self.dim,))
        return loc + (V @ eps1[..., None]).squeeze(-1) + d * eps2


# ========================== FFT guides (1D/2D) ===============================


class LowRankFFTGuide(AutoGuide):
    """
    1D FFT half-spectrum low-rank guide:
      - binds {prefix}_real : (K,)
      - binds {prefix}_imag : (K,) with DC (and Nyquist if even) imag=0
    """

    def __init__(
        self,
        model,
        prefix: str,
        *,
        padded_dim: int,
        K: Optional[int] = None,
        rank: int = 8,
        jitter: float = 1e-4,
        mode: str = "reparam",
    ):
        super().__init__(model)
        self.prefix = prefix
        self.padded_dim = int(padded_dim)
        self.k_half = self.padded_dim // 2 + 1
        self.K = self.k_half if (K is None or K > self.k_half) else int(K)
        self.sampler = _LowRankJointSampler(
            f"{prefix}_joint", dim=2 * self.K, rank=rank, jitter=jitter, mode=mode
        )

    def __call__(self, *args, **kwargs):
        z = self.sampler.sample()
        real_hp, imag_hp = jnp.split(z, 2, axis=-1)
        imag_hp = imag_hp.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.K == self.k_half) and (self.K > 1):
            imag_hp = imag_hp.at[-1].set(0.0)
        numpyro.sample(f"{self.prefix}_real", dist.Delta(real_hp).to_event(1))
        numpyro.sample(f"{self.prefix}_imag", dist.Delta(imag_hp).to_event(1))
        return {f"{self.prefix}_real": real_hp, f"{self.prefix}_imag": imag_hp}

    def sample_posterior(self, rng_key, params, sample_shape=()):
        z = self.sampler.sample_posterior(rng_key, params, sample_shape)
        real_hp, imag_hp = jnp.split(z, 2, axis=-1)
        imag_hp = imag_hp.at[..., 0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.K == self.k_half) and (self.K > 1):
            imag_hp = imag_hp.at[..., -1].set(0.0)
        return {f"{self.prefix}_real": real_hp, f"{self.prefix}_imag": imag_hp}


class LowRankAdaptiveFFTGuide(AutoGuide):
    """
    1D FFT adaptive guide (non-stationary):
      binds {prefix}_delta_alpha : (k_half,), {prefix}_real : (k_half,), {prefix}_imag : (k_half,)
    """

    def __init__(
        self,
        model,
        prefix: str,
        *,
        padded_dim: int,
        rank: int = 8,
        jitter: float = 1e-5,
        mode: str = "reparam",
    ):
        super().__init__(model)
        self.prefix = prefix
        self.padded_dim = int(padded_dim)
        self.k_half = self.padded_dim // 2 + 1
        self.sampler = _LowRankJointSampler(
            f"{prefix}_joint", dim=3 * self.k_half, rank=rank, jitter=jitter, mode=mode
        )

    def __call__(self, *args, **kwargs):
        z = self.sampler.sample()
        real_hp, imag_hp, delta_alpha = jnp.split(
            z, [self.k_half, 2 * self.k_half], axis=-1
        )
        imag_hp = imag_hp.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            imag_hp = imag_hp.at[-1].set(0.0)
        numpyro.sample(
            f"{self.prefix}_delta_alpha", dist.Delta(delta_alpha).to_event(1)
        )
        numpyro.sample(f"{self.prefix}_real", dist.Delta(real_hp).to_event(1))
        numpyro.sample(f"{self.prefix}_imag", dist.Delta(imag_hp).to_event(1))
        return {
            f"{self.prefix}_delta_alpha": delta_alpha,
            f"{self.prefix}_real": real_hp,
            f"{self.prefix}_imag": imag_hp,
        }

    def sample_posterior(self, rng_key, params, sample_shape=()):
        z = self.sampler.sample_posterior(rng_key, params, sample_shape)
        real_hp, imag_hp, delta_alpha = jnp.split(
            z, [self.k_half, 2 * self.k_half], axis=-1
        )
        imag_hp = imag_hp.at[..., 0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            imag_hp = imag_hp.at[..., -1].set(0.0)
        return {
            f"{self.prefix}_delta_alpha": delta_alpha,
            f"{self.prefix}_real": real_hp,
            f"{self.prefix}_imag": imag_hp,
        }


class LowRankRFFTGuide1d(AutoGuide):
    """
    Variational guide for RFFTCirculant1D that parameterizes the HALF-SPECTRUM
    active coefficients wr/wi of length K, matching the model sites:
      - {prefix}_real : (K,)
      - {prefix}_imag : (K,) with DC (and Nyquist if even & using full half) forced real.

    Optionally, include a variational factor for the model's scalar
    '{prefix}_alpha_z' by setting include_alpha=True (otherwise let another guide
    like AutoNormal handle it).

    Args:
        model: NumPyro model.
        prefix: Must match the layer's `name` (default "rfft1d").
        padded_dim: FFT length used by the layer (H_pad).
        K: Number of active half-spectrum coefficients (defaults to k_half).
        rank: Low-rank dimension for the joint Gaussian over [real, imag].
        jitter: Diagonal jitter for numerical stability.
        mode: Low-rank sampler mode (e.g., "reparam").
        zero_dc_nyquist_imag: Zero imaginary parts at DC (and Nyquist if applicable).
        include_alpha: If True, add a Normal variational factor for '{prefix}_alpha_z'.
    """

    def __init__(
        self,
        model,
        prefix: str,
        *,
        padded_dim: int,
        K: Optional[int] = None,
        rank: int = 8,
        jitter: float = 1e-4,
        mode: str = "reparam",
        zero_dc_nyquist_imag: bool = True,
        include_alpha: bool = False,
    ):
        super().__init__(model)
        self.prefix = prefix
        self.padded_dim = int(padded_dim)
        self.k_half = self.padded_dim // 2 + 1
        self.K = self.k_half if (K is None or K > self.k_half) else int(K)
        self.zero_dc_nyquist_imag = bool(zero_dc_nyquist_imag)
        self.include_alpha = bool(include_alpha)

        # joint low-rank Gaussian over concatenated [real[0:K], imag[0:K]]
        self.sampler = _LowRankJointSampler(
            f"{prefix}_joint",
            dim=2 * self.K,
            rank=rank,
            jitter=jitter,
            mode=mode,
        )

        if self.include_alpha:
            # scalar Normal variational factor for alpha_z
            self._alpha_loc_name = f"{prefix}_alpha_loc"
            self._alpha_rho_name = f"{prefix}_alpha_rho"

    def __call__(self, *args, **kwargs):
        z = self.sampler.sample()  # shape (..., 2K)
        real_k, imag_k = jnp.split(z, 2, axis=-1)  # each (..., K)

        if self.zero_dc_nyquist_imag:
            imag_k = imag_k.at[..., 0].set(0.0)
            if (self.padded_dim % 2 == 0) and (self.K == self.k_half) and (self.K > 1):
                imag_k = imag_k.at[..., -1].set(0.0)

        numpyro.sample(f"{self.prefix}_real", dist.Delta(real_k).to_event(1))
        numpyro.sample(f"{self.prefix}_imag", dist.Delta(imag_k).to_event(1))

        if self.include_alpha:
            loc = numpyro.param(self._alpha_loc_name, 0.0)
            rho = numpyro.param(self._alpha_rho_name, -2.0)
            scale = jax.nn.softplus(rho)
            numpyro.sample(f"{self.prefix}_alpha_z", dist.Normal(loc, scale))

        # returning is optional; provided for convenience
        out = {
            f"{self.prefix}_real": real_k,
            f"{self.prefix}_imag": imag_k,
        }
        if self.include_alpha:
            out[f"{self.prefix}_alpha_z"] = loc
        return out

    def sample_posterior(self, rng_key, params, sample_shape=()):
        z = self.sampler.sample_posterior(rng_key, params, sample_shape)  # (..., 2K)
        real_k, imag_k = jnp.split(z, 2, axis=-1)  # each (..., K)

        if self.zero_dc_nyquist_imag:
            imag_k = imag_k.at[..., 0].set(0.0)
            if (self.padded_dim % 2 == 0) and (self.K == self.k_half) and (self.K > 1):
                imag_k = imag_k.at[..., -1].set(0.0)

        out = {
            f"{self.prefix}_real": real_k,
            f"{self.prefix}_imag": imag_k,
        }

        if self.include_alpha:
            import jax.random as jr

            loc = params[self._alpha_loc_name]
            rho = params[self._alpha_rho_name]
            scale = jax.nn.softplus(rho)
            key_alpha, _ = jr.split(rng_key)
            alpha = dist.Normal(loc, scale).sample(key_alpha, sample_shape)
            out[f"{self.prefix}_alpha_z"] = alpha

        return out


class LowRankFFTGuide2d(AutoGuide):
    """
    2D FFT half-plane low-rank guide:
      produces full-plane {prefix}_real/{prefix}_imag via Hermitian completion.
    """

    def __init__(
        self,
        model,
        prefix: str,
        *,
        C_out: int,
        C_in: int,
        H_pad: int,
        W_pad: int,
        rank: int = 8,
        jitter: float = 1e-4,
        mode: str = "reparam",
    ):
        super().__init__(model)
        self.prefix = prefix
        self.C_out, self.C_in = int(C_out), int(C_in)
        self.H_pad, self.W_pad = int(H_pad), int(W_pad)
        self.W_half = self.W_pad // 2 + 1
        M = self.C_out * self.C_in * self.H_pad * self.W_half
        self.shape_hp = (self.C_out, self.C_in, self.H_pad, self.W_half)
        self.sampler = _LowRankJointSampler(
            f"{prefix}_joint", dim=2 * M, rank=rank, jitter=jitter, mode=mode
        )

    def __call__(self, *args, **kwargs):
        z = self.sampler.sample()
        real_hp_vec, imag_hp_vec = jnp.split(z, 2, axis=-1)
        real_hp = real_hp_vec.reshape(self.shape_hp)
        imag_hp = imag_hp_vec.reshape(self.shape_hp)
        r_full, i_full = halfplane_to_full_multi(
            real_hp, imag_hp, self.H_pad, self.W_pad
        )
        numpyro.sample(f"{self.prefix}_real", dist.Delta(r_full).to_event(4))
        numpyro.sample(f"{self.prefix}_imag", dist.Delta(i_full).to_event(4))
        return {f"{self.prefix}_real": r_full, f"{self.prefix}_imag": i_full}

    def sample_posterior(self, rng_key, params, sample_shape=()):
        z = self.sampler.sample_posterior(rng_key, params, sample_shape)
        real_hp_vec, imag_hp_vec = jnp.split(z, 2, axis=-1)
        shp_hp = sample_shape + self.shape_hp
        real_hp = real_hp_vec.reshape(shp_hp)
        imag_hp = imag_hp_vec.reshape(shp_hp)
        r_full, i_full = halfplane_to_full_multi(
            real_hp, imag_hp, self.H_pad, self.W_pad
        )
        return {f"{self.prefix}_real": r_full, f"{self.prefix}_imag": i_full}


class LowRankAdaptiveFFTGuide2d(AutoGuide):
    """
    2D FFT adaptive low-rank guide:
      binds {prefix}_delta_alpha (gh,gw) and full-plane {prefix}_real/{prefix}_imag
    """

    def __init__(
        self,
        model,
        prefix: str,
        *,
        C_out: int,
        C_in: int,
        H_pad: int,
        W_pad: int,
        alpha_coarse_shape: Tuple[int, int] = (8, 8),
        rank: int = 8,
        jitter: float = 1e-4,
        mode: str = "reparam",
    ):
        super().__init__(model)
        self.prefix = prefix
        self.C_out, self.C_in = int(C_out), int(C_in)
        self.H_pad, self.W_pad = int(H_pad), int(W_pad)
        self.W_half = self.W_pad // 2 + 1
        self.gh, self.gw = map(int, alpha_coarse_shape)
        self.shape_hp = (self.C_out, self.C_in, self.H_pad, self.W_half)
        M = self.C_out * self.C_in * self.H_pad * self.W_half
        G = self.gh * self.gw
        self.sampler = _LowRankJointSampler(
            f"{prefix}_joint", dim=2 * M + G, rank=rank, jitter=jitter, mode=mode
        )

    def __call__(self, *args, **kwargs):
        z = self.sampler.sample()
        M = self.C_out * self.C_in * self.H_pad * self.W_half
        z_f, delta_flat = jnp.split(z, [2 * M], axis=-1)
        real_hp_vec, imag_hp_vec = jnp.split(z_f, 2, axis=-1)
        real_hp = real_hp_vec.reshape(self.shape_hp)
        imag_hp = imag_hp_vec.reshape(self.shape_hp)
        delta_map = delta_flat.reshape(self.gh, self.gw)
        numpyro.sample(f"{self.prefix}_delta_alpha", dist.Delta(delta_map).to_event(2))
        r_full, i_full = halfplane_to_full_multi(
            real_hp, imag_hp, self.H_pad, self.W_pad
        )
        numpyro.sample(f"{self.prefix}_real", dist.Delta(r_full).to_event(4))
        numpyro.sample(f"{self.prefix}_imag", dist.Delta(i_full).to_event(4))
        return {
            f"{self.prefix}_delta_alpha": delta_map,
            f"{self.prefix}_real": r_full,
            f"{self.prefix}_imag": i_full,
        }

    def sample_posterior(self, rng_key, params, sample_shape=()):
        M = self.C_out * self.C_in * self.H_pad * self.W_half
        z = self.sampler.sample_posterior(rng_key, params, sample_shape)
        z_f, delta_flat = jnp.split(z, [2 * M], axis=-1)
        real_hp_vec, imag_hp_vec = jnp.split(z_f, 2, axis=-1)
        shp_hp = sample_shape + self.shape_hp
        real_hp = real_hp_vec.reshape(shp_hp)
        imag_hp = imag_hp_vec.reshape(shp_hp)
        delta_map = delta_flat.reshape(sample_shape + (self.gh, self.gw))
        r_full, i_full = halfplane_to_full_multi(
            real_hp, imag_hp, self.H_pad, self.W_pad
        )
        return {
            f"{self.prefix}_delta_alpha": delta_map,
            f"{self.prefix}_real": r_full,
            f"{self.prefix}_imag": i_full,
        }


class LowRankRFFTGuide2d(AutoGuide):
    """
    2D FFT low-rank guide over the HALF-PLANE (what rfft2 uses).
    - Samples half-plane sites `{prefix}_real` / `{prefix}_imag` with shape
      (C_out, C_in, H_pad, W_half).
    - Writes optional deterministic full-plane tensors for inspection:
      `{prefix}_real_full`, `{prefix}_imag_full`.
    """

    def __init__(
        self,
        model,
        prefix: str,
        *,
        C_out: int,
        C_in: int,
        H_pad: int,
        W_pad: int,
        rank: int = 8,
        jitter: float = 1e-4,
        mode: str = "reparam",
        zero_dc_nyquist_imag: bool = True,
    ):
        super().__init__(model)
        self.prefix = prefix
        self.C_out, self.C_in = int(C_out), int(C_in)
        self.H_pad, self.W_pad = int(H_pad), int(W_pad)
        self.W_half = self.W_pad // 2 + 1
        self.shape_hp = (self.C_out, self.C_in, self.H_pad, self.W_half)
        M = self.C_out * self.C_in * self.H_pad * self.W_half
        self.sampler = _LowRankJointSampler(
            f"{prefix}_joint", dim=2 * M, rank=rank, jitter=jitter, mode=mode
        )
        self.zero_dc_nyquist_imag = bool(zero_dc_nyquist_imag)

    def __call__(self, *args, **kwargs):
        # sample a joint low-rank vector and split into real/imag halves
        z = self.sampler.sample()
        real_hp_vec, imag_hp_vec = jnp.split(z, 2, axis=-1)
        real_hp = real_hp_vec.reshape(self.shape_hp)
        imag_hp = imag_hp_vec.reshape(self.shape_hp)

        # enforce purely real DC (and Nyquist column if even width), matching layer behavior
        if self.zero_dc_nyquist_imag:
            imag_hp = imag_hp.at[..., :, 0].set(0.0)
            if (self.W_pad % 2 == 0) and (self.W_half > 1):
                imag_hp = imag_hp.at[..., :, -1].set(0.0)

        # *** IMPORTANT: sample HALF-PLANE at the sites expected by the model ***
        numpyro.sample(f"{self.prefix}_real", dist.Delta(real_hp).to_event(4))
        numpyro.sample(f"{self.prefix}_imag", dist.Delta(imag_hp).to_event(4))

        # Optional: expose the hermitian-completed full-plane for debugging/visualization
        r_full, i_full = halfplane_to_full_multi(
            real_hp, imag_hp, self.H_pad, self.W_pad
        )
        numpyro.deterministic(f"{self.prefix}_real_full", r_full)
        numpyro.deterministic(f"{self.prefix}_imag_full", i_full)

        # return mapping (optional)
        return {f"{self.prefix}_real": real_hp, f"{self.prefix}_imag": imag_hp}

    def sample_posterior(self, rng_key, params, sample_shape=()):
        z = self.sampler.sample_posterior(rng_key, params, sample_shape)
        real_hp_vec, imag_hp_vec = jnp.split(z, 2, axis=-1)
        shp_hp = sample_shape + self.shape_hp
        real_hp = real_hp_vec.reshape(shp_hp)
        imag_hp = imag_hp_vec.reshape(shp_hp)

        if self.zero_dc_nyquist_imag:
            imag_hp = imag_hp.at[..., :, 0].set(0.0)
            if (self.W_pad % 2 == 0) and (self.W_half > 1):
                imag_hp = imag_hp.at[..., :, -1].set(0.0)

        # keep half-plane at sample sites; also provide full-plane dict for convenience
        r_full, i_full = halfplane_to_full_multi(
            real_hp, imag_hp, self.H_pad, self.W_pad
        )
        return {
            f"{self.prefix}_real": real_hp,
            f"{self.prefix}_imag": imag_hp,
            f"{self.prefix}_real_full": r_full,
            f"{self.prefix}_imag_full": i_full,
        }


# =========================== SVD guides (Dense/Conv) =========================


class LowRankSVDGuide(AutoGuide):
    """
    Low-rank joint guide for SVD-based layers (stationary):
      - binds {prefix}_s : (s_dim,)
      - optionally binds {prefix}_alpha_z : () if include_alpha=True
    Use for SpectralDense / SpectralConv2d with fixed U,V.
    """

    def __init__(
        self,
        model,
        prefix: str,
        *,
        U: Optional[jnp.ndarray] = None,
        s_dim: Optional[int] = None,
        include_alpha: bool = False,
        rank: int = 8,
        jitter: float = 1e-5,
        mode: str = "reparam",
    ):
        super().__init__(model)
        assert (U is not None) or (s_dim is not None), "Provide U or s_dim."
        self.prefix = prefix
        self.s_dim = int(U.shape[1] if (U is not None) else s_dim)
        self.include_alpha = bool(include_alpha)
        dim = self.s_dim + (1 if self.include_alpha else 0)
        self.sampler = _LowRankJointSampler(
            f"{prefix}_svd_joint", dim=dim, rank=rank, jitter=jitter, mode=mode
        )

    def __call__(self, *args, **kwargs):
        z = self.sampler.sample()
        if self.include_alpha:
            s_vec, alpha_z = z[:-1], z[-1]
            numpyro.sample(f"{self.prefix}_alpha_z", dist.Delta(alpha_z))
        else:
            s_vec = z
        numpyro.sample(f"{self.prefix}_s", dist.Delta(s_vec).to_event(1))
        return {f"{self.prefix}_s": s_vec}

    def sample_posterior(self, rng_key, params, sample_shape=()):
        z = self.sampler.sample_posterior(rng_key, params, sample_shape)
        if self.include_alpha:
            s_vec, alpha_z = z[..., :-1], z[..., -1]
            return {f"{self.prefix}_s": s_vec, f"{self.prefix}_alpha_z": alpha_z}
        return {f"{self.prefix}_s": z}


class LowRankAdaptiveSVDGuide(AutoGuide):
    """
    Low-rank joint guide for adaptive SVD-based layers:
      - binds {prefix}_s            : (s_dim,)
      - binds {prefix}_{delta_name} : (s_dim,)
      - optionally binds {prefix}_alpha_z if include_alpha=True
    Set delta_name to "delta" (AdaptiveSpectralDense) or "delta_z" (AdaptiveSpectralConv2d).
    """

    def __init__(
        self,
        model,
        prefix: str,
        *,
        U: Optional[jnp.ndarray] = None,
        s_dim: Optional[int] = None,
        delta_name: str = "delta",  # or "delta_z"
        include_alpha: bool = False,  # many models also have alpha_z
        rank: int = 8,
        jitter: float = 1e-5,
        mode: str = "reparam",
    ):
        super().__init__(model)
        assert (U is not None) or (s_dim is not None), "Provide U or s_dim."
        self.prefix = prefix
        self.delta_name = delta_name
        self.include_alpha = bool(include_alpha)
        self.s_dim = int(U.shape[1] if (U is not None) else s_dim)
        dim = (2 * self.s_dim) + (1 if self.include_alpha else 0)
        self.sampler = _LowRankJointSampler(
            f"{prefix}_svd_adapt_joint", dim=dim, rank=rank, jitter=jitter, mode=mode
        )

    def __call__(self, *args, **kwargs):
        z = self.sampler.sample()
        if self.include_alpha:
            core, alpha_z = z[:-1], z[-1]
        else:
            core = z
        s_vec, delta_vec = jnp.split(core, 2, axis=-1)

        numpyro.sample(f"{self.prefix}_s", dist.Delta(s_vec).to_event(1))
        numpyro.sample(
            f"{self.prefix}_{self.delta_name}", dist.Delta(delta_vec).to_event(1)
        )
        if self.include_alpha:
            numpyro.sample(f"{self.prefix}_alpha_z", dist.Delta(alpha_z))
            return {
                f"{self.prefix}_s": s_vec,
                f"{self.prefix}_{self.delta_name}": delta_vec,
                f"{self.prefix}_alpha_z": alpha_z,
            }
        return {
            f"{self.prefix}_s": s_vec,
            f"{self.prefix}_{self.delta_name}": delta_vec,
        }

    def sample_posterior(self, rng_key, params, sample_shape=()):
        z = self.sampler.sample_posterior(rng_key, params, sample_shape)
        if self.include_alpha:
            core, alpha_z = z[..., :-1], z[..., -1]
        else:
            core = z
        s_vec, delta_vec = jnp.split(core, 2, axis=-1)
        out = {f"{self.prefix}_s": s_vec, f"{self.prefix}_{self.delta_name}": delta_vec}
        if self.include_alpha:
            out[f"{self.prefix}_alpha_z"] = alpha_z
        return out


class MeanFieldSVDGuide(AutoGuide):
    """
    Mean-field guide for SVD-based layers:
      - binds {prefix}_s and optionally {prefix}_{delta_name} (adaptive) and {prefix}_alpha_z.
    """

    def __init__(
        self,
        model,
        prefix: str,
        *,
        s_dim: int,
        adaptive: bool = False,
        delta_name: str = "delta",
        include_alpha: bool = False,
        init_scale: float = 0.02,
    ):
        super().__init__(model)
        self.prefix = prefix
        self.s_dim = int(s_dim)
        self.adaptive = bool(adaptive)
        self.delta_name = delta_name
        self.include_alpha = bool(include_alpha)
        self.init_scale = float(init_scale)

    def __call__(self, *args, **kwargs):
        loc_s = numpyro.param(f"{self.prefix}_loc_s", jnp.zeros((self.s_dim,)))
        rho_s = numpyro.param(
            f"{self.prefix}_rho_s",
            jnp.full((self.s_dim,), math.log(math.expm1(self.init_scale))),
        )
        s_vec = numpyro.sample(
            f"{self.prefix}_s", dist.Normal(loc_s, _softplus(rho_s)).to_event(1)
        )

        if self.adaptive:
            loc_d = numpyro.param(
                f"{self.prefix}_loc_{self.delta_name}", jnp.zeros((self.s_dim,))
            )
            rho_d = numpyro.param(
                f"{self.prefix}_rho_{self.delta_name}",
                jnp.full((self.s_dim,), math.log(math.expm1(self.init_scale))),
            )
            d_vec = numpyro.sample(
                f"{self.prefix}_{self.delta_name}",
                dist.Normal(loc_d, _softplus(rho_d)).to_event(1),
            )
        if self.include_alpha:
            loc_a = numpyro.param(f"{self.prefix}_loc_alpha_z", jnp.array(0.0))
            rho_a = numpyro.param(
                f"{self.prefix}_rho_alpha_z",
                jnp.array(math.log(math.expm1(self.init_scale))),
            )
            a = numpyro.sample(
                f"{self.prefix}_alpha_z", dist.Normal(loc_a, _softplus(rho_a))
            )
        return {}


# ============================= Mean-field FFT =================================


class MeanFieldFFTGuide(AutoGuide):
    """1D FFT mean-field guide (fastest)."""

    def __init__(
        self,
        model,
        prefix: str,
        *,
        padded_dim: int,
        K: Optional[int] = None,
        init_scale: float = 0.02,
    ):
        super().__init__(model)
        self.prefix = prefix
        self.padded_dim = int(padded_dim)
        self.k_half = self.padded_dim // 2 + 1
        self.K = self.k_half if (K is None or K > self.k_half) else int(K)
        self.init_scale = float(init_scale)

    def __call__(self, *args, **kwargs):
        loc_r = numpyro.param(f"{self.prefix}_loc_r", jnp.zeros((self.K,)))
        loc_i = numpyro.param(f"{self.prefix}_loc_i", jnp.zeros((self.K,)))
        rho_r = numpyro.param(
            f"{self.prefix}_rho_r",
            jnp.full((self.K,), math.log(math.expm1(self.init_scale))),
        )
        rho_i = numpyro.param(
            f"{self.prefix}_rho_i",
            jnp.full((self.K,), math.log(math.expm1(self.init_scale))),
        )
        real_hp = numpyro.sample(
            f"{self.prefix}_real_aux",
            dist.Normal(loc_r, _softplus(rho_r)).to_event(1),
            infer={"is_auxiliary": True},
        )
        imag_hp = numpyro.sample(
            f"{self.prefix}_imag_aux",
            dist.Normal(loc_i, _softplus(rho_i)).to_event(1),
            infer={"is_auxiliary": True},
        )
        imag_hp = imag_hp.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.K == self.k_half) and (self.K > 1):
            imag_hp = imag_hp.at[-1].set(0.0)
        numpyro.sample(f"{self.prefix}_real", dist.Delta(real_hp).to_event(1))
        numpyro.sample(f"{self.prefix}_imag", dist.Delta(imag_hp).to_event(1))
        return {}


class MeanFieldFFTGuide2d(AutoGuide):
    """2D FFT mean-field guide (per-coefficient independent in half-plane)."""

    def __init__(
        self,
        model,
        prefix: str,
        *,
        C_out: int,
        C_in: int,
        H_pad: int,
        W_pad: int,
        init_scale: float = 0.02,
    ):
        super().__init__(model)
        self.prefix = prefix
        self.C_out, self.C_in = int(C_out), int(C_in)
        self.H_pad, self.W_pad = int(H_pad), int(W_pad)
        self.W_half = self.W_pad // 2 + 1
        self.shape_hp = (self.C_out, self.C_in, self.H_pad, self.W_half)
        self.init_scale = float(init_scale)

    def __call__(self, *args, **kwargs):
        loc_r = numpyro.param(f"{self.prefix}_loc_r", jnp.zeros(self.shape_hp))
        loc_i = numpyro.param(f"{self.prefix}_loc_i", jnp.zeros(self.shape_hp))
        rho_r = numpyro.param(
            f"{self.prefix}_rho_r",
            jnp.full(self.shape_hp, math.log(math.expm1(self.init_scale))),
        )
        rho_i = numpyro.param(
            f"{self.prefix}_rho_i",
            jnp.full(self.shape_hp, math.log(math.expm1(self.init_scale))),
        )
        real_hp = numpyro.sample(
            f"{self.prefix}_real_hp", dist.Normal(loc_r, _softplus(rho_r)).to_event(4)
        )
        imag_hp = numpyro.sample(
            f"{self.prefix}_imag_hp", dist.Normal(loc_i, _softplus(rho_i)).to_event(4)
        )
        r_full, i_full = halfplane_to_full_multi(
            real_hp, imag_hp, self.H_pad, self.W_pad
        )
        numpyro.sample(f"{self.prefix}_real", dist.Delta(r_full).to_event(4))
        numpyro.sample(f"{self.prefix}_imag", dist.Delta(i_full).to_event(4))
        return {}
