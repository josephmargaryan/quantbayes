from typing import Tuple
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoGuide
import math

__all__ = [
    "LowRankFFTGuide2d",
    "LowRankFFTGuide",
    "LowRankAdaptiveFFTGuide2d",
    "LowRankAdaptiveFFTGuide",
    "LowRankPSM1DGuide",
    "LowRankPSM2DGuide",
]

"""
Usage:

# build the AutoGuideList by hand:
guide = AutoGuideList(model)

# 1) low‑rank guide for the spectral layer
guide.append(
    LowRankFFTGuide(
        model,
        prefix="mix",
        H_pad=28,
        W_pad=28,
        rank=4,
    )
)

guide.append(
    AutoNormal(
        numpyro.handlers.block(
        model, hide=["mix_real", "mix_imag", "logits"]
        ),
        init_loc_fn=numpyro.infer.init_to_feasible,
    )
)

"""


def _enforce_hermitian(fft2d: jnp.ndarray) -> jnp.ndarray:
    """Project (..., H, W) complex tensor onto Hermitian subspace."""
    H, W = fft2d.shape[-2:]
    conj_flip = jnp.flip(jnp.conj(fft2d), axis=(-2, -1))
    herm = 0.5 * (fft2d + conj_flip)
    herm = herm.at[..., 0, 0].set(jnp.real(herm[..., 0, 0]))
    if H % 2 == 0:
        herm = herm.at[..., H // 2, :].set(jnp.real(herm[..., H // 2, :]))
    if W % 2 == 0:
        herm = herm.at[..., :, W // 2].set(jnp.real(herm[..., :, W // 2]))
    return herm


class LowRankFFTGuide(AutoGuide):
    """
    Low‑rank Gaussian guide over {prefix}_real and {prefix}_imag
    for a 1D spectral layer that only samples K active frequencies.
    """

    def __init__(
        self,
        model,
        prefix: str,
        padded_dim: int,
        K: int = None,
        rank: int = 8,
        jitter: float = 1e-4,
    ):
        super().__init__(model)
        self.prefix = prefix
        self.padded_dim = padded_dim
        self.k_half = padded_dim // 2 + 1
        self.K = self.k_half if (K is None or K > self.k_half) else K

        self.M = self.K
        self.rank = rank
        self.jitter = jitter
        self.joint = f"{prefix}_joint"

    def _build_lowrank(self):
        dim = 2 * self.M
        loc = numpyro.param(f"{self.joint}_loc", jnp.zeros(dim))
        V_init = (
            0.05
            * jax.random.normal(numpyro.prng_key(), (dim, self.rank))
            / math.sqrt(dim)
        )
        V = numpyro.param(f"{self.joint}_V", V_init)
        d_raw = numpyro.param(f"{self.joint}_d_raw", jnp.zeros(dim))
        log_tau = numpyro.param(f"{self.joint}_log_tau", jnp.array(0.0))
        diag = jax.nn.softplus(d_raw) * jnp.exp(log_tau) + self.jitter

        cov = V @ V.T + jnp.diag(diag)
        return numpyro.sample(
            self.joint,
            dist.MultivariateNormal(loc, covariance_matrix=cov),
            infer={"is_auxiliary": True},
        )

    def __call__(self, *args, **kwargs):

        z = self._build_lowrank()
        real_hp, imag_hp = jnp.split(z, 2, axis=-1)

        imag_hp = imag_hp.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.K == self.k_half):
            imag_hp = imag_hp.at[-1].set(0.0)

        numpyro.sample(
            f"{self.prefix}_real",
            dist.Delta(real_hp).to_event(1),
        )
        numpyro.sample(
            f"{self.prefix}_imag",
            dist.Delta(imag_hp).to_event(1),
        )

        return {
            f"{self.prefix}_real": real_hp,
            f"{self.prefix}_imag": imag_hp,
        }

    def sample_posterior(self, rng_key, params, sample_shape=()):
        loc = params[f"{self.joint}_loc"]
        V = params[f"{self.joint}_V"]
        d_raw = params[f"{self.joint}_d_raw"]
        log_tau = params[f"{self.joint}_log_tau"]
        diag = jax.nn.softplus(d_raw) * jnp.exp(log_tau) + self.jitter
        cov = V @ V.T + jnp.diag(diag)

        mvn = dist.MultivariateNormal(loc, covariance_matrix=cov)
        z = mvn.sample(rng_key, sample_shape)  # (..., 2*K)

        real_hp = z[..., : self.M]
        imag_hp = z[..., self.M :]

        imag_hp = imag_hp.at[..., 0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.K == self.k_half):
            imag_hp = imag_hp.at[..., -1].set(0.0)

        return {
            f"{self.prefix}_real": real_hp,
            f"{self.prefix}_imag": imag_hp,
        }


class LowRankAdaptiveFFTGuide(AutoGuide):
    """
    Low‑rank guide over three half‑spectrum sites in the 1‑D non‑stationary model:
      - {prefix}_delta_alpha   shape (k_half,)
      - {prefix}_real          shape (k_half,)
      - {prefix}_imag          shape (k_half,)
    Total joint latent size = 3 * k_half.
    """

    def __init__(
        self,
        model,
        prefix: str,
        padded_dim: int,
        rank: int = 8,
        jitter: float = 1e-5,
    ):
        super().__init__(model)
        self.prefix = prefix
        self.padded_dim = padded_dim
        self.k_half = padded_dim // 2 + 1
        self.M = self.k_half
        self.rank = rank
        self.jitter = jitter
        self.joint = f"{prefix}_joint"

    def _build_lowrank(self):
        dim = 3 * self.M
        loc = numpyro.param(f"{self.joint}_loc", jnp.zeros(dim))
        V_init = (
            0.05
            * jax.random.normal(numpyro.prng_key(), (dim, self.rank))
            / math.sqrt(dim)
        )
        V = numpyro.param(f"{self.joint}_V", V_init)
        d_raw = numpyro.param(f"{self.joint}_d_raw", -3.0 * jnp.ones(dim))
        log_tau = numpyro.param(f"{self.joint}_log_tau", jnp.array(0.0))
        diag = jax.nn.softplus(d_raw) * jnp.exp(log_tau) + self.jitter

        cov = V @ V.T + jnp.diag(diag)
        return numpyro.sample(
            self.joint,
            dist.MultivariateNormal(loc, covariance_matrix=cov),
            infer={"is_auxiliary": True},
        )

    def __call__(self, *args, **kwargs):
        z = self._build_lowrank()

        real_hp, imag_hp, delta_alpha = jnp.split(z, [self.M, 2 * self.M], axis=-1)

        imag_hp = imag_hp.at[0].set(0.0)
        if self.padded_dim % 2 == 0:
            imag_hp = imag_hp.at[-1].set(0.0)

        numpyro.sample(
            f"{self.prefix}_delta_alpha",
            dist.Delta(delta_alpha).to_event(1),
        )
        numpyro.sample(
            f"{self.prefix}_real",
            dist.Delta(real_hp).to_event(1),
        )
        numpyro.sample(
            f"{self.prefix}_imag",
            dist.Delta(imag_hp).to_event(1),
        )

        return {
            f"{self.prefix}_delta_alpha": delta_alpha,
            f"{self.prefix}_real": real_hp,
            f"{self.prefix}_imag": imag_hp,
        }

    def sample_posterior(self, rng_key, params, sample_shape=()):
        loc = params[f"{self.joint}_loc"]
        V = params[f"{self.joint}_V"]
        d_raw = params[f"{self.joint}_d_raw"]
        log_tau = params[f"{self.joint}_log_tau"]
        diag = jax.nn.softplus(d_raw) * jnp.exp(log_tau) + self.jitter
        cov = V @ V.T + jnp.diag(diag)

        mvn = dist.MultivariateNormal(loc, covariance_matrix=cov)
        z = mvn.sample(rng_key, sample_shape)  # (..., 3*M)

        real_hp, imag_hp, delta_alpha = jnp.split(z, [self.M, 2 * self.M], axis=-1)
        imag_hp = imag_hp.at[..., 0].set(0.0)
        if self.padded_dim % 2 == 0:
            imag_hp = imag_hp.at[..., -1].set(0.0)

        return {
            f"{self.prefix}_delta_alpha": delta_alpha,
            f"{self.prefix}_real": real_hp,
            f"{self.prefix}_imag": imag_hp,
        }


def halfplane_to_full_multi(real_hp, imag_hp, H_pad, W_pad):
    """
    real_hp, imag_hp : (..., H_pad, W_half)
       leading dims may be (C_out, C_in) or batch.
    Returns complex Hermitian (..., H_pad, W_pad) array.
    """
    W_half = real_hp.shape[-1]
    full = jnp.zeros(real_hp.shape[:-1] + (W_pad,), dtype=real_hp.dtype) + 0j
    full = full.at[..., :W_half].set(real_hp + 1j * imag_hp)

    full = _enforce_hermitian(full)
    return full.real, full.imag


class LowRankFFTGuide2d(AutoGuide):
    """
    Low-rank Gaussian guide over the **entire channelled half-plane**:
        real_hp : (C_out, C_in, H_pad, W_half)
        imag_hp : same
    """

    def __init__(
        self,
        model,
        prefix: str,
        C_out: int,
        C_in: int,
        H_pad: int,
        W_pad: int,
        rank: int = 8,
        jitter: float = 1e-4,
    ):
        super().__init__(model)
        self.prefix = prefix
        self.C_out, self.C_in = C_out, C_in
        self.H_pad, self.W_pad = H_pad, W_pad
        self.W_half = W_pad // 2 + 1
        self.M = C_out * C_in * H_pad * self.W_half
        self.rank = rank
        self.jitter = jitter
        self.joint = f"{prefix}_joint"

    def _build_lowrank(self):
        dim = 2 * self.M
        loc = numpyro.param(f"{self.joint}_loc", jnp.zeros(dim))
        V = numpyro.param(
            f"{self.joint}_V",
            0.05
            * jax.random.normal(numpyro.prng_key(), (dim, self.rank))
            / math.sqrt(dim),
        )
        d_raw = numpyro.param(f"{self.joint}_d_raw", jnp.zeros(dim))
        log_tau = numpyro.param(f"{self.joint}_log_tau", jnp.array(0.0))
        diag = jax.nn.softplus(d_raw) * jnp.exp(log_tau) + self.jitter
        cov = V @ V.T + jnp.diag(diag)

        return numpyro.sample(
            self.joint,
            dist.MultivariateNormal(loc, covariance_matrix=cov),
            infer={"is_auxiliary": True},
        )

    def __call__(self, *args, **kwargs):
        z = self._build_lowrank()

        real_hp, imag_hp = jnp.split(z, 2, axis=-1)
        shape_hp = (self.C_out, self.C_in, self.H_pad, self.W_half)
        real_hp = real_hp.reshape(shape_hp)
        imag_hp = imag_hp.reshape(shape_hp)

        r_full, i_full = halfplane_to_full_multi(
            real_hp, imag_hp, self.H_pad, self.W_pad
        )

        numpyro.sample(
            f"{self.prefix}_real",
            dist.Delta(r_full).to_event(4),
        )
        numpyro.sample(
            f"{self.prefix}_imag",
            dist.Delta(i_full).to_event(4),
        )

        return {f"{self.prefix}_real": r_full, f"{self.prefix}_imag": i_full}

    def sample_posterior(self, rng_key, params, sample_shape=()):
        dim = 2 * self.M
        loc = params[f"{self.joint}_loc"]
        V = params[f"{self.joint}_V"]
        d_raw = params[f"{self.joint}_d_raw"]
        ltau = params[f"{self.joint}_log_tau"]
        diag = jax.nn.softplus(d_raw) * jnp.exp(ltau) + self.jitter
        cov = V @ V.T + jnp.diag(diag)

        mvn = dist.MultivariateNormal(loc, covariance_matrix=cov)
        z = mvn.sample(rng_key, sample_shape)
        real_hp, imag_hp = jnp.split(z, 2, axis=-1)

        shape_hp = sample_shape + (self.C_out, self.C_in, self.H_pad, self.W_half)
        real_hp = real_hp.reshape(shape_hp)
        imag_hp = imag_hp.reshape(shape_hp)

        r_full, i_full = jax.vmap(
            lambda r, i: halfplane_to_full_multi(r, i, self.H_pad, self.W_pad),
            in_axes=0,
        )(real_hp, imag_hp)

        return {f"{self.prefix}_real": r_full, f"{self.prefix}_imag": i_full}


class LowRankAdaptiveFFTGuide2d(AutoGuide):
    """
    Joint low-rank guide over:
        - Δα coarse grid  (gh,gw)
        - multi-channel half-plane fourier coeffs
    """

    def __init__(
        self,
        model,
        prefix: str,
        C_out: int,
        C_in: int,
        H_pad: int,
        W_pad: int,
        alpha_coarse_shape: Tuple[int, int] = (8, 8),
        rank: int = 8,
        jitter: float = 1e-4,
    ):
        super().__init__(model)
        self.prefix = prefix
        self.C_out, self.C_in = C_out, C_in
        self.H_pad, self.W_pad = H_pad, W_pad
        self.W_half = W_pad // 2 + 1
        self.M = C_out * C_in * H_pad * self.W_half

        self.gh, self.gw = alpha_coarse_shape
        self.G = self.gh * self.gw
        self.rank = rank
        self.jitter = jitter
        self.joint = f"{prefix}_joint"

    def _build_lowrank(self):
        dim = 2 * self.M + self.G
        loc = numpyro.param(f"{self.joint}_loc", jnp.zeros(dim))
        V = numpyro.param(
            f"{self.joint}_V",
            0.05
            * jax.random.normal(numpyro.prng_key(), (dim, self.rank))
            / math.sqrt(dim),
        )
        d_raw = numpyro.param(f"{self.joint}_d_raw", -3.0 * jnp.ones(dim))
        log_tau = numpyro.param(f"{self.joint}_log_tau", jnp.array(0.0))
        diag = jax.nn.softplus(d_raw) * jnp.exp(log_tau) + self.jitter
        cov = V @ V.T + jnp.diag(diag)

        return numpyro.sample(
            self.joint,
            dist.MultivariateNormal(loc, covariance_matrix=cov),
            infer={"is_auxiliary": True},
        )

    def __call__(self, *args, **kwargs):
        z = self._build_lowrank()

        z_f, delta_flat = jnp.split(z, [2 * self.M], axis=-1)
        real_hp, imag_hp = jnp.split(z_f, 2, axis=-1)

        shape_hp = (self.C_out, self.C_in, self.H_pad, self.W_half)
        real_hp = real_hp.reshape(shape_hp)
        imag_hp = imag_hp.reshape(shape_hp)
        delta_map = delta_flat.reshape(self.gh, self.gw)

        numpyro.sample(
            f"{self.prefix}_delta_alpha",
            dist.Delta(delta_map).to_event(2),
        )

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
        dim = 2 * self.M + self.G
        loc = params[f"{self.joint}_loc"]
        V = params[f"{self.joint}_V"]
        d_raw = params[f"{self.joint}_d_raw"]
        ltau = params[f"{self.joint}_log_tau"]
        diag = jax.nn.softplus(d_raw) * jnp.exp(ltau) + self.jitter
        cov = V @ V.T + jnp.diag(diag)

        mvn = dist.MultivariateNormal(loc, covariance_matrix=cov)
        z = mvn.sample(rng_key, sample_shape)

        z_f, delta_flat = jnp.split(z, [2 * self.M], axis=-1)
        real_hp, imag_hp = jnp.split(z_f, 2, axis=-1)

        shp_hp = sample_shape + (self.C_out, self.C_in, self.H_pad, self.W_half)
        real_hp = real_hp.reshape(shp_hp)
        imag_hp = imag_hp.reshape(shp_hp)
        delta_map = delta_flat.reshape(sample_shape + (self.gh, self.gw))

        r_full, i_full = jax.vmap(
            lambda r, i: halfplane_to_full_multi(r, i, self.H_pad, self.W_pad),
            in_axes=0,
        )(real_hp, imag_hp)

        return {
            f"{self.prefix}_delta_alpha": delta_map,
            f"{self.prefix}_real": r_full,
            f"{self.prefix}_imag": i_full,
        }


class LowRankPSM1DGuide(AutoGuide):
    """
    Low-rank+diag guide over the 1D-PSM layer latents:
      • logits: (P, Q)    ← model batch dims, event_dim=0
      • mu    : (P, Q)    ← model batch dims, event_dim=0
      • sigma : (P, Q)    ← model batch dims, event_dim=0 (we parametrize raw_logσ then exp)
      • bias  : (P, l)    ← model to_event(2)
    Joint size D = P*(3Q + l).

    Example Usage:
        guide1 = AutoGuideList(model_1d)
        guide1.append(LowRankPSM1DGuide(model_1d, prefix="psm1d", P=P1, Q=Q1, l=l1, rank=8))
        hide1 = ["psm1d_logits","psm1d_mu","psm1d_sigma","psm1d_bias"]
        guide1.append(AutoNormal(numpyro.handlers.block(model_1d, hide=hide1)))
    """

    def __init__(
        self,
        model,
        prefix: str,
        P: int,
        Q: int,
        l: int,
        rank: int = 8,
        jitter: float = 1e-6,
    ):
        super().__init__(model)
        self.prefix, self.P, self.Q, self.l = prefix, P, Q, l
        self.D = P * (3 * Q + l)
        self.rank, self.jitter = rank, jitter
        self.joint = f"{prefix}_psm1d_joint"

    def _build_lowrank(self):
        loc = numpyro.param(f"{self.joint}_loc", jnp.zeros(self.D))
        V_init = (
            0.05
            * jax.random.normal(numpyro.prng_key(), (self.D, self.rank))
            / math.sqrt(self.D)
        )
        V = numpyro.param(f"{self.joint}_V", V_init)
        d_raw = numpyro.param(f"{self.joint}_d_raw", jnp.zeros(self.D))
        diag = jax.nn.softplus(d_raw) + self.jitter

        return numpyro.sample(
            self.joint,
            dist.LowRankMultivariateNormal(loc, V, diag),
            infer={"is_auxiliary": True},
        )

    def __call__(self, *args, **kwargs):
        z = self._build_lowrank()
        s1 = self.P * self.Q
        s2 = s1 + self.P * self.Q
        s3 = s2 + self.P * self.Q

        flat_logits = z[:s1].reshape(self.P, self.Q)
        flat_mu = z[s1:s2].reshape(self.P, self.Q)
        raw_logsig = z[s2:s3].reshape(self.P, self.Q)
        flat_bias = z[s3:].reshape(self.P, self.l)

        sigma = jnp.exp(raw_logsig)

        numpyro.sample(f"{self.prefix}_logits", dist.Delta(flat_logits, event_dim=0))
        numpyro.sample(f"{self.prefix}_mu", dist.Delta(flat_mu, event_dim=0))
        numpyro.sample(f"{self.prefix}_sigma", dist.Delta(sigma, event_dim=0))
        numpyro.sample(f"{self.prefix}_bias", dist.Delta(flat_bias, event_dim=2))

        return {
            f"{self.prefix}_logits": flat_logits,
            f"{self.prefix}_mu": flat_mu,
            f"{self.prefix}_sigma": sigma,
            f"{self.prefix}_bias": flat_bias,
        }

    def sample_posterior(self, rng_key, params, sample_shape=()):
        from numpyro.handlers import substitute, seed, trace

        guided = substitute(self, params)
        guided = seed(guided, rng_key)
        tr = trace(guided).get_trace()
        return {
            f"{self.prefix}_logits": tr[f"{self.prefix}_logits"]["value"],
            f"{self.prefix}_mu": tr[f"{self.prefix}_mu"]["value"],
            f"{self.prefix}_sigma": tr[f"{self.prefix}_sigma"]["value"],
            f"{self.prefix}_bias": tr[f"{self.prefix}_bias"]["value"],
        }


class LowRankPSM2DGuide(AutoGuide):
    """
    Low-rank+diag guide over the 2D-PSM layer latents:
      • logits: (P, Q)        ← event_dim=0
      • mu    : (P, Q, 2)     ← event_dim=0
      • sigma : (P, Q, 2)     ← event_dim=0 (raw → exp)
      • biasₚ : (ph, pw) × P  ← each to_event(2)
    Joint size D = P*(Q + 2Q + 2Q + ph*pw).

    Example Usage:
        guide2 = AutoGuideList(model_2d)
        guide2.append(LowRankPSM2DGuide(model_2d, prefix="psm2d", P=P2, Q=Q2, ph=ph2, pw=pw2, rank=8))
        hide2 = ["psm2d_logits","psm2d_mu","psm2d_sigma"] + [f"psm2d_bias_{p}" for p in range(P2)]
        guide2.append(AutoNormal(numpyro.handlers.block(model_2d, hide=hide2)))
    """

    def __init__(
        self,
        model,
        prefix: str,
        P: int,
        Q: int,
        ph: int,
        pw: int,
        rank: int = 8,
        jitter: float = 1e-6,
    ):
        super().__init__(model)
        self.prefix, self.P, self.Q = prefix, P, Q
        self.ph, self.pw = ph, pw
        d_logits = P * Q
        d_mu = P * Q * 2
        d_sig = P * Q * 2
        d_bias = P * ph * pw
        self.D = d_logits + d_mu + d_sig + d_bias
        self.rank, self.jitter = rank, jitter
        self.joint = f"{prefix}_psm2d_joint"

    def _build_lowrank(self):
        loc = numpyro.param(f"{self.joint}_loc", jnp.zeros(self.D))
        V_init = (
            0.05
            * jax.random.normal(numpyro.prng_key(), (self.D, self.rank))
            / math.sqrt(self.D)
        )
        V = numpyro.param(f"{self.joint}_V", V_init)
        d_raw = numpyro.param(f"{self.joint}_d_raw", jnp.zeros(self.D))
        diag = jax.nn.softplus(d_raw) + self.jitter

        return numpyro.sample(
            self.joint,
            dist.LowRankMultivariateNormal(loc, V, diag),
            infer={"is_auxiliary": True},
        )

    def __call__(self, *args, **kwargs):
        z = self._build_lowrank()

        s1 = self.P * self.Q
        s2 = s1 + self.P * self.Q * 2
        s3 = s2 + self.P * self.Q * 2

        flat_logits = z[:s1].reshape(self.P, self.Q)
        flat_mu = z[s1:s2].reshape(self.P, self.Q, 2)
        raw_logsig = z[s2:s3].reshape(self.P, self.Q, 2)
        flat_bias = z[s3:].reshape(self.P, self.ph, self.pw)

        sigma = jnp.exp(raw_logsig)

        numpyro.sample(f"{self.prefix}_logits", dist.Delta(flat_logits, event_dim=0))
        numpyro.sample(f"{self.prefix}_mu", dist.Delta(flat_mu, event_dim=0))
        numpyro.sample(f"{self.prefix}_sigma", dist.Delta(sigma, event_dim=0))

        for p in range(self.P):
            numpyro.sample(
                f"{self.prefix}_bias_{p}", dist.Delta(flat_bias[p], event_dim=2)
            )

        out = {
            f"{self.prefix}_logits": flat_logits,
            f"{self.prefix}_mu": flat_mu,
            f"{self.prefix}_sigma": sigma,
        }
        for p in range(self.P):
            out[f"{self.prefix}_bias_{p}"] = flat_bias[p]
        return out

    def sample_posterior(self, rng_key, params, sample_shape=()):
        from numpyro.handlers import substitute, seed, trace

        guided = substitute(self, params)
        guided = seed(guided, rng_key)
        tr = trace(guided).get_trace()
        out = {
            f"{self.prefix}_logits": tr[f"{self.prefix}_logits"]["value"],
            f"{self.prefix}_mu": tr[f"{self.prefix}_mu"]["value"],
            f"{self.prefix}_sigma": tr[f"{self.prefix}_sigma"]["value"],
        }
        for p in range(self.P):
            out[f"{self.prefix}_bias_{p}"] = tr[f"{self.prefix}_bias_{p}"]["value"]
        return out
