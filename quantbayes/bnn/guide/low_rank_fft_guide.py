from typing import Callable, Sequence, Optional
import math
import jax
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from numpyro.infer.autoguide import AutoGuide, AutoGuideList, AutoNormal


class LowRankFFTGuide(AutoGuide):
    """Generic low‑rank Gaussian guide over a pair of sampling sites
            {prefix}_real  and  {prefix}_imag .
    The guide stores a joint vector z ∈ ℝ^{2M} where M = H_pad * (W_pad//2+1)
    (half‑plane).  The covariance is V Vᵀ + diag(softplus(d)) with learnable rank.
    A global scale τ = exp(log_tau) is applied to the diagonal part, following
    Dusenberry et al. (2020) for better calibration.
    """

    def __init__(
        self,
        model: Callable,
        prefix: str,
        H_pad: int,
        W_pad: int,
        rank: int = 8,
        jitter: float = 1e-5,
    ):
        super().__init__(model)
        self.prefix = prefix
        self.H_pad, self.W_pad = H_pad, W_pad
        self.W_half = W_pad // 2 + 1
        self.M = H_pad * self.W_half
        self.rank = rank
        self.jitter = jitter
        self.joint = f"{prefix}_joint"

    # ------------------------------------------------------------------
    def _build_lowrank(self):
        dim = 2 * self.M
        # ----- parameters -----
        loc = numpyro.param(f"{self.joint}_loc", jnp.zeros(dim))

        key = numpyro.prng_key()
        V_init = 0.05 * jax.random.normal(key, (dim, self.rank)) / math.sqrt(dim)
        V = numpyro.param(f"{self.joint}_V", V_init)

        d_raw = numpyro.param(f"{self.joint}_d_raw", -3.0 * jnp.ones(dim))
        log_tau = numpyro.param(f"{self.joint}_log_tau", jnp.array(0.0))
        diag = jax.nn.softplus(d_raw) * jnp.exp(log_tau) + self.jitter

        cov = V @ V.T + jnp.diag(diag)

        return numpyro.sample(
            self.joint,
            dist.MultivariateNormal(loc, covariance_matrix=cov),
            infer={"is_auxiliary": True},
        )  # z ∈ ℝ^{2M}

    # ------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        z = self._build_lowrank()  # (2M,)

        real_hp, imag_hp = jnp.split(z, 2)
        real_hp = real_hp.reshape(self.H_pad, self.W_half)
        imag_hp = imag_hp.reshape(self.H_pad, self.W_half)

        # Reconstruct full grid via conjugate symmetry
        real_full = jnp.zeros((self.H_pad, self.W_pad))
        imag_full = jnp.zeros((self.H_pad, self.W_pad))
        real_full = real_full.at[:, : self.W_half].set(real_hp)
        imag_full = imag_full.at[:, : self.W_half].set(imag_hp)

        u = jnp.arange(self.H_pad)[:, None]
        v = jnp.arange(1, self.W_half)[None, :]
        u_conj = (-u) % self.H_pad
        v_conj = (-v) % self.W_pad
        real_full = real_full.at[u_conj, v_conj].set(real_hp[u_conj, v])
        imag_full = imag_full.at[u_conj, v_conj].set(-imag_hp[u_conj, v])

        # DC & Nyquist purely real
        imag_full = imag_full.at[0, 0].set(0.0)
        if self.H_pad % 2 == 0:
            imag_full = imag_full.at[self.H_pad // 2, :].set(0.0)
        if self.W_pad % 2 == 0:
            imag_full = imag_full.at[:, self.W_pad // 2].set(0.0)

        numpyro.sample(f"{self.prefix}_real", dist.Delta(real_full).to_event(2))
        numpyro.sample(f"{self.prefix}_imag", dist.Delta(imag_full).to_event(2))

        return {f"{self.prefix}_real": real_full, f"{self.prefix}_imag": imag_full}

    # ------------------------------------------------------------------
    def sample_posterior(self, rng_key, params, sample_shape=()):
        from numpyro.contrib.funsor import log_prob_sum

        dim = 2 * self.M
        loc = params[f"{self.joint}_loc"]
        V = params[f"{self.joint}_V"]
        d_raw = params[f"{self.joint}_d_raw"]
        log_tau = params[f"{self.joint}_log_tau"]
        diag = jax.nn.softplus(d_raw) * jnp.exp(log_tau) + self.jitter
        cov = V @ V.T + jnp.diag(diag)

        mvn = dist.MultivariateNormal(loc, covariance_matrix=cov)
        z = mvn.sample(rng_key, sample_shape)

        real_hp, imag_hp = jnp.split(z, 2, axis=-1)
        real_hp = real_hp.reshape(sample_shape + (self.H_pad, self.W_half))
        imag_hp = imag_hp.reshape(sample_shape + (self.H_pad, self.W_half))

        # Reconstruct full grid (vectorised)
        real_full = jnp.zeros(sample_shape + (self.H_pad, self.W_pad))
        imag_full = jnp.zeros(sample_shape + (self.H_pad, self.W_pad))
        real_full = real_full.at[..., :, : self.W_half].set(real_hp)
        imag_full = imag_full.at[..., :, : self.W_half].set(imag_hp)

        u = jnp.arange(self.H_pad)[:, None]
        v = jnp.arange(1, self.W_half)[None, :]
        u_conj = (-u) % self.H_pad
        v_conj = (-v) % self.W_pad
        real_full = real_full.at[..., u_conj, v_conj].set(real_hp[..., u_conj, v])
        imag_full = imag_full.at[..., u_conj, v_conj].set(-imag_hp[..., u_conj, v])

        return {f"{self.prefix}_real": real_full, f"{self.prefix}_imag": imag_full}


# -----------------------------------------------------------------------------
# Convenience: build a full AutoGuideList (low‑rank + AutoNormal remainder)
# -----------------------------------------------------------------------------


def make_spectral_guides(
    model: Callable,
    spectral_specs: Sequence[dict],
    autonormal_kwargs: Optional[dict] = None,
):
    """Return an AutoGuideList with one LowRankFFTGuide per specification plus
    an AutoNormal for remaining latent sites.

    Parameters
    ----------
    model            : callable NumPyro model
    spectral_specs   : iterable of dicts, each with keys
                        {"prefix", "H_pad", "W_pad", "rank"}
    autonormal_kwargs: forwarded to AutoNormal (e.g. init_scale, mean_field)
    """
    guide = AutoGuideList(model)
    for spec in spectral_specs:
        guide.append(
            LowRankFFTGuide(
                model,
                prefix=spec["prefix"],
                H_pad=spec["H_pad"],
                W_pad=spec["W_pad"],
                rank=spec.get("rank", 8),
            )
        )
    guide.append(AutoNormal(model, **(autonormal_kwargs or {})))
    return guide
