import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoGuide
import math

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

def _halfplane_to_full(
    real_hp: jnp.ndarray,
    imag_hp: jnp.ndarray,
    H_pad: int,
    W_pad: int
):
    """
    Mirror a half‐plane (H_pad, W_half) into a full Hermitian grid (H_pad, W_pad).
    """
    # compute W_half from inputs, but we already know it from imag_hp.shape[1]
    W_half = imag_hp.shape[1]

    # allocate full arrays
    real_full = jnp.zeros((H_pad, W_pad), dtype=real_hp.dtype)
    imag_full = jnp.zeros((H_pad, W_pad), dtype=imag_hp.dtype)

    # fill the "positive‐frequency" side
    real_full = real_full.at[:, :W_half].set(real_hp)
    imag_full = imag_full.at[:, :W_half].set(imag_hp)

    # mirror indices (skip DC column at v=0)
    u = jnp.arange(H_pad)[:, None]        # shape (H_pad,1)
    v = jnp.arange(1, W_half)[None, :]    # shape (1,W_half-1)
    u_conj = (-u) % H_pad
    v_conj = (-v) % W_pad

    # real symmetric, imag antisymmetric
    real_full = real_full.at[u_conj, v_conj].set(real_hp[u_conj, v])
    imag_full = imag_full.at[u_conj, v_conj].set(-imag_hp[u_conj, v])

    # enforce purely real DC and Nyquist lines
    imag_full = imag_full.at[0, 0].set(0.0)
    if H_pad % 2 == 0:
        imag_full = imag_full.at[H_pad // 2, :].set(0.0)
    if W_pad % 2 == 0:
        imag_full = imag_full.at[:, W_pad // 2].set(0.0)

    return real_full, imag_full



class LowRankFFTGuide(AutoGuide):
    """
    Generic low‑rank Gaussian guide over {prefix}_real and {prefix}_imag.
    """

    def __init__(self, model, prefix, H_pad, W_pad, rank=8, jitter=1e-4):
        super().__init__(model)
        self.prefix = prefix
        self.H_pad, self.W_pad = H_pad, W_pad
        self.W_half = W_pad // 2 + 1
        self.M = H_pad * self.W_half
        self.rank = rank
        self.jitter = jitter
        self.joint = f"{prefix}_joint"

    def _build_lowrank(self):
        dim = 2 * self.M
        loc = numpyro.param(f"{self.joint}_loc", jnp.zeros(dim))
        V_init = 0.05 * jax.random.normal(numpyro.prng_key(), (dim, self.rank)) \
                 / math.sqrt(dim)
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
        z = self._build_lowrank()                  # shape (2*M,)
        real_hp, imag_hp = jnp.split(z, 2, axis=-1)  # two arrays of shape (M,)
        real_hp = real_hp.reshape(self.H_pad, self.W_half)
        imag_hp = imag_hp.reshape(self.H_pad, self.W_half)

        # mirror to full
        real_full, imag_full = _halfplane_to_full(
            real_hp, imag_hp, self.H_pad, self.W_pad
        )

        # register Delta sites
        numpyro.sample(f"{self.prefix}_real",
                       dist.Delta(real_full).to_event(2))
        numpyro.sample(f"{self.prefix}_imag",
                       dist.Delta(imag_full).to_event(2))

        return {
            f"{self.prefix}_real": real_full,
            f"{self.prefix}_imag": imag_full,
        }

    def sample_posterior(self, rng_key, params, sample_shape=()):
        dim = 2 * self.M
        loc    = params[f"{self.joint}_loc"]
        V      = params[f"{self.joint}_V"]
        d_raw  = params[f"{self.joint}_d_raw"]
        log_tau= params[f"{self.joint}_log_tau"]
        diag   = jax.nn.softplus(d_raw) * jnp.exp(log_tau) + self.jitter
        cov    = V @ V.T + jnp.diag(diag)

        mvn = dist.MultivariateNormal(loc, covariance_matrix=cov)
        z   = mvn.sample(rng_key, sample_shape)  # (..., 2*M)

        # split & reshape
        real_hp = z[..., : self.M].reshape(sample_shape + (self.H_pad, self.W_half))
        imag_hp = z[..., self.M :].reshape(sample_shape + (self.H_pad, self.W_half))

        # mirror each sample via vmap over axis=0
        real_full, imag_full = jax.vmap(
            lambda rhp, ihp: _halfplane_to_full(rhp, ihp, self.H_pad, self.W_pad)
        )(real_hp, imag_hp)

        return {
            f"{self.prefix}_real": real_full,
            f"{self.prefix}_imag": imag_full,
        }



class LowRankNonStatFFTGuide(LowRankFFTGuide):
    """
    Appends the Δα‑map (coarse grid) to the joint latent vector.
    """

    def __init__(self, model, prefix, H_pad, W_pad,
                 alpha_coarse_shape=(8, 8), rank=8, jitter=1e-4):
        super().__init__(model, prefix, H_pad, W_pad, rank, jitter)
        self.gh, self.gw = alpha_coarse_shape
        self.G = self.gh * self.gw

    def _build_lowrank(self):
        # override to expand dim by G
        dim = 2 * self.M + self.G
        loc = numpyro.param(f"{self.joint}_loc", jnp.zeros(dim))
        V_init = 0.05 * jax.random.normal(numpyro.prng_key(),
                                          (dim, self.rank)) / math.sqrt(dim)
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
        z = self._build_lowrank()                  # shape (2*M + G,)
        # split off alpha‐grid
        f_real, f_imag, delta_flat = jnp.split(
            z, [self.M, 2 * self.M], axis=-1
        )
        real_hp = f_real.reshape(self.H_pad, self.W_half)
        imag_hp = f_imag.reshape(self.H_pad, self.W_half)
        delta_map = delta_flat.reshape(self.gh, self.gw)

        # expose the map
        numpyro.sample(
            f"{self.prefix}_delta_alpha_map",
            dist.Delta(delta_map).to_event(2),
        )

        # mirror real+imag
        real_full, imag_full = _halfplane_to_full(
            real_hp, imag_hp, self.H_pad, self.W_pad
        )

        numpyro.sample(f"{self.prefix}_real",
                       dist.Delta(real_full).to_event(2))
        numpyro.sample(f"{self.prefix}_imag",
                       dist.Delta(imag_full).to_event(2))

        return {
            f"{self.prefix}_real": real_full,
            f"{self.prefix}_imag": imag_full,
            f"{self.prefix}_delta_alpha_map": delta_map,
        }

class LowRankNonStat1DGuide(AutoGuide):
    """
    Low‑rank guide over three half‑spectrum sites in the 1‑D model:
      - {prefix}_delta_alpha   shape (k_half,)
      - {prefix}_real          shape (k_half,)
      - {prefix}_imag          shape (k_half,)
    Total joint latent size = 3 * k_half.
    """

    def __init__(self, model, prefix, H_pad, W_pad, rank=8, jitter=1e-5):
        super().__init__(model)
        self.prefix = prefix
        self.H_pad, self.W_pad = H_pad, W_pad
        # number of frequencies in the half‑spectrum
        self.k_half = W_pad // 2 + 1
        # we'll treat M = k_half for clarity
        self.M = self.k_half
        self.rank = rank
        self.jitter = jitter
        self.joint = f"{prefix}_joint"

    def _build_lowrank(self):
        # joint dimension = real_hp + imag_hp + delta_alpha
        dim = 3 * self.M
        loc   = numpyro.param(f"{self.joint}_loc", jnp.zeros(dim))
        V_init= 0.05 * jax.random.normal(
                    numpyro.prng_key(), (dim, self.rank)
                ) / math.sqrt(dim)
        V     = numpyro.param(f"{self.joint}_V", V_init)
        d_raw = numpyro.param(f"{self.joint}_d_raw", -3.0 * jnp.ones(dim))
        log_tau = numpyro.param(f"{self.joint}_log_tau", jnp.array(0.0))
        diag  = jax.nn.softplus(d_raw) * jnp.exp(log_tau) + self.jitter
        cov   = V @ V.T + jnp.diag(diag)

        return numpyro.sample(
            self.joint,
            dist.MultivariateNormal(loc, covariance_matrix=cov),
            infer={"is_auxiliary": True},
        )

    def __call__(self, *args, **kwargs):
        # 1) draw the entire joint z = [real_hp; imag_hp; delta_alpha]
        z = self._build_lowrank()                # shape = (3*M,)

        # 2) split into three length‑M vectors
        real_hp, imag_hp, delta_alpha = jnp.split(
            z, [self.M, 2*self.M], axis=-1
        )
        # each is shape (M,)

        # 3) register them as Delta‑sites (guide overrides model)
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
        # exactly mirror the _build_lowrank and split logic
        dim = 3 * self.M
        loc      = params[f"{self.joint}_loc"]
        V        = params[f"{self.joint}_V"]
        d_raw    = params[f"{self.joint}_d_raw"]
        log_tau  = params[f"{self.joint}_log_tau"]
        diag     = jax.nn.softplus(d_raw) * jnp.exp(log_tau) + self.jitter
        cov      = V @ V.T + jnp.diag(diag)
        mvn      = dist.MultivariateNormal(loc, covariance_matrix=cov)

        # sample z with leading sample_shape
        z = mvn.sample(rng_key, sample_shape)  # shape = sample_shape + (3*M,)

        # split and reshape
        real_hp = z[..., : self.M]
        imag_hp = z[..., self.M : 2*self.M]
        delta   = z[..., 2*self.M : 3*self.M]

        return {
            f"{self.prefix}_delta_alpha": delta,
            f"{self.prefix}_real": real_hp,
            f"{self.prefix}_imag": imag_hp,
        }