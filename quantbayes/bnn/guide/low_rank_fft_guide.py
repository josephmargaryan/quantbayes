import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer.autoguide import AutoGuide
import math

__all__ = [
    "LowRankFFTGuide2d",
    "LowRankFFTGuide",
    "LowRankNonStatFFTGuide2d",
    "LowRankNonStatGuide",
    "Patch1DAutoGuide",
    "Patch2DAutoGuide",
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


def _halfplane_to_full(
    real_hp: jnp.ndarray, imag_hp: jnp.ndarray, H_pad: int, W_pad: int
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
    u = jnp.arange(H_pad)[:, None]  # shape (H_pad,1)
    v = jnp.arange(1, W_half)[None, :]  # shape (1,W_half-1)
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


class LowRankFFTGuide2d(AutoGuide):
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
        z = self._build_lowrank()  # shape (2*M,)
        real_hp, imag_hp = jnp.split(z, 2, axis=-1)  # two arrays of shape (M,)
        real_hp = real_hp.reshape(self.H_pad, self.W_half)
        imag_hp = imag_hp.reshape(self.H_pad, self.W_half)

        # mirror to full
        real_full, imag_full = _halfplane_to_full(
            real_hp, imag_hp, self.H_pad, self.W_pad
        )

        # register Delta sites
        numpyro.sample(f"{self.prefix}_real", dist.Delta(real_full).to_event(2))
        numpyro.sample(f"{self.prefix}_imag", dist.Delta(imag_full).to_event(2))

        return {
            f"{self.prefix}_real": real_full,
            f"{self.prefix}_imag": imag_full,
        }

    def sample_posterior(self, rng_key, params, sample_shape=()):
        dim = 2 * self.M
        loc = params[f"{self.joint}_loc"]
        V = params[f"{self.joint}_V"]
        d_raw = params[f"{self.joint}_d_raw"]
        log_tau = params[f"{self.joint}_log_tau"]
        diag = jax.nn.softplus(d_raw) * jnp.exp(log_tau) + self.jitter
        cov = V @ V.T + jnp.diag(diag)

        mvn = dist.MultivariateNormal(loc, covariance_matrix=cov)
        z = mvn.sample(rng_key, sample_shape)  # (..., 2*M)

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
        # total half‑spectrum length
        self.k_half = padded_dim // 2 + 1
        # how many active frequencies your layer actually samples
        self.K = self.k_half if (K is None or K > self.k_half) else K

        # dimension of our low‑rank joint = real + imag = 2*K
        self.M = self.K
        self.rank = rank
        self.jitter = jitter
        self.joint = f"{prefix}_joint"

    def _build_lowrank(self):
        dim = 2 * self.M
        # mean vector
        loc = numpyro.param(f"{self.joint}_loc", jnp.zeros(dim))
        # low‑rank factor V
        V_init = (
            0.05
            * jax.random.normal(numpyro.prng_key(), (dim, self.rank))
            / math.sqrt(dim)
        )
        V = numpyro.param(f"{self.joint}_V", V_init)
        # diagonal variances
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
        # 1) draw our low‑rank joint z of length 2*K
        z = self._build_lowrank()  # shape (2*K,)
        real_hp, imag_hp = jnp.split(z, 2, axis=-1)  # each shape (K,)

        # 2) Hermitian constraints on the half‑spectrum
        #    – DC imagery = 0
        imag_hp = imag_hp.at[0].set(0.0)
        #    – Nyquist imagery = 0 if we actually model it (i.e. K == full half)
        if (self.padded_dim % 2 == 0) and (self.K == self.k_half):
            imag_hp = imag_hp.at[-1].set(0.0)

        # 3) override exactly your two model sites of shape (K,)
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
        # reconstruct posterior covariance
        dim = 2 * self.M
        loc = params[f"{self.joint}_loc"]
        V = params[f"{self.joint}_V"]
        d_raw = params[f"{self.joint}_d_raw"]
        log_tau = params[f"{self.joint}_log_tau"]
        diag = jax.nn.softplus(d_raw) * jnp.exp(log_tau) + self.jitter
        cov = V @ V.T + jnp.diag(diag)

        mvn = dist.MultivariateNormal(loc, covariance_matrix=cov)
        z = mvn.sample(rng_key, sample_shape)  # (..., 2*K)

        # split into real & imag half‑spectrum
        real_hp = z[..., : self.M]
        imag_hp = z[..., self.M :]

        # enforce Hermitian
        imag_hp = imag_hp.at[..., 0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.K == self.k_half):
            imag_hp = imag_hp.at[..., -1].set(0.0)

        return {
            f"{self.prefix}_real": real_hp,
            f"{self.prefix}_imag": imag_hp,
        }


class LowRankNonStatFFTGuide2d(AutoGuide):
    """
    Low‑rank Gaussian guide over:
      - a coarse Δα‑map of shape (gh, gw),
      - the 2D half‑plane real and imag coefficients
    for a non‑stationary 2D spectral power‑law conv.
    Joint latent dim = G + 2*M, where
      M = H_pad * (W_pad//2 + 1),  G = gh * gw.
    """

    def __init__(
        self,
        model,
        prefix: str,
        H_pad: int,
        W_pad: int,
        alpha_coarse_shape=(8, 8),
        rank: int = 8,
        jitter: float = 1e-4,
    ):
        super().__init__(model)
        self.prefix = prefix
        self.H_pad, self.W_pad = H_pad, W_pad
        self.W_half = W_pad // 2 + 1
        self.M = H_pad * self.W_half

        self.gh, self.gw = alpha_coarse_shape
        self.G = self.gh * self.gw

        self.rank = rank
        self.jitter = jitter
        self.joint = f"{prefix}_joint"

    def _build_lowrank(self):
        dim = 2 * self.M + self.G
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
        # 1) joint draw of size 2*M + G
        z = self._build_lowrank()  # shape (2*M + G,)
        # 2) split into [real_hp; imag_hp; delta_flat]
        z_f, delta_flat = jnp.split(z, [2 * self.M], axis=-1)
        f_real, f_imag = jnp.split(z_f, 2, axis=-1)

        # reshape half-plane
        real_hp = f_real.reshape(self.H_pad, self.W_half)
        imag_hp = f_imag.reshape(self.H_pad, self.W_half)
        delta_map = delta_flat.reshape(self.gh, self.gw)

        # expose the Δα map
        numpyro.sample(
            f"{self.prefix}_delta_alpha_map",
            dist.Delta(delta_map).to_event(2),
        )

        # enforce Hermitian and build full 2D grid
        real_full, imag_full = _halfplane_to_full(
            real_hp, imag_hp, self.H_pad, self.W_pad
        )

        numpyro.sample(
            f"{self.prefix}_real",
            dist.Delta(real_full).to_event(2),
        )
        numpyro.sample(
            f"{self.prefix}_imag",
            dist.Delta(imag_full).to_event(2),
        )

        return {
            f"{self.prefix}_delta_alpha_map": delta_map,
            f"{self.prefix}_real": real_full,
            f"{self.prefix}_imag": imag_full,
        }

    def sample_posterior(self, rng_key, params, sample_shape=()):
        # reconstruct posterior MVN of dim=2*M+G
        dim = 2 * self.M + self.G
        loc = params[f"{self.joint}_loc"]
        V = params[f"{self.joint}_V"]
        d_raw = params[f"{self.joint}_d_raw"]
        log_tau = params[f"{self.joint}_log_tau"]
        diag = jax.nn.softplus(d_raw) * jnp.exp(log_tau) + self.jitter
        cov = V @ V.T + jnp.diag(diag)

        mvn = dist.MultivariateNormal(loc, covariance_matrix=cov)
        z = mvn.sample(rng_key, sample_shape)  # (..., 2*M+G)

        # split out
        z_f, delta_flat = jnp.split(z, [2 * self.M], axis=-1)
        f_real, f_imag = jnp.split(z_f, 2, axis=-1)

        # reshape
        real_hp = f_real.reshape(sample_shape + (self.H_pad, self.W_half))
        imag_hp = f_imag.reshape(sample_shape + (self.H_pad, self.W_half))
        delta_map = delta_flat.reshape(sample_shape + (self.gh, self.gw))

        # Hermitian mirror
        real_full, imag_full = jax.vmap(
            lambda rhp, ihp: _halfplane_to_full(rhp, ihp, self.H_pad, self.W_pad)
        )(real_hp, imag_hp)

        return {
            f"{self.prefix}_delta_alpha_map": delta_map,
            f"{self.prefix}_real": real_full,
            f"{self.prefix}_imag": imag_full,
        }


class LowRankNonStatGuide(AutoGuide):
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
        self.k_half = padded_dim // 2 + 1  # length of half‑spectrum
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
        # draw joint z = [real_hp; imag_hp; delta_alpha]
        z = self._build_lowrank()  # shape (3*M,)

        # split into three M‑vectors
        real_hp, imag_hp, delta_alpha = jnp.split(z, [self.M, 2 * self.M], axis=-1)

        # enforce Hermitian constraints
        imag_hp = imag_hp.at[0].set(0.0)
        if self.padded_dim % 2 == 0:
            imag_hp = imag_hp.at[-1].set(0.0)

        # register as Delta sites
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
        # rebuild posterior MVN
        dim = 3 * self.M
        loc = params[f"{self.joint}_loc"]
        V = params[f"{self.joint}_V"]
        d_raw = params[f"{self.joint}_d_raw"]
        log_tau = params[f"{self.joint}_log_tau"]
        diag = jax.nn.softplus(d_raw) * jnp.exp(log_tau) + self.jitter
        cov = V @ V.T + jnp.diag(diag)

        mvn = dist.MultivariateNormal(loc, covariance_matrix=cov)
        z = mvn.sample(rng_key, sample_shape)  # (..., 3*M)

        # split and enforce
        real_hp, imag_hp, delta_alpha = jnp.split(z, [self.M, 2 * self.M], axis=-1)
        imag_hp = imag_hp.at[..., 0].set(0.0)
        if self.padded_dim % 2 == 0:
            imag_hp = imag_hp.at[..., -1].set(0.0)

        return {
            f"{self.prefix}_delta_alpha": delta_alpha,
            f"{self.prefix}_real": real_hp,
            f"{self.prefix}_imag": imag_hp,
        }


# -----------------------------------------------------------------------------
# 1D Patch‑Wise Spectral‑Mixture AutoGuide
# -----------------------------------------------------------------------------


class Patch1DAutoGuide(AutoGuide):
    """
    AutoGuide for PatchWiseSpectralMixture1DLayer:
      - n_features: total input dim
      - patch_size: sub‑sequence length
      - n_mixtures: number of Gaussians per patch
    """

    def __init__(
        self, model, prefix, n_features, patch_size, n_mixtures=3, jitter=1e-6
    ):
        super().__init__(model)
        assert (
            n_features % patch_size == 0
        ), "n_features must be divisible by patch_size"
        self.prefix = prefix
        self.N = n_features
        self.P = patch_size
        self.M = n_mixtures
        self.n_patches = self.N // self.P
        self.jitter = jitter
        # precompute frequency bins for real FFT
        self.freqs = jnp.fft.rfftfreq(self.P)

    def __call__(self, X, y=None):
        # 1) define variational parameters (loc & scale) for logits, raw_mu, raw_sigma
        def make_loc_scale(name, shape):
            loc = numpyro.param(f"{self.prefix}_{name}_loc", jnp.zeros(shape))
            scale = numpyro.param(
                f"{self.prefix}_{name}_scale",
                0.1 * jnp.ones(shape),
                constraint=constraints.positive,
            )
            return loc, scale

        loc_w, scale_w = make_loc_scale("logits_w", (self.n_patches, self.M))
        loc_mu, scale_mu = make_loc_scale("raw_mu", (self.n_patches, self.M))
        loc_sig, scale_sig = make_loc_scale("raw_sigma", (self.n_patches, self.M))

        # 2) sample them as factorized Gaussians
        logits_w = numpyro.sample(
            f"{self.prefix}_logits_w", dist.Normal(loc_w, scale_w).to_event(2)
        )
        raw_mu = numpyro.sample(
            f"{self.prefix}_raw_mu", dist.Normal(loc_mu, scale_mu).to_event(2)
        )
        raw_sig = numpyro.sample(
            f"{self.prefix}_raw_sigma", dist.Normal(loc_sig, scale_sig).to_event(2)
        )

        # 3) deterministic transform into PSD S per patch
        w = jax.nn.softmax(logits_w, axis=-1)  # (n_patches, M)
        mu = 0.5 * jax.nn.sigmoid(raw_mu)  # (n_patches, M)
        sigma = jax.nn.softplus(raw_sig) + self.jitter  # (n_patches, M)

        def make_S(p):
            # for patch p, compute mixture sum over frequencies
            freq = self.freqs[None, :]  # (1, P//2+1)
            mu_p = mu[p : p + 1, :]  # (1, M)
            sig_p = sigma[p : p + 1, :]  # (1, M)
            diffs1 = ((freq - mu_p[..., None]) / sig_p[..., None]) ** 2
            diffs2 = ((freq + mu_p[..., None]) / sig_p[..., None]) ** 2
            gauss = jnp.exp(-0.5 * diffs1) + jnp.exp(-0.5 * diffs2)
            return (w[p : p + 1, :, None] * gauss).sum(axis=1).squeeze(0) + self.jitter

        S = jax.vmap(make_S)(jnp.arange(self.n_patches))  # (n_patches, P//2+1)

        # 4) register S as a single Delta site for use in the model
        numpyro.sample(f"{self.prefix}_S", dist.Delta(S).to_event(2))
        return {f"{self.prefix}_S": S}

    def sample_posterior(self, rng_key, params, sample_shape=()):
        # Split RNG for the three parameter groups
        key_w, key_mu, key_sig = jax.random.split(rng_key, 3)

        # helper to draw from the learned normals
        def draw(name, key):
            loc = params[f"{self.prefix}_{name}_loc"]
            scale = params[f"{self.prefix}_{name}_scale"]
            return dist.Normal(loc, scale).sample(key, sample_shape)

        # sample unconstrained latent variables
        logits_w = draw("logits_w", key_w)  # (..., n_patches, M)
        raw_mu = draw("raw_mu", key_mu)  # (..., n_patches, M)
        raw_sig = draw("raw_sigma", key_sig)  # (..., n_patches, M)

        # same transforms as in __call__
        w = jax.nn.softmax(logits_w, axis=-1)
        mu = 0.5 * jax.nn.sigmoid(raw_mu)
        sigma = jax.nn.softplus(raw_sig) + self.jitter

        def make_S(p):
            freq = self.freqs[None, :]
            mu_p = mu[..., p : p + 1, :]
            sig_p = sigma[..., p : p + 1, :]
            diffs1 = ((freq - mu_p[..., None]) / sig_p[..., None]) ** 2
            diffs2 = ((freq + mu_p[..., None]) / sig_p[..., None]) ** 2
            gauss = jnp.exp(-0.5 * diffs1) + jnp.exp(-0.5 * diffs2)
            return (w[..., p : p + 1, :, None] * gauss).sum(axis=-2).squeeze(
                -2
            ) + self.jitter

        # vectorize over patch index dimension
        idxs = jnp.arange(self.n_patches)
        S = jax.vmap(make_S, in_axes=0, out_axes=0)(idxs)  # (..., n_patches, P//2+1)
        return {f"{self.prefix}_S": S}


# -----------------------------------------------------------------------------
# 2D Patch‑Wise Spectral‑Mixture AutoGuide
# -----------------------------------------------------------------------------


class Patch2DAutoGuide(AutoGuide):
    """
    AutoGuide for PatchWiseSpectralMixture2DLayer:
      - H,W: full image size
      - ph,pw: patch dims
      - n_mixtures: # mixtures per patch
    """

    def __init__(self, model, prefix, H, W, ph, pw, n_mixtures=3, jitter=1e-4):
        super().__init__(model)
        assert H % ph == 0 and W % pw == 0, "Image dims must divide evenly into patches"
        self.prefix = prefix
        self.H, self.W, self.ph, self.pw, self.M = H, W, ph, pw, n_mixtures
        self.nh, self.nw = H // ph, W // pw
        self.n_patches = self.nh * self.nw
        self.jitter = jitter
        # precompute 2D frequency grid once per patch
        fy = jnp.fft.fftfreq(self.ph)
        fx = jnp.fft.fftfreq(self.pw)
        self.FY, self.FX = jnp.meshgrid(fy, fx, indexing="ij")  # (ph, pw)

    def __call__(self, X, y=None):
        # param factory
        def make_loc_scale(name, shape):
            loc = numpyro.param(f"{self.prefix}_{name}_loc", jnp.zeros(shape))
            scale = numpyro.param(
                f"{self.prefix}_{name}_scale",
                0.1 * jnp.ones(shape),
                constraint=constraints.positive,
            )
            return loc, scale

        loc_w, scale_w = make_loc_scale("logits_w", (self.n_patches, self.M))
        loc_mu, scale_mu = make_loc_scale("raw_mu", (self.n_patches, self.M, 2))
        loc_sig, scale_sig = make_loc_scale("raw_sigma", (self.n_patches, self.M, 2))

        logits_w = numpyro.sample(
            f"{self.prefix}_logits_w", dist.Normal(loc_w, scale_w).to_event(2)
        )
        raw_mu = numpyro.sample(
            f"{self.prefix}_raw_mu", dist.Normal(loc_mu, scale_mu).to_event(3)
        )
        raw_sig = numpyro.sample(
            f"{self.prefix}_raw_sigma", dist.Normal(loc_sig, scale_sig).to_event(3)
        )

        w = jax.nn.softmax(logits_w, axis=-1)  # (n_patches, M)
        mu = 0.5 * jax.nn.sigmoid(raw_mu)  # (n_patches, M, 2)
        sigma = jax.nn.softplus(raw_sig) + self.jitter  # (n_patches, M, 2)

        def make_S(idx):
            w_i, mu_i, sig_i = w[idx], mu[idx], sigma[idx]
            # build Gaussian over 2D freq grid
            FYi = self.FY[None, :, :]  # (1, ph, pw)
            FXi = self.FX[None, :, :]
            mu_y = mu_i[..., 0:1]  # (M,1)
            mu_x = mu_i[..., 1:2]
            sig_y = sig_i[..., 0:1]
            sig_x = sig_i[..., 1:2]
            dyi = (FYi - mu_y[..., None, None]) / sig_y[..., None, None]
            dxi = (FXi - mu_x[..., None, None]) / sig_x[..., None, None]
            ga1 = jnp.exp(-0.5 * (dyi**2 + dxi**2))
            # mirror at negative freqs
            dyi2 = (FYi + mu_y[..., None, None]) / sig_y[..., None, None]
            dxi2 = (FXi + mu_x[..., None, None]) / sig_x[..., None, None]
            ga2 = jnp.exp(-0.5 * (dyi2**2 + dxi2**2))
            return (w_i[..., None, None, None] * (ga1 + ga2)).sum(axis=0) + self.jitter

        S = jax.vmap(make_S)(jnp.arange(self.n_patches))  # (n_patches, ph, pw)
        numpyro.sample(f"{self.prefix}_S", dist.Delta(S).to_event(3))
        return {f"{self.prefix}_S": S}

    def sample_posterior(self, rng_key, params, sample_shape=()):
        # split rng keys
        key_w, key_mu, key_sig = jax.random.split(rng_key, 3)

        def draw(name, key, event_dim):
            loc = params[f"{self.prefix}_{name}_loc"]
            scale = params[f"{self.prefix}_{name}_scale"]
            return (
                dist.Normal(loc, scale)
                .sample(key, sample_shape)
                .reshape(sample_shape + loc.shape)
            )

        logits_w = draw("logits_w", key_w, event_dim=2)  # (..., n_patches, M)
        raw_mu = draw("raw_mu", key_mu, event_dim=3)  # (..., n_patches, M, 2)
        raw_sig = draw("raw_sigma", key_sig, event_dim=3)

        w = jax.nn.softmax(logits_w, axis=-1)
        mu = 0.5 * jax.nn.sigmoid(raw_mu)
        sigma = jax.nn.softplus(raw_sig) + self.jitter

        def make_S(idx):
            w_i, mu_i, sig_i = w[..., idx, :], mu[..., idx, :, :], sigma[..., idx, :, :]
            FYi = self.FY[None, :, :]
            FXi = self.FX[None, :, :]
            dy = (FYi - mu_i[..., 0:1, None, None]) / sig_i[..., 0:1, None, None]
            dx = (FXi - mu_i[..., 1:2, None, None]) / sig_i[..., 1:2, None, None]
            ga1 = jnp.exp(-0.5 * (dy**2 + dx**2))
            dy2 = (FYi + mu_i[..., 0:1, None, None]) / sig_i[..., 0:1, None, None]
            dx2 = (FXi + mu_i[..., 1:2, None, None]) / sig_i[..., 1:2, None, None]
            ga2 = jnp.exp(-0.5 * (dy2**2 + dx2**2))
            return (w_i[..., None, None] * (ga1 + ga2)).sum(axis=-1) + self.jitter

        idxs = jnp.arange(self.n_patches)
        S = jax.vmap(make_S, in_axes=0, out_axes=0)(idxs)  # (..., n_patches, ph, pw)
        return {f"{self.prefix}_S": S}
