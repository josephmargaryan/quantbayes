"""
spectral_diagnostics.py

Module providing diagnostic plots for Numpyro SVI spectral-guides.

Usage:

diag = SpectralDiagnostics(
    model=model,
    guide=guide,
    params=params,
    X=X_train, y=y_train,
    H_pad=28, W_pad=28,
    prefix="mix"
)
diag.run_all()
print("Import and run SpectralDiagnostics from your pipeline.")
"""

import re
import logging
from typing import Any, Dict, Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from numpyro.infer import Trace_ELBO
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def find_param_key(params: Dict[str, Any], pattern: str) -> str:
    """
    Find exactly one key in `params` matching the regex `pattern`.

    Raises:
        KeyError: if none or multiple matches are found.
    """
    matches = [k for k in params if re.fullmatch(pattern, k)]
    if not matches:
        msg = f"No params key matching /{pattern}/"
        logger.error(msg)
        raise KeyError(msg)
    if len(matches) > 1:
        msg = f"Multiple params keys matching /{pattern}/: {matches}"
        logger.error(msg)
        raise KeyError(msg)
    return matches[0]


def get_alpha(params: Dict[str, Any], prefix: str) -> float:
    """
    Extract the posterior mean of the alpha hyperparameter.
    Expects a key like "{prefix}_alpha_auto_loc" in params.
    """
    key_pattern = rf"{prefix}_alpha_auto_loc"
    key = find_param_key(params, key_pattern)
    return float(params[key])


def reconstruct_fft_grid(z: np.ndarray, H_pad: int, W_pad: int) -> np.ndarray:
    """
    Rebuild full Hermitian FFT grid from vector z of length 2*M.
    """
    M = z.shape[-1] // 2
    real_flat, imag_flat = z[:M], z[M:]
    W_half = W_pad // 2 + 1
    real_hp = real_flat.reshape(H_pad, W_half)
    imag_hp = imag_flat.reshape(H_pad, W_half)

    real_full = np.zeros((H_pad, W_pad))
    imag_full = np.zeros((H_pad, W_pad))
    real_full[:, :W_half] = real_hp
    imag_full[:, :W_half] = imag_hp

    # Hermitian symmetry
    u = np.arange(H_pad)[:, None]
    v = np.arange(1, W_half)[None, :]
    u_conj = (-u) % H_pad
    v_conj = (-v) % W_pad
    real_full[u_conj, v_conj] = real_hp[u_conj, v]
    imag_full[u_conj, v_conj] = -imag_hp[u_conj, v]

    return real_full + 1j * imag_full


def get_spectral_guide(guide: Any) -> Any:
    """
    Find the LowRankSpectral2DGuide sub-guide by looking for .joint.
    """
    if hasattr(guide, "_guides"):
        return next(g for g in guide if hasattr(g, "joint"))
    return guide


def plot_posterior_mean_kernel(
    guide: Any, params: Dict[str, Any], H_pad: int, W_pad: int
) -> None:
    """
    Plot the posterior mean spatial kernel k(x,y).
    """
    spec = get_spectral_guide(guide)
    joint = spec.joint
    loc = np.array(params[f"{joint}_loc"])
    K_mean = reconstruct_fft_grid(loc, H_pad, W_pad)
    k = np.fft.ifft2(K_mean).real
    k = np.fft.fftshift(k)

    plt.figure(figsize=(4, 4))
    plt.imshow(k, interpolation="nearest")
    plt.title(r"Posterior mean kernel $\bar{k}(x,y)$")
    plt.axis("off")
    plt.colorbar(label="weight")
    plt.show()


def plot_log_magnitude_spectrum(
    guide: Any, params: Dict[str, Any], H_pad: int, W_pad: int
) -> None:
    """
    Plot the log-magnitude of the posterior mean spectrum.
    """
    spec = get_spectral_guide(guide)
    joint = spec.joint
    loc = np.array(params[f"{joint}_loc"])
    K_mean = reconstruct_fft_grid(loc, H_pad, W_pad)
    mag = np.log(np.abs(K_mean) + 1e-6)
    mag = np.fft.fftshift(mag)

    plt.figure(figsize=(4, 4))
    plt.imshow(mag, interpolation="nearest")
    plt.title(r"Log‑magnitude of $\mathrm{E}[K(u,v)]$")
    plt.axis("off")
    plt.colorbar(label="log |K(u,v)|")
    plt.show()


def sample_and_plot_kernels(
    guide: Any, params: Dict[str, Any], H_pad: int, W_pad: int, n_samples: int = 5
) -> None:
    """
    Sample MVN posterior and plot spatial kernel samples.
    """
    spec = get_spectral_guide(guide)
    joint = spec.joint
    loc = jnp.array(params[f"{joint}_loc"])
    V = jnp.array(params[f"{joint}_V"])
    d_raw = jnp.array(params[f"{joint}_d_raw"])
    diag = jax.nn.softplus(d_raw) + 1e-5
    cov = V @ V.T + jnp.diag(diag)

    key = jr.PRNGKey(42)
    L = jnp.linalg.cholesky(cov)
    eps = jax.random.normal(key, (n_samples, loc.size))
    zs = np.array(loc + (eps @ L.T))

    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 2, 2))
    for i, z in enumerate(zs):
        K_i = reconstruct_fft_grid(z, H_pad, W_pad)
        k_i = np.fft.ifft2(K_i).real
        k_i = np.fft.fftshift(k_i)
        ax = axes[i]
        ax.imshow(k_i, interpolation="nearest")
        ax.set_title(f"sample {i+1}")
        ax.axis("off")
    plt.suptitle("Posterior samples of spatial kernel")
    plt.show()


def plot_smoothed_psd(
    guide: Any, params: Dict[str, Any], H_pad: int, W_pad: int, radial_bins: int = 50
) -> None:
    """
    Plot smoothed posterior power spectral density.
    """
    spec = get_spectral_guide(guide)
    joint = spec.joint
    loc = np.array(params[f"{joint}_loc"])
    K_mean = reconstruct_fft_grid(loc, H_pad, W_pad)
    U, Vv = np.meshgrid(np.arange(H_pad), np.arange(W_pad), indexing="ij")
    radii = np.sqrt(U**2 + Vv**2).ravel()
    mags = np.abs(K_mean).ravel()
    edges = np.linspace(0, radii.max(), radial_bins + 1)
    bins = np.clip(np.digitize(radii, edges) - 1, 0, radial_bins - 1)
    sums = np.bincount(bins, mags, minlength=radial_bins)
    counts = np.bincount(bins, minlength=radial_bins)
    profile = sums / counts
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure()
    plt.plot(centers, profile, lw=2)
    plt.xlabel("Radial frequency r")
    plt.ylabel("Avg. |K(u,v)|")
    plt.title("Smoothed posterior PSD")
    plt.show()


def plot_prior_vs_posterior_psd_2d(
    H_pad: int,
    W_pad: int,
    params: Dict[str, Any],
    guide: Any,
    prefix: str,
    num_mc: int = 200,
    radial_bins: int = 50,
) -> None:
    """
    Compare prior vs posterior PSD (radial average).
    """
    alpha = get_alpha(params, prefix)
    u = jnp.arange(H_pad)
    v = jnp.arange(W_pad)
    UU, VV = jnp.meshgrid(u, v, indexing="ij")
    R = jnp.sqrt(UU**2 + VV**2)
    prior = (1.0 / jnp.sqrt(1.0 + R**alpha)).ravel()

    coords = jnp.stack(jnp.meshgrid(u, v, indexing="ij"), axis=-1).reshape(-1, 2)
    radii = jnp.linalg.norm(coords, axis=-1)
    edges = jnp.linspace(0, float(radii.max()), radial_bins + 1)
    bins = jnp.clip(jnp.digitize(radii, edges) - 1, 0, radial_bins - 1)
    counts = jnp.bincount(bins, length=radial_bins)
    sums = jnp.bincount(bins, prior, length=radial_bins)
    prior_radial = sums / counts

    spec = get_spectral_guide(guide)
    joint = spec.joint
    loc = jnp.array(params[f"{joint}_loc"])
    V = jnp.array(params[f"{joint}_V"])
    d_raw = jnp.array(params[f"{joint}_d_raw"])
    diag = jax.nn.softplus(d_raw) + 1e-5
    cov = V @ V.T + jnp.diag(diag)

    L = jnp.linalg.cholesky(cov)
    key = jr.PRNGKey(0)
    eps = jax.random.normal(key, (num_mc, loc.size))
    zs = loc + eps @ L.T

    fft_stds = []
    for i in range(num_mc):
        z = np.array(zs[i])
        K = reconstruct_fft_grid(z, H_pad, W_pad)
        fft_stds.append(np.abs(K).ravel())
    fft_stds = jnp.stack(fft_stds)

    post_means = jnp.mean(fft_stds, axis=0)
    post_sums = jnp.bincount(bins, post_means, length=radial_bins)
    post_radial = post_sums / counts

    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.figure()
    plt.plot(centers, prior_radial, label="Prior σ(r)")
    plt.plot(centers, post_radial, label="Posterior σ(r)")
    plt.xlabel("Radial frequency")
    plt.ylabel("Std of coefficient")
    plt.legend()
    plt.title("Prior vs Posterior PSD (radial average)")
    plt.show()


def plot_cov_eigenspectrum(
    guide: Any, params: Dict[str, Any], prefix: str, eps: float = 1e-6
) -> None:
    """
    Plot singular values of the low-rank V matrix in the spectral guide.
    """
    spec = get_spectral_guide(guide)
    joint = spec.joint
    V_key = find_param_key(params, rf"{joint}_V")
    V = np.array(params[V_key])
    s = np.linalg.svd(V, compute_uv=False)

    plt.figure()
    if np.all(s <= 0):
        plt.plot(s, "o-", label="singular values")
        plt.yscale("linear")
        plt.title("Eigenspectrum (all zeros — linear scale)")
    else:
        plt.semilogy(s + eps, "o-", label="singular values")
        plt.title("Low-rank guide: top singular values of V (log scale)")
    plt.xlabel("Mode index")
    plt.ylabel("Singular value")
    plt.legend()
    plt.show()


def plot_gradient_variance(
    model: Any,
    guide: Any,
    params: Dict[str, Any],
    X: Any,
    y: Optional[Any] = None,
    num_mc: int = 20,
) -> None:
    """
    Compute and plot histogram of SVI gradient norms.
    """

    def _elbo(p, m, g, Xb, yb, key):
        return (
            Trace_ELBO().loss(key, p, m, g, Xb, yb)
            if yb is not None
            else Trace_ELBO().loss(key, p, m, g, Xb)
        )

    grad_fn = jax.grad(_elbo)
    keys = jr.split(jr.PRNGKey(0), num_mc)
    norms = []
    for k in keys:
        g = grad_fn(params, model, guide, X, y, k)
        flat_g, _ = ravel_pytree(g)
        norms.append(jnp.linalg.norm(flat_g))
    norms = np.array(norms)

    plt.figure()
    plt.hist(norms, bins=10, density=True)
    plt.xlabel("Gradient norm")
    plt.ylabel("Density")
    plt.title("Histogram of SVI gradient norms")
    plt.show()
    logger.info(f"Mean norm: {norms.mean():.3f}, Var: {norms.var():.3f}")


def plot_loss_landscape_2d(
    model: Any,
    guide: Any,
    params: Dict[str, Any],
    X: Any,
    y: Optional[Any],
    prefix: str,
    grid: int = 21,
    span: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice the ELBO loss by perturbing the guide's joint_loc along its top two singular vectors.
    Returns:
        loss_grid: (grid, grid) array of ELBO losses
        scales: 1D array of perturbation scales
    """
    spec = get_spectral_guide(guide)
    joint = spec.joint
    loc_key = find_param_key(params, rf"{joint}_loc")
    V_key = find_param_key(params, rf"{joint}_V")

    loc = params[loc_key]
    V = params[V_key]
    U, S, _ = jnp.linalg.svd(V, full_matrices=False)
    dir1, dir2 = U[:, 0], U[:, 1]

    scales = np.linspace(-span, span, grid)
    loss_grid = np.zeros((grid, grid))
    base_key = jr.PRNGKey(0)
    lebo = Trace_ELBO()

    for i, a in enumerate(scales):
        for j, b in enumerate(scales):
            new_loc = loc + a * dir1 + b * dir2
            new_params = dict(params)
            new_params[loc_key] = new_loc
            loss_grid[i, j] = (
                lebo.loss(base_key, new_params, model, guide, X, y)
                if y is not None
                else lebo.loss(base_key, new_params, model, guide, X)
            )

    # 2D contour
    plt.figure(figsize=(5, 4))
    cs = plt.contourf(scales, scales, loss_grid, levels=30)
    plt.colorbar(cs, label="ELBO loss")
    plt.xlabel("mode 1")
    plt.ylabel("mode 2")
    plt.title("ELBO Landscape (joint_loc modes)")
    plt.show()

    # 3D surface
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    Xg, Yg = np.meshgrid(scales, scales)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xg, Yg, loss_grid, cmap="viridis", edgecolor="none", alpha=0.8)
    ax.set_xlabel("mode 1")
    ax.set_ylabel("mode 2")
    ax.set_zlabel("ELBO loss")
    plt.title("3D Loss Surface")
    plt.show()

    return loss_grid, scales


class SpectralDiagnostics:
    """
    High-level driver to run all spectral diagnostics with a single prefix.
    """

    def __init__(
        self,
        model: Any,
        guide: Any,
        params: Dict[str, Any],
        X: Any,
        y: Optional[Any],
        H_pad: int,
        W_pad: int,
        prefix: str,
    ):
        self.model = model
        self.guide = guide
        self.params = params
        self.X = X
        self.y = y
        self.H = H_pad
        self.W = W_pad
        self.prefix = prefix

    def run_all(self) -> None:
        # spatial kernel diagnostics
        plot_posterior_mean_kernel(self.guide, self.params, self.H, self.W)
        plot_log_magnitude_spectrum(self.guide, self.params, self.H, self.W)
        sample_and_plot_kernels(self.guide, self.params, self.H, self.W)
        plot_smoothed_psd(self.guide, self.params, self.H, self.W)
        # radial PSD
        plot_prior_vs_posterior_psd_2d(
            self.H, self.W, self.params, self.guide, self.prefix
        )
        # covariance eigenspectrum
        plot_cov_eigenspectrum(self.guide, self.params, self.prefix)
        # gradient variance
        plot_gradient_variance(self.model, self.guide, self.params, self.X, self.y)
        # loss landscape
        plot_loss_landscape_2d(
            self.model, self.guide, self.params, self.X, self.y, self.prefix
        )
