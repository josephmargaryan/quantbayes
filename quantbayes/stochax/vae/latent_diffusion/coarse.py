# quantbayes/stochax/vae/latent_diffusion/coarse.py
from __future__ import annotations

from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp


def ink_fraction_and_grad_01(
    x01: jnp.ndarray,
    *,
    thr: float = 0.35,
    temp: float = 0.08,
    eps: float = 1e-12,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Ink fraction on images in [0,1]. x01: (B,1,H,W) or (1,H,W).
    Returns:
      d: (B,)
      grad_x: same shape as x01
    """
    x = jnp.asarray(x01)
    if x.ndim == 3:
        x = x[None, ...]

    temp = jnp.maximum(jnp.asarray(temp, x.dtype), eps)
    u = (x - thr) / temp
    s = jax.nn.sigmoid(u)  # (B,1,H,W)

    d = jnp.mean(s, axis=(1, 2, 3))  # (B,)

    n_pix = x.shape[1] * x.shape[2] * x.shape[3]
    grad = s * (1.0 - s) * (1.0 / temp) * (1.0 / n_pix)

    return d, grad


def ink_fraction_01(
    x01: jnp.ndarray, *, thr: float = 0.35, temp: float = 0.08
) -> jnp.ndarray:
    d, _ = ink_fraction_and_grad_01(x01, thr=thr, temp=temp)
    return d


def gaussian_pdf(x: np.ndarray, mu: float, tau: float) -> np.ndarray:
    tau = float(max(tau, 1e-12))
    return (1.0 / (np.sqrt(2.0 * np.pi) * tau)) * np.exp(-0.5 * ((x - mu) / tau) ** 2)


def plot_ink_z_hist(
    *,
    z_ref: np.ndarray,
    z_base: np.ndarray,
    z_pk: np.ndarray,
    mu_z: float,
    tau_z: float,
    out_path,
    title: str,
    bins: int = 60,
):
    lo = min(z_ref.min(), z_base.min(), z_pk.min(), mu_z - 4 * tau_z)
    hi = max(z_ref.max(), z_base.max(), z_pk.max(), mu_z + 4 * tau_z)
    grid = np.linspace(lo, hi, 600)

    plt.figure(figsize=(10, 4))
    plt.hist(z_ref, bins=bins, density=True, alpha=0.25, label="π(z) (ref)")
    plt.hist(
        z_base,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="samples (base)",
    )
    plt.hist(
        z_pk,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="samples (PK)",
    )
    plt.plot(
        grid,
        gaussian_pdf(grid, mu_z, tau_z),
        linewidth=2,
        label="target p(z) (Gaussian)",
    )
    plt.title(title)
    plt.xlabel("z")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.show()


def finite_diff_score_from_hist(
    z: np.ndarray, grid: np.ndarray, smooth_sigma_bins: float = 2.0
):
    # histogram density
    hist, edges = np.histogram(
        z, bins=len(grid) - 1, range=(grid.min(), grid.max()), density=True
    )
    centers = 0.5 * (edges[:-1] + edges[1:])

    # gaussian smoothing in "bin units"
    ksize = int(max(5, 6 * smooth_sigma_bins)) | 1
    xs = np.arange(ksize) - ksize // 2
    kern = np.exp(-0.5 * (xs / smooth_sigma_bins) ** 2)
    kern = kern / np.sum(kern)

    smooth = np.convolve(hist, kern, mode="same")
    smooth = np.maximum(smooth, 1e-12)

    # score ≈ d/dz log p(z)
    logp = np.log(smooth)
    dz = centers[1] - centers[0]
    score = np.gradient(logp, dz)
    return centers, score


def plot_ref_score_check(z_ref: np.ndarray, score_net, out_path, title: str):
    # pick a grid over the support
    lo, hi = np.quantile(z_ref, [0.001, 0.999])
    lo -= 1.0
    hi += 1.0
    grid = np.linspace(lo, hi, 400)

    centers, fd_score = finite_diff_score_from_hist(z_ref, np.linspace(lo, hi, 401))

    # learned score
    import jax.numpy as jnp

    s = np.asarray(score_net(jnp.asarray(grid, dtype=jnp.float32)))  # (G,)

    plt.figure(figsize=(8, 4))
    plt.plot(centers, fd_score, linewidth=2, label="finite-diff (smoothed hist)")
    plt.plot(grid, s, linewidth=2, label="learned s_pi(z)")
    plt.title(title)
    plt.xlabel("z")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.show()
