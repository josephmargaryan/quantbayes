# tests/selftest_dp_kmeans.py
# ------------------------------------------------------------
# End-to-end (no pytest) for DP k-means from the assignment.
# - Synthetic Gaussian blobs, row clipping.
# - Runs different (sigma, sigma') settings.
# - Visualizations saved to tests_out/.
# ------------------------------------------------------------

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantbayes.dp.utils.clip import clip_rows_l2
from quantbayes.dp.models.kmeans_dp import DPkMeans, dp_kmeans_rho
from quantbayes.dp.accounting import zcdp_to_epsdelta

OUTDIR = "tests_out"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed=0):
    np.random.seed(seed)


def make_kmeans_data(n_per=250, d=5, k=3, sep=3.0, seed=7):
    rng = np.random.RandomState(seed)
    centers = rng.normal(size=(k, d))
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12) * sep
    X_list = []
    for i in range(k):
        X_i = centers[i] + rng.normal(scale=1.0, size=(n_per, d))
        X_list.append(X_i)
    X = np.vstack(X_list)
    rng.shuffle(X)
    return X


def inertia_to_centers(X, centers):
    dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return float(np.min(dists, axis=1).sum())


def plot_inertia_vs_sigma(sigmas, inertias, out_path):
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(sigmas, inertias, marker="o", linewidth=2)
    plt.xlabel("sigma = sigma'")
    plt.ylabel("Inertia (sum of squared distances)")
    plt.title("DP k-means: inertia vs sigma")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_scatter_first2(X, centers, out_path):
    # First two dimensions scatter (for a quick look)
    plt.figure(figsize=(5.2, 5.2))
    plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.6)
    plt.scatter(centers[:, 0], centers[:, 1], s=100, marker="x")
    plt.xlabel("x[0]")
    plt.ylabel("x[1]")
    plt.title("Data (first 2 dims) + final centers")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run_kmeans_suite():
    print("\n" + "=" * 80)
    print("DP k-means smoke test")
    print("=" * 80)
    ensure_dir(OUTDIR)

    X_raw = make_kmeans_data(n_per=250, d=5, k=3, sep=3.0, seed=7)
    X = clip_rows_l2(X_raw, 1.0)
    max_norm = float(np.linalg.norm(X, axis=1).max())
    print(f"Data: X shape={X.shape}, row norms max={max_norm:.3f}")

    configs = [
        {"sigma": 1.0, "sigma_prime": 1.0},
        {"sigma": 0.6, "sigma_prime": 0.6},
        {"sigma": 0.3, "sigma_prime": 0.3},
    ]
    k, t = 3, 8

    inertias = []
    eps_list = []
    sigmas_scalar = []

    for cfg in configs:
        sigma, sigma_p = cfg["sigma"], cfg["sigma_prime"]
        dpkm = DPkMeans(k=k, t=t, sigma=sigma, sigma_prime=sigma_p, seed=0)
        centers, transcript = dpkm.fit(X, return_transcript=True)
        inertia = inertia_to_centers(X, centers)
        inertias.append(inertia)
        sigmas_scalar.append(sigma)

        rho = dp_kmeans_rho(t=t, sigma=sigma, sigma_prime=sigma_p)
        eps = zcdp_to_epsdelta(rho, delta=1e-6)
        eps_list.append(eps)
        print(
            f"sigmas=({sigma:.2f},{sigma_p:.2f}) | inertia={inertia:.1f} | rho={rho:.3f} | eps(δ=1e-6)={eps:.3f}"
        )

        # Light scatter for first 2 dims (same axes for all runs)
        plot_scatter_first2(
            X, centers, os.path.join(OUTDIR, f"kmeans_scatter_sigma_{sigma:.2f}.png")
        )

    print("\nInertia vs. sigma:", [f"{v:.1f}" for v in inertias])
    print("Approx-DP eps vs. sigma:", [f"{v:.3f}" for v in eps_list])

    plot_inertia_vs_sigma(
        sigmas_scalar, inertias, os.path.join(OUTDIR, "kmeans_inertia_vs_sigma.png")
    )


if __name__ == "__main__":
    set_seed(0)
    run_kmeans_suite()
    print("\nKMeans suite done. Figures saved in:", OUTDIR)
