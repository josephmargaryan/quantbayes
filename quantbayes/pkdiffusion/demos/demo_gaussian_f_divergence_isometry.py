# quantbayes/pkdiffusion/demos/demo_gaussian_f_divergence_isometry.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from quantbayes.pkdiffusion.gaussian_pk import (
    GaussianND,
    pk_update_linear_gaussian,
    marginal_1d,
    sample_gaussian,
)
from quantbayes.pkdiffusion.metrics import kl_gaussian_1d, kl_gaussian_nd
from quantbayes.pkdiffusion.f_divergences import (
    hellinger2_gaussian_nd,
    chi2_divergence_mc_gaussian,
)

OUT_DIR = Path("reports/pkdiffusion/gaussian_f_divergence_isometry")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log_gaussian_nd(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    d = mean.size

    xc = x - mean[None, :]
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, xc.T)  # (d,n)
    quad = np.sum(sol * sol, axis=0)  # (n,)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


def log_gaussian_1d(y: np.ndarray, m: float, v: float) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    return -0.5 * (np.log(2.0 * np.pi * v) + (y - m) ** 2 / v)


def main():
    rng = np.random.default_rng(0)

    # Prior in X-space
    prior = GaussianND(mean=np.zeros(2), cov=np.eye(2))

    # Coarse y = a^T x
    a = np.array([1.0, 0.7], dtype=float)
    a = a / np.linalg.norm(a)
    py_m, py_v = marginal_1d(prior, a)

    # Two evidences q1, q2 on y
    q1_m, q1_v = 1.0, 0.25**2
    q2_m, q2_v = -0.4, 0.35**2

    # PK posteriors
    p1 = pk_update_linear_gaussian(prior, a, q_mean=q1_m, q_var=q1_v)
    p2 = pk_update_linear_gaussian(prior, a, q_mean=q2_m, q_var=q2_v)

    # KL isometry (analytic)
    kl_x = kl_gaussian_nd(p1.mean, p1.cov, p2.mean, p2.cov)
    kl_y = kl_gaussian_1d(q1_m, q1_v, q2_m, q2_v)

    # Hellinger^2 isometry (analytic)
    h2_x = hellinger2_gaussian_nd(p1.mean, p1.cov, p2.mean, p2.cov)
    h2_y = hellinger2_gaussian_nd(
        np.array([q1_m]), np.array([[q1_v]]), np.array([q2_m]), np.array([[q2_v]])
    )

    # Chi-square isometry (MC)
    n_mc = 200_000
    Xq = sample_gaussian(GaussianND(mean=p2.mean, cov=p2.cov), n=n_mc, rng=rng)
    Yq = rng.normal(loc=q2_m, scale=np.sqrt(q2_v), size=n_mc)

    chi2_x = chi2_divergence_mc_gaussian(
        sample_q=Xq,
        logp_fn=lambda x: log_gaussian_nd(x, p1.mean, p1.cov),
        logq_fn=lambda x: log_gaussian_nd(x, p2.mean, p2.cov),
    )
    chi2_y = chi2_divergence_mc_gaussian(
        sample_q=Yq,
        logp_fn=lambda y: log_gaussian_1d(y, q1_m, q1_v),
        logq_fn=lambda y: log_gaussian_1d(y, q2_m, q2_v),
    )

    print("=== Gaussian f-divergence isometry demo ===")
    print(
        f"KL:          X-space={kl_x:.8f}  Y-space={kl_y:.8f}  absdiff={abs(kl_x-kl_y):.3e}"
    )
    print(
        f"Hellinger^2: X-space={h2_x:.8f}  Y-space={h2_y:.8f}  absdiff={abs(h2_x-h2_y):.3e}"
    )
    print(
        f"Chi^2 (MC):  X-space={chi2_x:.8f}  Y-space={chi2_y:.8f}  absdiff={abs(chi2_x-chi2_y):.3e}"
    )

    # Plot
    labels = ["KL", "Hellinger^2", "Chi^2 (MC)"]
    x_vals = [kl_x, h2_x, chi2_x]
    y_vals = [kl_y, h2_y, chi2_y]

    fig = plt.figure(figsize=(8.5, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    idx = np.arange(len(labels))
    width = 0.35
    ax.bar(idx - width / 2, y_vals, width, label="Evidence space (Y)")
    ax.bar(idx + width / 2, x_vals, width, label="Posterior space (X)")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_ylabel("divergence value")
    ax.set_title("PK update induces f-divergence isometries (Gaussian linear case)")
    ax.legend()
    fig.tight_layout()

    fig_path = OUT_DIR / "divergence_isometry.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")

    summary = (
        "=== Gaussian f-divergence isometry ===\n"
        f"a={a.tolist()}\n"
        f"q1=N({q1_m},{q1_v})  q2=N({q2_m},{q2_v})\n"
        f"KL(X)={kl_x:.10f} KL(Y)={kl_y:.10f}\n"
        f"H2(X)={h2_x:.10f} H2(Y)={h2_y:.10f}\n"
        f"chi2(X)={chi2_x:.10f} chi2(Y)={chi2_y:.10f}\n"
        f"Saved figure: {fig_path}\n"
    )
    (OUT_DIR / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
