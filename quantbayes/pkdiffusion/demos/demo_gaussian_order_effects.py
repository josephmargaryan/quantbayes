# quantbayes/pkdiffusion/demos/demo_gaussian_order_effects.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from quantbayes.pkdiffusion.gaussian_pk import (
    GaussianND,
    pk_update_linear_gaussian,
    alternating_pk_updates,
    sample_gaussian,
    marginal_1d,
)
from quantbayes.pkdiffusion.metrics import kl_gaussian_nd, w2_gaussian_nd


OUT_DIR = Path("reports/pkdiffusion/gaussian_order_effects")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    rng = np.random.default_rng(0)

    # Prior: 2D standard normal
    prior = GaussianND(mean=np.zeros(2), cov=np.eye(2))

    # Two coarse maps, deliberately non-orthogonal to create order effects.
    a1 = np.array([1.0, 0.35])
    a1 = a1 / np.linalg.norm(a1)
    a2 = np.array([0.4, 1.0])
    a2 = a2 / np.linalg.norm(a2)

    # Two Gaussian evidences
    q1_mean, q1_var = 1.0, 0.25**2
    q2_mean, q2_var = -0.6, 0.20**2

    # Sequential: q1 then q2
    p1 = pk_update_linear_gaussian(prior, a1, q_mean=q1_mean, q_var=q1_var)
    p12 = pk_update_linear_gaussian(p1, a2, q_mean=q2_mean, q_var=q2_var)

    # Sequential: q2 then q1
    p2 = pk_update_linear_gaussian(prior, a2, q_mean=q2_mean, q_var=q2_var)
    p21 = pk_update_linear_gaussian(p2, a1, q_mean=q1_mean, q_var=q1_var)

    # Alternating projections (IPF-style)
    hist = alternating_pk_updates(
        prior,
        [a1, a2],
        [q1_mean, q2_mean],
        [q1_var, q2_var],
        num_iters=20,
    )
    p_ipf = hist[-1]

    # Diagnostics: marginal means/vars
    def marg_stats(g, a):
        m, v = marginal_1d(g, a)
        return m, v

    m12_1, v12_1 = marg_stats(p12, a1)
    m12_2, v12_2 = marg_stats(p12, a2)

    m21_1, v21_1 = marg_stats(p21, a1)
    m21_2, v21_2 = marg_stats(p21, a2)

    mi_1, vi_1 = marg_stats(p_ipf, a1)
    mi_2, vi_2 = marg_stats(p_ipf, a2)

    # Divergences between the two one-pass sequential solutions
    kl_12_21 = kl_gaussian_nd(p12.mean, p12.cov, p21.mean, p21.cov)
    kl_21_12 = kl_gaussian_nd(p21.mean, p21.cov, p12.mean, p12.cov)
    w2_12_21 = w2_gaussian_nd(p12.mean, p12.cov, p21.mean, p21.cov)

    # Convergence trace: constraint errors over hist
    errs = []
    for g in hist:
        m1, v1 = marg_stats(g, a1)
        m2, v2 = marg_stats(g, a2)
        err = max(
            abs(m1 - q1_mean),
            abs(v1 - q1_var),
            abs(m2 - q2_mean),
            abs(v2 - q2_var),
        )
        errs.append(err)

    # Samples for plots
    n = 20_000
    X12 = sample_gaussian(p12, n=n, rng=rng)
    X21 = sample_gaussian(p21, n=n, rng=rng)
    XI = sample_gaussian(p_ipf, n=n, rng=rng)

    # --- Plot ---
    fig = plt.figure(figsize=(13, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(errs, linewidth=2)
    ax1.set_yscale("log")
    ax1.set_title("Alternating PK updates: max constraint error")
    ax1.set_xlabel("update step (each step enforces one marginal)")
    ax1.set_ylabel("max(|mean-mean*|, |var-var*|)")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(X12[::10, 0], X12[::10, 1], s=5, alpha=0.15, label="q1→q2")
    ax2.scatter(X21[::10, 0], X21[::10, 1], s=5, alpha=0.15, label="q2→q1")
    ax2.scatter(XI[::10, 0], XI[::10, 1], s=5, alpha=0.15, label="IPF limit")
    ax2.set_aspect("equal", "box")
    ax2.set_title("Order effects (one pass) vs simultaneous (IPF limit)")
    ax2.legend()

    fig.tight_layout()
    fig_path = OUT_DIR / "order_effects.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")

    summary = (
        "=== Gaussian PK order effects demo ===\n"
        f"a1={a1.tolist()}\n"
        f"a2={a2.tolist()}\n"
        f"q1: mean={q1_mean:.6f}, var={q1_var:.6f}\n"
        f"q2: mean={q2_mean:.6f}, var={q2_var:.6f}\n\n"
        "One-pass sequential results:\n"
        f"  q1→q2:  (Y1 mean,var)=({m12_1:.6f},{v12_1:.6f})  (Y2 mean,var)=({m12_2:.6f},{v12_2:.6f})\n"
        f"  q2→q1:  (Y1 mean,var)=({m21_1:.6f},{v21_1:.6f})  (Y2 mean,var)=({m21_2:.6f},{v21_2:.6f})\n\n"
        "IPF limit:\n"
        f"  IPF:    (Y1 mean,var)=({mi_1:.6f},{vi_1:.6f})  (Y2 mean,var)=({mi_2:.6f},{vi_2:.6f})\n\n"
        "Divergences between one-pass solutions:\n"
        f"  KL(p12||p21) = {kl_12_21:.8f}\n"
        f"  KL(p21||p12) = {kl_21_12:.8f}\n"
        f"  W2(p12,p21)  = {w2_12_21:.8f}\n\n"
        f"Saved figure: {fig_path}\n"
    )
    (OUT_DIR / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
