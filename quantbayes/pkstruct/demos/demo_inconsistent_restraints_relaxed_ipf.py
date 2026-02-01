# quantbayes/pkstruct/demos/demo_inconsistent_restraints_relaxed_ipf.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

OUT_DIR = Path("reports/pkstruct/inconsistent_restraints_relaxed_ipf")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def trunc_normal_pdf(
    x: np.ndarray, *, mean: float, sd: float, low: float = 0.0, high: float = np.inf
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    z = (x - mean) / sd
    pdf = norm.pdf(z) / sd
    Z = norm.cdf((high - mean) / sd) - norm.cdf((low - mean) / sd)
    return pdf / max(float(Z), 1e-12)


def distance(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x - a[None, :], axis=-1)


def logsumexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    m = float(np.max(x))
    return float(m + np.log(np.sum(np.exp(x - m)) + 1e-300))


def softmax_logw(logw: np.ndarray) -> np.ndarray:
    lse = logsumexp(logw)
    w = np.exp(logw - lse)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0.0:
        return np.ones_like(logw) / logw.size
    return w / s


def hist_density(
    values: np.ndarray, weights: np.ndarray, bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = weights / max(float(np.sum(weights)), 1e-300)

    mass, _ = np.histogram(values, bins=bins, weights=weights)
    binw = np.diff(bins)
    pdf = mass / (binw + 1e-12)
    return pdf, mass


def kl_discrete(p_mass: np.ndarray, q_mass: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p_mass, dtype=float)
    q = np.asarray(q_mass, dtype=float)
    p = p / max(float(np.sum(p)), eps)
    q = q / max(float(np.sum(q)), eps)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def main():
    rng = np.random.default_rng(0)

    n = 80_000
    X0 = rng.normal(size=(n, 2))
    logw0 = np.zeros(n, dtype=float)

    a1 = np.array([-1.0, 0.0])
    a2 = np.array([+1.0, 0.0])
    d_anchors = float(np.linalg.norm(a1 - a2))
    print(f"Anchor distance d={d_anchors:.3f}")

    def y1_fn(x):
        return distance(x, a1)

    def y2_fn(x):
        return distance(x, a2)

    # intentionally inconsistent tight evidences
    q1_mean, q1_sd = 0.15, 0.03
    q2_mean, q2_sd = 0.15, 0.03

    y_max = 5.0
    bins = np.linspace(0.0, y_max, 220)
    mids = 0.5 * (bins[:-1] + bins[1:])
    binw = np.diff(bins)

    q1_pdf = trunc_normal_pdf(mids, mean=q1_mean, sd=q1_sd, low=0.0)
    q2_pdf = trunc_normal_pdf(mids, mean=q2_mean, sd=q2_sd, low=0.0)
    q1_mass = q1_pdf * binw
    q1_mass = q1_mass / np.sum(q1_mass)
    q2_mass = q2_pdf * binw
    q2_mass = q2_mass / np.sum(q2_mass)

    def one_run(
        tau: float, num_iters: int = 80, resample_every: int = 5, eps: float = 1e-12
    ) -> dict:
        X = X0.copy()
        logw = logw0.copy()

        kl1_hist, kl2_hist = [], []

        for it in range(num_iters):
            if not np.isfinite(logw).all():
                logw[:] = 0.0

            w = softmax_logw(logw)

            # diagnostics before updates
            y1 = y1_fn(X)
            y2 = y2_fn(X)

            p1_pdf, p1_mass = hist_density(y1, w, bins)
            p2_pdf, p2_mass = hist_density(y2, w, bins)

            kl1_hist.append(kl_discrete(p1_mass, q1_mass))
            kl2_hist.append(kl_discrete(p2_mass, q2_mass))

            # update toward constraint 1
            idx1 = np.clip(np.digitize(y1, bins) - 1, 0, p1_pdf.size - 1)
            logp1 = np.log(p1_pdf[idx1] + eps)
            logq1 = np.log(trunc_normal_pdf(y1, mean=q1_mean, sd=q1_sd, low=0.0) + eps)
            logw = logw + tau * (logq1 - logp1)
            logw = logw - logsumexp(logw)

            # update toward constraint 2
            w = softmax_logw(logw)
            y2 = y2_fn(X)
            p2_pdf, _ = hist_density(y2, w, bins)
            idx2 = np.clip(np.digitize(y2, bins) - 1, 0, p2_pdf.size - 1)
            logp2 = np.log(p2_pdf[idx2] + eps)
            logq2 = np.log(trunc_normal_pdf(y2, mean=q2_mean, sd=q2_sd, low=0.0) + eps)
            logw = logw + tau * (logq2 - logp2)
            logw = logw - logsumexp(logw)

            # resample for particle health
            if (resample_every > 0) and ((it + 1) % resample_every == 0):
                w = softmax_logw(logw)
                ess = 1.0 / np.sum(w * w)
                if ess < 0.25 * n:
                    idx = rng.choice(n, size=n, replace=True, p=w)
                    X = X[idx]
                    logw[:] = 0.0

        return dict(
            X=X, w=softmax_logw(logw), kl1=np.array(kl1_hist), kl2=np.array(kl2_hist)
        )

    hard = one_run(tau=1.0)
    damp = one_run(tau=0.25)

    fig = plt.figure(figsize=(12.5, 5.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(hard["kl1"], linewidth=2, label="KL(y1||q1) hard (tau=1)")
    ax1.plot(hard["kl2"], linewidth=2, label="KL(y2||q2) hard (tau=1)")
    ax1.plot(
        damp["kl1"], linewidth=2, linestyle="--", label="KL(y1||q1) damped (tau=0.25)"
    )
    ax1.plot(
        damp["kl2"], linewidth=2, linestyle="--", label="KL(y2||q2) damped (tau=0.25)"
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("discrete KL on histogram")
    ax1.set_title("Inconsistent restraints: hard oscillation vs damped convergence")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(X0[::80, 0], X0[::80, 1], s=6, alpha=0.08, label="prior")
    ax2.scatter(damp["X"][::80, 0], damp["X"][::80, 1], s=6, alpha=0.12, label="damped")
    ax2.scatter(hard["X"][::80, 0], hard["X"][::80, 1], s=6, alpha=0.12, label="hard")
    ax2.scatter([a1[0], a2[0]], [a1[1], a2[1]], s=80, marker="x", label="anchors")
    ax2.set_aspect("equal", "box")
    ax2.set_title("Final particle clouds")
    ax2.legend()

    fig.tight_layout()
    fig_path = OUT_DIR / "relaxed_ipf.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")

    summary = (
        "=== Inconsistent restraints: relaxed IPF demo (stable log-weights) ===\n"
        f"anchors distance d={d_anchors:.3f}\n"
        f"q1 ~ TN(mean={q1_mean}, sd={q1_sd}), q2 ~ TN(mean={q2_mean}, sd={q2_sd})\n"
        "Exact satisfaction is impossible; damped projections converge to a compromise.\n"
        f"Saved figure: {fig_path}\n"
    )
    (OUT_DIR / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
