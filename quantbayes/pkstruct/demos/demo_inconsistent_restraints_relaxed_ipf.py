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


def hist_density(
    values: np.ndarray, weights: np.ndarray, bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted histogram density estimate on bins.
    Returns (pdf_per_bin, mass_per_bin).
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = weights / np.sum(weights)

    counts, _ = np.histogram(values, bins=bins, weights=weights)
    binw = np.diff(bins)
    mass = counts  # already sums to 1 approximately
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

    # Prior over x in R^2
    n = 80_000
    X = rng.normal(size=(n, 2))  # N(0,I)
    w = np.ones(n, dtype=float) / n

    # Two anchor points separated by distance d=2.0
    a1 = np.array([-1.0, 0.0])
    a2 = np.array([+1.0, 0.0])
    d_anchors = float(np.linalg.norm(a1 - a2))
    print(f"Anchor distance d={d_anchors:.3f}")

    # Coarse observables: y1=||x-a1||, y2=||x-a2||
    def y1_fn(x):
        return distance(x, a1)

    def y2_fn(x):
        return distance(x, a2)

    # Inconsistent soft evidences: both very small (but y1+y2 >= d must hold)
    q1_mean, q1_sd = 0.15, 0.03
    q2_mean, q2_sd = 0.15, 0.03

    # Histogram grid
    y_max = 5.0
    bins = np.linspace(0.0, y_max, 220)
    mids = 0.5 * (bins[:-1] + bins[1:])
    binw = np.diff(bins)

    # Evidence masses on bins (approx by pdf(mid)*binw then normalize)
    q1_pdf = trunc_normal_pdf(mids, mean=q1_mean, sd=q1_sd, low=0.0)
    q2_pdf = trunc_normal_pdf(mids, mean=q2_mean, sd=q2_sd, low=0.0)
    q1_mass = q1_pdf * binw
    q2_mass = q2_pdf * binw
    q1_mass = q1_mass / np.sum(q1_mass)
    q2_mass = q2_mass / np.sum(q2_mass)

    def one_run(tau: float, num_iters: int = 60, resample_every: int = 5) -> dict:
        Xc = X.copy()
        wc = w.copy()

        kl1_hist, kl2_hist = [], []

        for it in range(num_iters):
            y1 = y1_fn(Xc)
            y2 = y2_fn(Xc)

            p1_pdf, p1_mass = hist_density(y1, wc, bins)
            p2_pdf, p2_mass = hist_density(y2, wc, bins)

            kl1_hist.append(kl_discrete(p1_mass, q1_mass))
            kl2_hist.append(kl_discrete(p2_mass, q2_mass))

            # Update toward constraint 1
            idx1 = np.clip(np.digitize(y1, bins) - 1, 0, p1_pdf.size - 1)
            p1_y = p1_pdf[idx1]
            q1_y = trunc_normal_pdf(y1, mean=q1_mean, sd=q1_sd, low=0.0)
            ratio1 = (q1_y / (p1_y + 1e-12)) ** tau
            wc = wc * ratio1
            wc = wc / np.sum(wc)

            # Update toward constraint 2
            y2 = y2_fn(Xc)
            p2_pdf, _ = hist_density(y2, wc, bins)
            idx2 = np.clip(np.digitize(y2, bins) - 1, 0, p2_pdf.size - 1)
            p2_y = p2_pdf[idx2]
            q2_y = trunc_normal_pdf(y2, mean=q2_mean, sd=q2_sd, low=0.0)
            ratio2 = (q2_y / (p2_y + 1e-12)) ** tau
            wc = wc * ratio2
            wc = wc / np.sum(wc)

            # Occasional resampling for particle health
            if (resample_every > 0) and ((it + 1) % resample_every == 0):
                ess = 1.0 / np.sum(wc * wc)
                # resample if ESS is low
                if ess < 0.25 * n:
                    idx = rng.choice(n, size=n, replace=True, p=wc)
                    Xc = Xc[idx]
                    wc = np.ones(n, dtype=float) / n

        return dict(X=Xc, w=wc, kl1=np.array(kl1_hist), kl2=np.array(kl2_hist))

    # Compare: hard projections (tau=1) vs damped (tau<1)
    hard = one_run(tau=1.0, num_iters=80)
    damp = one_run(tau=0.25, num_iters=80)

    # Plot KL traces
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

    # Scatter: final particle clouds
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(X[::80, 0], X[::80, 1], s=6, alpha=0.08, label="prior")
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
        "=== Inconsistent restraints: relaxed IPF demo ===\n"
        f"anchors distance d={d_anchors:.3f}\n"
        f"q1 ~ TN(mean={q1_mean}, sd={q1_sd}), q2 ~ TN(mean={q2_mean}, sd={q2_sd})\n"
        "Expectation: constraints incompatible, so exact enforcement is impossible.\n"
        f"Saved figure: {fig_path}\n"
    )
    (OUT_DIR / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
