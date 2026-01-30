from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from quantbayes.pkdiffusion.gaussian_pk import (
    GaussianND,
    pk_update_linear_gaussian,
    sample_gaussian,
    marginal_1d,
)
from quantbayes.pkdiffusion.metrics import kl_gaussian_1d, kl_gaussian_nd


OUT_DIR = Path("reports/pkdiffusion/gaussian_kl_identity")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    rng = np.random.default_rng(0)

    # Prior: 2D standard normal
    prior = GaussianND(mean=np.zeros(2), cov=np.eye(2))

    # Coarse map Y = a^T X (choose a non-axis direction)
    a = np.array([1.0, 0.7], dtype=float)
    a = a / np.linalg.norm(a)

    mu_y, var_y = marginal_1d(prior, a)

    # Evidence q(y) = N(mq, vq)
    mq = 1.25
    vq = 0.20**2

    post = pk_update_linear_gaussian(prior, a, q_mean=mq, q_var=vq)

    # --- KL identity check ---
    # The theorem says: KL(post||prior) = KL(q||p_Y)
    kl_post_prior = kl_gaussian_nd(post.mean, post.cov, prior.mean, prior.cov)
    kl_q_py = kl_gaussian_1d(mq, vq, mu_y, var_y)

    # Also the reverse direction (not the main identity, but sanity):
    kl_prior_post = kl_gaussian_nd(prior.mean, prior.cov, post.mean, post.cov)
    kl_py_q = kl_gaussian_1d(mu_y, var_y, mq, vq)

    # --- Samples for visualization ---
    n = 25_000
    Xp = sample_gaussian(prior, n=n, rng=rng)
    Xq = sample_gaussian(post, n=n, rng=rng)
    Yp = Xp @ a
    Yq = Xq @ a

    # --- Figure ---
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(Xp[::10, 0], Xp[::10, 1], s=5, alpha=0.2, label="prior")
    ax1.scatter(Xq[::10, 0], Xq[::10, 1], s=5, alpha=0.2, label="PK posterior")
    ax1.set_aspect("equal", "box")
    ax1.set_title("2D samples (prior vs PK posterior)")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(Yp, bins=80, density=True, alpha=0.5, label="prior marginal p_Y")
    ax2.hist(
        Yq,
        bins=80,
        density=True,
        alpha=0.5,
        label="posterior marginal (should match q)",
    )
    # Plot the evidence q analytically
    ygrid = np.linspace(
        min(Yp.min(), Yq.min()) - 0.5, max(Yp.max(), Yq.max()) + 0.5, 400
    )
    qpdf = (1.0 / np.sqrt(2 * np.pi * vq)) * np.exp(-0.5 * (ygrid - mq) ** 2 / vq)
    ax2.plot(ygrid, qpdf, linewidth=2, label="q(y) target")
    ax2.set_title("Coarse marginal check: Y=aᵀX")
    ax2.legend()

    fig.tight_layout()

    fig_path = OUT_DIR / "kl_identity.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")

    summary = (
        "=== Gaussian PK KL identity demo ===\n"
        f"a={a.tolist()}\n"
        f"Prior Y mean={mu_y:.6f}, var={var_y:.6f}\n"
        f"Evidence q mean={mq:.6f}, var={vq:.6f}\n\n"
        f"KL(post || prior) = {kl_post_prior:.10f}\n"
        f"KL(q || p_Y)      = {kl_q_py:.10f}\n"
        f"ABS diff          = {abs(kl_post_prior - kl_q_py):.3e}\n\n"
        f"KL(prior || post) = {kl_prior_post:.10f}\n"
        f"KL(p_Y || q)      = {kl_py_q:.10f}\n"
        f"ABS diff          = {abs(kl_prior_post - kl_py_q):.3e}\n"
        f"\nSaved figure: {fig_path}\n"
    )
    (OUT_DIR / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
