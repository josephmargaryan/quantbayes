# quantbayes/pkdiffusion/demos/demo_gaussian_reference_free_ratio.py
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
from quantbayes.pkdiffusion.metrics import kl_gaussian_nd, w2_gaussian_nd
from quantbayes.pkstruct.ratio.rff_ratio import fit_rff_logistic_ratio_1d

OUT_DIR = Path("reports/pkdiffusion/gaussian_reference_free_ratio")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    rng = np.random.default_rng(0)

    # Prior and coarse map
    prior = GaussianND(mean=np.zeros(2), cov=np.eye(2))
    a = np.array([1.0, 0.7], dtype=float)
    a = a / np.linalg.norm(a)

    py_m, py_v = marginal_1d(prior, a)

    # Evidence q(y) = N(mq, vq)
    q_m = 1.25
    q_v = 0.20**2

    # True PK posterior (analytic, for evaluation)
    post_true = pk_update_linear_gaussian(prior, a, q_mean=q_m, q_var=q_v)

    # --- Samples for ratio training ---
    n_train = 60_000
    X_prior = sample_gaussian(prior, n=n_train, rng=rng)
    y_neg = X_prior @ a  # samples from p_Y (prior predictive)

    y_pos = rng.normal(loc=q_m, scale=np.sqrt(q_v), size=n_train)  # samples from q

    # Fit ratio model log(q/p_Y) via RFF logistic
    model = fit_rff_logistic_ratio_1d(
        y_pos=y_pos,
        y_neg=y_neg,
        m=256,
        lengthscale=None,  # median heuristic
        reg=1e-4,
        lr=2e-2,
        num_steps=5000,
        batch_size=2048,
        seed=0,
        print_every=500,
    )

    model_path = OUT_DIR / "ratio_model_rff.npz"
    model.save_npz(str(model_path))
    print(f"Saved ratio model: {model_path}")

    # --- Evaluate ratio curve ---
    grid = np.linspace(
        min(y_neg.min(), y_pos.min()) - 1.0, max(y_neg.max(), y_pos.max()) + 1.0, 500
    )

    # True log ratio (since we know p_Y analytically in this toy)
    def logN(y, m, v):
        return -0.5 * (np.log(2 * np.pi * v) + (y - m) ** 2 / v)

    log_ratio_true = logN(grid, q_m, q_v) - logN(grid, py_m, py_v)
    log_ratio_hat = model.log_ratio(grid).reshape(-1)

    rmse = float(np.sqrt(np.mean((log_ratio_hat - log_ratio_true) ** 2)))
    print(f"log-ratio RMSE on grid: {rmse:.6f}")

    # --- Posterior via importance resampling from prior using learned ratio ---
    n_prop = 250_000
    X_prop = sample_gaussian(prior, n=n_prop, rng=rng)
    y_prop = X_prop @ a

    logw = model.log_ratio(y_prop).reshape(-1)
    logw = logw - np.max(logw)
    logw = np.clip(logw, -80.0, 0.0)
    w = np.exp(logw)
    w = w / np.sum(w)

    ess = float(1.0 / np.sum(w * w))
    print(f"IS ESS: {ess:.1f} ({ess/n_prop:.4f} of proposal)")

    n_post = 40_000
    idx = rng.choice(n_prop, size=n_post, replace=True, p=w)
    X_is = X_prop[idx]

    # Empirical mean/cov
    m_emp = np.mean(X_is, axis=0)
    C_emp = np.cov(X_is.T, bias=True)

    # Compare to true posterior Gaussian
    kl = float(kl_gaussian_nd(m_emp, C_emp, post_true.mean, post_true.cov))
    w2 = float(w2_gaussian_nd(m_emp, C_emp, post_true.mean, post_true.cov))
    print(f"KL(emp||true)={kl:.6e}  W2(emp,true)={w2:.6e}")

    # --- Figure: ratio curve + scatter ---
    X_true = sample_gaussian(post_true, n=n_post, rng=rng)

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(grid, log_ratio_true, linewidth=2, label="true log(q/pY)")
    ax1.plot(
        grid,
        log_ratio_hat,
        linewidth=2,
        linestyle="--",
        label="learned log-ratio (RFF)",
    )
    ax1.set_title("Reference-free density-ratio estimation in evidence space")
    ax1.set_xlabel("y")
    ax1.set_ylabel("log ratio")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(
        X_prop[::60, 0], X_prop[::60, 1], s=5, alpha=0.10, label="prior samples"
    )
    ax2.scatter(
        X_is[::60, 0],
        X_is[::60, 1],
        s=5,
        alpha=0.10,
        label="IS posterior (ratio-learned)",
    )
    ax2.scatter(
        X_true[::60, 0], X_true[::60, 1], s=5, alpha=0.10, label="true PK posterior"
    )
    ax2.set_aspect("equal", "box")
    ax2.set_title("Posterior via ratio-resampling matches true PK posterior")
    ax2.legend()

    fig.tight_layout()
    fig_path = OUT_DIR / "reference_free_ratio.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")

    summary = (
        "=== Gaussian reference-free ratio demo ===\n"
        f"a={a.tolist()}\n"
        f"Evidence q: mean={q_m:.6f}, var={q_v:.6f}\n"
        f"Prior predictive pY: mean={py_m:.6f}, var={py_v:.6f}\n"
        f"log-ratio RMSE (grid) = {rmse:.6f}\n"
        f"IS ESS = {ess:.1f} ({ess/n_prop:.6f} frac)\n"
        f"KL(emp||true) = {kl:.6e}\n"
        f"W2(emp,true)  = {w2:.6e}\n"
        f"Saved model: {model_path}\n"
        f"Saved figure: {fig_path}\n"
    )
    (OUT_DIR / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
