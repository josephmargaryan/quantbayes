# quantbayes/pkdiffusion/demos/demo_gaussian_reference_free_ratio.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

from quantbayes.pkdiffusion.gaussian_pk import (
    GaussianND,
    pk_update_linear_gaussian,
    sample_gaussian,
    marginal_1d,
)
from quantbayes.pkdiffusion.metrics import (
    kl_gaussian_nd,
    w2_gaussian_nd,
    w2_empirical_1d,
)
from quantbayes.pkstruct.ratio.rff_ratio import fit_rff_logistic_ratio_1d

OUT_DIR = Path("reports/pkdiffusion/gaussian_reference_free_ratio")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def logN(y: np.ndarray, m: float, v: float) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    return -0.5 * (np.log(2 * np.pi * v) + (y - m) ** 2 / v)


def main():
    rng = np.random.default_rng(0)

    prior = GaussianND(mean=np.zeros(2), cov=np.eye(2))
    a = np.array([1.0, 0.7], dtype=float)
    a = a / np.linalg.norm(a)

    py_m, py_v = marginal_1d(prior, a)

    q_m = 1.25
    q_v = 0.20**2

    post_true = pk_update_linear_gaussian(prior, a, q_mean=q_m, q_var=q_v)

    # training samples
    n_train = 60_000
    X_prior = sample_gaussian(prior, n=n_train, rng=rng)
    y_neg = X_prior @ a  # p_Y samples
    y_pos = rng.normal(loc=q_m, scale=np.sqrt(q_v), size=n_train)  # q samples

    # fit ratio
    model = fit_rff_logistic_ratio_1d(
        y_pos=y_pos,
        y_neg=y_neg,
        m=256,
        lengthscale=None,
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

    # True log ratio
    def true_logratio(y):
        return logN(y, q_m, q_v) - logN(y, py_m, py_v)

    # --- Evaluate log-ratio error in meaningful regions ---
    # Central-quantile grid (avoids tail domination)
    y_all = np.concatenate([y_pos, y_neg])
    lo, hi = np.quantile(y_all, [0.001, 0.999])
    grid = np.linspace(lo, hi, 500)

    lr_true_grid = true_logratio(grid)
    lr_hat_grid = model.log_ratio(grid).reshape(-1)
    rmse_central = float(np.sqrt(np.mean((lr_hat_grid - lr_true_grid) ** 2)))

    # Sample-weighted RMSE under p_Y and q
    y_neg_s = y_neg[:50_000]
    y_pos_s = y_pos[:50_000]
    rmse_py = float(
        np.sqrt(
            np.mean(
                (model.log_ratio(y_neg_s).reshape(-1) - true_logratio(y_neg_s)) ** 2
            )
        )
    )
    rmse_q = float(
        np.sqrt(
            np.mean(
                (model.log_ratio(y_pos_s).reshape(-1) - true_logratio(y_pos_s)) ** 2
            )
        )
    )

    print(f"log-ratio RMSE (central quantiles) = {rmse_central:.6f}")
    print(f"log-ratio RMSE under p_Y samples    = {rmse_py:.6f}")
    print(f"log-ratio RMSE under q samples      = {rmse_q:.6f}")

    # --- posterior via ratio-resampling ---
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

    # Compare to true posterior in X
    m_emp = np.mean(X_is, axis=0)
    C_emp = np.cov(X_is.T, bias=True)
    kl_x = float(kl_gaussian_nd(m_emp, C_emp, post_true.mean, post_true.cov))
    w2_x = float(w2_gaussian_nd(m_emp, C_emp, post_true.mean, post_true.cov))

    # Coarse-space validation: y_IS should match q
    y_is = X_is @ a
    w2_y = float(w2_empirical_1d(y_is, y_pos))  # against evidence samples
    D_y, p_y = kstest(
        y_is, "norm", args=(q_m, np.sqrt(q_v))
    )  # one-sample vs target Normal

    print(f"KL(emp||true)={kl_x:.6e}  W2(emp,true)={w2_x:.6e}")
    print(f"W2(y_IS, y~q)={w2_y:.6e}  KS D(y_IS vs N(q))={D_y:.6f} (p={p_y:.3g})")

    # --- figure ---
    X_true = sample_gaussian(post_true, n=n_post, rng=rng)

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(grid, lr_true_grid, linewidth=2, label="true log(q/pY) (central range)")
    ax1.plot(
        grid, lr_hat_grid, linewidth=2, linestyle="--", label="learned log-ratio (RFF)"
    )
    ax1.set_title("Reference-free density-ratio estimation (central quantiles)")
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
        "=== Gaussian reference-free ratio demo (fixed metrics) ===\n"
        f"a={a.tolist()}\n"
        f"Evidence q: mean={q_m:.6f}, var={q_v:.6f}\n"
        f"Prior predictive pY: mean={py_m:.6f}, var={py_v:.6f}\n"
        f"log-ratio RMSE (central quantiles) = {rmse_central:.6f}\n"
        f"log-ratio RMSE under pY = {rmse_py:.6f}\n"
        f"log-ratio RMSE under q  = {rmse_q:.6f}\n"
        f"IS ESS = {ess:.1f} ({ess/n_prop:.6f} frac)\n"
        f"KL_X(emp||true) = {kl_x:.6e}\n"
        f"W2_X(emp,true)  = {w2_x:.6e}\n"
        f"W2_Y(y_IS, y~q) = {w2_y:.6e}\n"
        f"KS D(y_IS vs N(q)) = {D_y:.6f} (p={p_y:.3g})\n"
        f"Saved model: {model_path}\n"
        f"Saved figure: {fig_path}\n"
    )
    (OUT_DIR / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
