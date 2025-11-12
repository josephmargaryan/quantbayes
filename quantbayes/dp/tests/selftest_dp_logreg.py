# tests/selftest_dp_logreg.py
# ------------------------------------------------------------
# End-to-end (no pytest) for DP-ERM on logistic regression.
# - Synthetic data (±1 labels), row clipping.
# - Mechanisms: output perturbation, objective perturbation,
#   DP-GD with "zcdp" and "lecture" calibrations.
# - Visualizations saved to tests_out/.
# ------------------------------------------------------------

import os, math, time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === Import your library ===
from quantbayes.dp.utils.clip import clip_rows_l2
from quantbayes.dp.utils.validation import assert_2d
from quantbayes.dp.accounting import zcdp_sigma_for_gd
from quantbayes.dp.models.logistic_regression import train_logreg_gd, logistic_loss
from quantbayes.dp.optim.output_perturbation import output_perturbation
from quantbayes.dp.optim.objective_perturbation import train_objective_perturbed
from quantbayes.dp.optim.dp_gd import dp_gradient_descent

OUTDIR = "tests_out"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed=0):
    np.random.seed(seed)


def make_logreg_data(n=2000, d=60, seed=42, margin_noise=0.25):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    w_true /= np.linalg.norm(w_true) + 1e-12
    logits = X @ w_true + margin_noise * rng.normal(size=n)
    y = np.where(logits >= 0.0, 1, -1).astype(int)
    assert_2d(X, "X")
    return X, y


def monotone_nonincreasing(values, rtol=0.15, atol=1e-6):
    values = list(values)
    for i in range(1, len(values)):
        if values[i] > (1 + rtol) * values[i - 1] + atol:
            return False
    return True


def plot_loss_vs_eps(eps_list, losses_dict, out_path):
    plt.figure(figsize=(6.4, 4.2))
    for label, ys in losses_dict.items():
        plt.plot(eps_list, ys, marker="o", linewidth=2, label=label)
    plt.xlabel("epsilon (ε)")
    plt.ylabel("Training log-loss (lower is better)")
    plt.title("DP-ERM: loss vs ε")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_sigma_vs_eps(eps_list, sigmas_dict, out_path):
    plt.figure(figsize=(6.4, 4.2))
    for label, ys in sigmas_dict.items():
        plt.plot(eps_list, ys, marker="o", linewidth=2, label=label)
    plt.xlabel("epsilon (ε)")
    plt.ylabel("Per-step σ (DP-GD)")
    plt.title("DP-GD σ vs ε (calibration)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run_accounting_check():
    print("\n" + "=" * 80)
    print("Accounting sanity check (DP-GD zCDP calibration)")
    print("=" * 80)
    n, d = 1000, 40
    L = 1.0
    steps = 800
    eps = 1.0
    delta = 1e-5
    Delta = (2.0 * L) / n
    sigma = zcdp_sigma_for_gd(eps=eps, delta=delta, T=steps, delta_l2=Delta)
    rho = steps * (Delta**2) / (2.0 * sigma**2)
    eps_recovered = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
    print(
        f"Calibrated sigma={sigma:.6f}, recovered eps={eps_recovered:.4f} (target {eps})"
    )
    assert abs(eps_recovered - eps) <= 0.05, "zCDP conversion/recovery check failed."


def run_logreg_suite():
    print("\n" + "=" * 80)
    print("DP-ERM smoke test (Logistic Regression)")
    print("=" * 80)
    ensure_dir(OUTDIR)

    # Data & clip
    X_raw, y = make_logreg_data(n=2000, d=60, seed=42, margin_noise=0.25)
    X = clip_rows_l2(X_raw, 1.0)
    norms = np.linalg.norm(X, axis=1)
    assert np.all(norms <= 1.0 + 1e-9), "Row clipping failed."

    lam = 0.5  # must be >~ 0.25 for objective perturbation proof
    L = 1.0
    delta = 1e-5

    # Baseline ERM
    t0 = time.time()
    w_hat = train_logreg_gd(X, y, lam=lam, steps=2000, lr=0.1)
    base_loss = logistic_loss(w_hat, X, y)
    t1 = time.time()
    print(f"Baseline ERM: train log-loss={base_loss:.4f} (time {t1 - t0:.2f}s)")

    eps_list = [0.3, 0.7, 1.5]
    seeds = [0, 1, 2]

    loss_output, loss_objective, loss_dpgd_zcdp, loss_dpgd_lecture = [], [], [], []
    sigma_zcdp, sigma_lecture = [], []

    for eps in eps_list:
        # Output perturbation
        losses = []
        for s in seeds:
            w_out, _ = output_perturbation(
                w_hat, n=X.shape[0], lam=lam, eps=eps, L=L, seed=s
            )
            losses.append(logistic_loss(w_out, X, y))
        loss_output.append(float(np.mean(losses)))

        # Objective perturbation (beta = eps/(2L), requires lam>c)
        losses = []
        for s in seeds:
            w_obj, _ = train_objective_perturbed(
                X,
                y,
                lam=lam,
                eps=eps,
                L=L,
                steps=4000,
                lr=0.05,
                seed=s,
                enforce_assumptions=True,
            )
            losses.append(logistic_loss(w_obj, X, y))
        loss_objective.append(float(np.mean(losses)))

        # DP-GD (zcdp)
        losses, sigmas = [], []
        for s in seeds:
            w_gd, sigma = dp_gradient_descent(
                X,
                y,
                lam=lam,
                eps=eps,
                delta=delta,
                steps=1200,
                lr=0.1,
                L=L,
                proj_radius=None,
                average=True,
                seed=s,
                calibration="zcdp",
            )
            losses.append(logistic_loss(w_gd, X, y))
            sigmas.append(sigma)
        loss_dpgd_zcdp.append(float(np.mean(losses)))
        sigma_zcdp.append(float(np.mean(sigmas)))
        print(
            f"eps={eps:.2f} | DP-GD zcdp sigma (mean across seeds) = {np.mean(sigmas):.6f}"
        )

        # DP-GD (lecture)
        losses, sigmas = [], []
        for s in seeds:
            w_gd, sigma = dp_gradient_descent(
                X,
                y,
                lam=lam,
                eps=eps,
                delta=delta,
                steps=1200,
                lr=0.1,
                L=L,
                proj_radius=None,
                average=True,
                seed=s,
                calibration="lecture",
            )
            losses.append(logistic_loss(w_gd, X, y))
            sigmas.append(sigma)
        loss_dpgd_lecture.append(float(np.mean(losses)))
        sigma_lecture.append(float(np.mean(sigmas)))
        print(
            f"eps={eps:.2f} | DP-GD lecture sigma (mean across seeds) = {np.mean(sigmas):.6f}"
        )

    # Print summary
    print("\nMean train log-loss vs epsilon (lower is better):")
    header = ["eps", "output", "objective", "dpgd(zcdp)", "dpgd(lecture)"]
    print("{:<6} {:>10} {:>12} {:>12} {:>14}".format(*header))
    for i, eps in enumerate(eps_list):
        print(
            "{:<6} {:>10} {:>12} {:>12} {:>14}".format(
                f"{eps:.2f}",
                f"{loss_output[i]:.4f}",
                f"{loss_objective[i]:.6f}",
                f"{loss_dpgd_zcdp[i]:.4f}",
                f"{loss_dpgd_lecture[i]:.4f}",
            )
        )
    for name, arr in {
        "output": loss_output,
        "objective": loss_objective,
        "dpgd_zcdp": loss_dpgd_zcdp,
        "dpgd_lecture": loss_dpgd_lecture,
    }.items():
        ok = monotone_nonincreasing(arr, rtol=0.15)
        print(f"Monotonicity check ({name}): {'OK' if ok else 'WARN'}")
    print("Baseline (for reference):", base_loss)

    # Visualizations
    losses_dict = {
        "output": loss_output,
        "objective": loss_objective,
        "dpgd(zcdp)": loss_dpgd_zcdp,
        "dpgd(lecture)": loss_dpgd_lecture,
    }
    plot_loss_vs_eps(
        eps_list, losses_dict, os.path.join(OUTDIR, "logreg_loss_vs_eps.png")
    )

    sigmas_dict = {
        "dpgd(zcdp)": sigma_zcdp,
        "dpgd(lecture)": sigma_lecture,
    }
    plot_sigma_vs_eps(
        eps_list, sigmas_dict, os.path.join(OUTDIR, "logreg_sigma_vs_eps.png")
    )


if __name__ == "__main__":
    set_seed(0)
    run_accounting_check()
    run_logreg_suite()
    print("\nLogReg suite done. Figures saved in:", OUTDIR)
