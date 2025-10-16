# tests/selftest_dp_sota_dpsgd.py
# ------------------------------------------------------------
# Bleeding-edge: DP-SGD with Poisson subsampling + RDP accounting.
# - Trains DP-SGD at target eps in {0.5, 1.0, 2.0}.
# - Visualizes loss vs eps, sigma vs eps.
# - Also plots eps vs sigma (accountant-only) around calibrated sigma range.
# ------------------------------------------------------------

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantbayes.dp.utils.clip import clip_rows_l2
from quantbayes.dp.utils.validation import assert_2d
from quantbayes.dp.optim.dp_sgd_rdp import dp_sgd_rdp_logreg
from quantbayes.dp.accounting_rdp_subsampled import eps_from_sigma_subsampled_rdp

OUTDIR = "tests_out"


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def set_seed(seed=0):
    np.random.seed(seed)


def make_logreg_data(n=4000, d=80, seed=123, margin_noise=0.35):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    w_true /= np.linalg.norm(w_true) + 1e-12
    logits = X @ w_true + margin_noise * rng.normal(size=n)
    y = np.where(logits >= 0.0, 1, -1).astype(int)
    assert_2d(X, "X")
    return X, y


def logistic_loss(w, X, y):
    z = -y * (X @ w)
    return float(np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z))).mean())


def plot_curve(x, curves: dict, xlabel: str, ylabel: str, title: str, out_png: str):
    plt.figure(figsize=(6.4, 4.2))
    for label, y in curves.items():
        plt.plot(x, y, marker="o", linewidth=2, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def run_dpsgd_suite():
    ensure_dir(OUTDIR)
    print("\n" + "=" * 80)
    print("Bleeding-edge DP-SGD (Poisson + RDP) smoke test")
    print("=" * 80)

    X_raw, y = make_logreg_data(n=4000, d=80, seed=123, margin_noise=0.35)
    X = clip_rows_l2(X_raw, 1.0)

    lam = 0.0
    steps = 2000
    lr = 0.05
    q = 0.1
    C = 1.0
    delta = 1e-5
    eps_targets = [0.5, 1.0, 2.0]
    seeds = [0, 1, 2]

    mean_losses, mean_sigmas = [], []

    for eps in eps_targets:
        losses, sigmas = [], []
        for s in seeds:
            w, sigma, eps_ach = dp_sgd_rdp_logreg(
                X,
                y,
                lam=lam,
                eps=eps,
                delta=delta,
                steps=steps,
                lr=lr,
                clip_norm=C,
                sample_rate=q,
                seed=s,
            )
            loss = logistic_loss(w, X, y)
            losses.append(loss)
            sigmas.append(sigma)
            print(
                f"eps_target={eps:.2f}, seed={s}: sigma={sigma:.4f}, eps_ach≈{eps_ach:.3f}, loss={loss:.4f}"
            )
        mean_losses.append(float(np.mean(losses)))
        mean_sigmas.append(float(np.mean(sigmas)))

    plot_curve(
        eps_targets,
        {"DP-SGD (RDP)": mean_losses},
        xlabel="epsilon (ε)",
        ylabel="Training log-loss",
        title="DP-SGD: loss vs ε",
        out_png=os.path.join(OUTDIR, "dpsgd_loss_vs_eps.png"),
    )
    plot_curve(
        eps_targets,
        {"DP-SGD (RDP)": mean_sigmas},
        xlabel="epsilon (ε)",
        ylabel="Calibrated σ",
        title="DP-SGD: σ vs ε",
        out_png=os.path.join(OUTDIR, "dpsgd_sigma_vs_eps.png"),
    )

    # Accountant-only epsilon vs sigma: sweep around the calibrated σ for ε=1.0
    try:
        idx = eps_targets.index(1.0)
        sigma_pivot = mean_sigmas[idx]
    except ValueError:
        # fallback: median of calibrated sigmas
        sigma_pivot = float(np.median(mean_sigmas))
    sigmas = np.linspace(0.5 * sigma_pivot, 1.5 * sigma_pivot, 10)
    eps_vals = []
    for s in sigmas:
        e, _ = eps_from_sigma_subsampled_rdp(sigma=s, q=q, T=steps, delta=delta)
        eps_vals.append(e)
    plot_curve(
        sigmas,
        {"RDP accountant": eps_vals},
        xlabel="σ (noise stddev per coordinate)",
        ylabel="ε (δ=1e-5)",
        title=f"RDP (Poisson q={q}, steps={steps}): ε vs σ",
        out_png=os.path.join(OUTDIR, "dpsgd_eps_vs_sigma.png"),
    )

    print("\nSummary (means across seeds):")
    print(" eps |   loss    sigma")
    for e, l, s in zip(eps_targets, mean_losses, mean_sigmas):
        print(f" {e:>3.1f} | {l:7.4f}  {s:6.4f}")
    print("\nFigures saved in:", OUTDIR)


if __name__ == "__main__":
    set_seed(0)
    run_dpsgd_suite()
