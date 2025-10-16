# tests/selftest_dp_logreg_rdp.py
# ------------------------------------------------------------
# Compare zCDP vs RDP accounting for DP-GD on logistic regression.
# - Shows RDP gives tighter eps for same sigma, or smaller sigma for same eps.
# - Runs the RDP-calibrated GD and prints achieved eps.
# - Saves one figure: eps_vs_sigma_comparison.png
# ------------------------------------------------------------

import os, math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantbayes.dp.utils.clip import clip_rows_l2
from quantbayes.dp.utils.validation import assert_2d
from quantbayes.dp.accounting import zcdp_sigma_for_gd
from quantbayes.dp.accounting_rdp import eps_from_sigma_rdp, sigma_for_target_eps_rdp
from quantbayes.dp.optim.dp_gd_rdp import dp_gradient_descent_rdp
from quantbayes.dp.models.logistic_regression import logistic_loss

OUTDIR = "tests_out"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed=0):
    np.random.seed(seed)


def make_logreg_data(n=1500, d=50, seed=11, margin_noise=0.3):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    w_true /= np.linalg.norm(w_true) + 1e-12
    logits = X @ w_true + margin_noise * rng.normal(size=n)
    y = np.where(logits >= 0.0, 1, -1).astype(int)
    assert_2d(X, "X")
    return X, y


def compare_eps_vs_sigma():
    ensure_dir(OUTDIR)
    n, d = 1000, 40
    L = 1.0
    steps = 800
    delta = 1e-5
    Delta = (2.0 * L) / n

    # Sweep some sigma values and compare eps_zcdp vs eps_rdp
    sigmas = np.linspace(0.08, 0.40, 8)
    eps_zcdp, eps_rdp = [], []
    for sigma in sigmas:
        # zCDP -> eps
        rho = steps * (Delta**2) / (2.0 * sigma**2)
        ez = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
        eps_zcdp.append(ez)

        # RDP -> eps (optimize alpha over a grid)
        er, _ = eps_from_sigma_rdp(sigma=sigma, delta_l2=Delta, T=steps, delta=delta)
        eps_rdp.append(er)

    # Plot comparison
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(sigmas, eps_zcdp, marker="o", linewidth=2, label="zCDP→ε")
    plt.plot(sigmas, eps_rdp, marker="o", linewidth=2, label="RDP→ε (min over α)")
    plt.xlabel("Per-step σ")
    plt.ylabel("ε for δ=1e-5")
    plt.title("ε vs σ: RDP vs zCDP (DP-GD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "eps_vs_sigma_comparison.png"), dpi=180)
    plt.close()

    print("\nσ  |   ε_zCDP    ε_RDP")
    for s, ez, er in zip(sigmas, eps_zcdp, eps_rdp):
        print(f"{s:0.3f} | {ez:9.4f} {er:9.4f}")


def run_rdp_calibrated_gd():
    print("\n" + "=" * 80)
    print("RDP-calibrated DP-GD on logistic regression")
    print("=" * 80)
    X_raw, y = make_logreg_data(n=1500, d=50, seed=11, margin_noise=0.3)
    X = clip_rows_l2(X_raw, 1.0)

    lam = 0.5
    L = 1.0
    eps_target = 1.0
    delta = 1e-5
    steps = 1200

    w_rdp, sigma_used, eps_ach = dp_gradient_descent_rdp(
        X,
        y,
        lam=lam,
        eps=eps_target,
        delta=delta,
        steps=steps,
        lr=0.1,
        L=L,
        proj_radius=None,
        average=True,
        seed=0,
    )
    loss = logistic_loss(w_rdp, X, y)
    print(
        f"RDP DP-GD: sigma_used={sigma_used:.6f}, eps_achieved≈{eps_ach:.4f}, loss={loss:.4f}"
    )

    # Compare with sigma from zCDP at same (eps, delta)
    n = X.shape[0]
    Delta = (2.0 * L) / n
    sigma_z = zcdp_sigma_for_gd(eps=eps_target, delta=delta, T=steps, delta_l2=Delta)
    print(f"zCDP sigma for same (eps,delta): {sigma_z:.6f} (RDP should be ≤ zCDP)")


if __name__ == "__main__":
    set_seed(0)
    compare_eps_vs_sigma()
    run_rdp_calibrated_gd()
    print("\nRDP vs zCDP suite done. Figures saved in:", OUTDIR)
