# quantbayes/pkstruct/demos/demo_vrw_seed_sweep.py
from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

import jax

jax.config.update("jax_enable_x64", True)
import jax.random as jr

from quantbayes.bnn.pkstruct.vrw_numpyro import run_nuts_vrw_pk, VRWNUTSConfig


# =========================
# USER SETTINGS (edit here)
# =========================
OUT_DIR = Path("reports/vrw_seed_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SEEDS = 10
BASE_SEED = 1000

# Paper settings
N = 5
MU = 0.0
KAPPA = 10.0
ALPHA = 10.0
BETA = 10.0

# Keep a bit lighter than Fig.3 demo for speed
NUM_WARMUP = 600
NUM_SAMPLES = 600
THIN = 5


def main():
    print("=== VRW PK seed sweep demo ===")
    rows = []

    base = dict(
        N=N,
        mu=MU,
        kappa=KAPPA,
        alpha=ALPHA,
        beta=BETA,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
    )

    for i in range(NUM_SEEDS):
        seed_pk = BASE_SEED + 2 * i
        seed_ab = BASE_SEED + 2 * i + 1

        r_pk = run_nuts_vrw_pk(
            jr.PRNGKey(seed_pk), VRWNUTSConfig(**base, use_reference=True)
        )["r"]
        r_ab = run_nuts_vrw_pk(
            jr.PRNGKey(seed_ab), VRWNUTSConfig(**base, use_reference=False)
        )["r"]

        r_pk_s = r_pk[::THIN] / N
        r_ab_s = r_ab[::THIN] / N

        D_pk, p_pk = kstest(r_pk_s, "beta", args=(ALPHA, BETA))
        D_ab, p_ab = kstest(r_ab_s, "beta", args=(ALPHA, BETA))

        rows.append(
            dict(
                seed=i,
                seed_pk=seed_pk,
                seed_ab=seed_ab,
                D_pk=D_pk,
                p_pk=p_pk,
                D_ab=D_ab,
                p_ab=p_ab,
            )
        )

        print(f"[{i:02d}] D_pk={D_pk:.3f}  D_ab={D_ab:.3f}")

    D_pk_vals = np.array([r["D_pk"] for r in rows], dtype=float)
    D_ab_vals = np.array([r["D_ab"] for r in rows], dtype=float)

    print("\nSummary:")
    print(f"PK:   D mean±std = {D_pk_vals.mean():.3f} ± {D_pk_vals.std(ddof=1):.3f}")
    print(f"ABLA: D mean±std = {D_ab_vals.mean():.3f} ± {D_ab_vals.std(ddof=1):.3f}")

    # Save CSV for slides
    csv_path = OUT_DIR / "vrw_seed_sweep.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {csv_path}")

    summary_path = OUT_DIR / "summary.txt"
    summary_path.write_text(
        "=== VRW PK seed sweep ===\n"
        f"NUM_SEEDS={NUM_SEEDS} N={N} mu={MU} kappa={KAPPA} alpha={ALPHA} beta={BETA}\n"
        f"PK:   D mean±std = {D_pk_vals.mean():.6f} ± {D_pk_vals.std(ddof=1):.6f}\n"
        f"ABLA: D mean±std = {D_ab_vals.mean():.6f} ± {D_ab_vals.std(ddof=1):.6f}\n"
    )
    print(f"Saved: {summary_path}")

    # Quick visualization for slides
    plt.figure(figsize=(8, 4))
    plt.hist(D_pk_vals, bins=10, alpha=0.6, label="PK KS D")
    plt.hist(D_ab_vals, bins=10, alpha=0.6, label="Ablation KS D")
    plt.xlabel("KS D (lower is better)")
    plt.ylabel("count")
    plt.title("VRW PK vs no-ref across seeds")
    plt.show()
    plt.legend()

    fig_path = OUT_DIR / "vrw_seed_sweep_hist.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {fig_path}")

    # optional: plt.show()


if __name__ == "__main__":
    main()
