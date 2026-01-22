from __future__ import annotations

from pathlib import Path
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
OUT_DIR = Path("reports/bnn_vrw_nuts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paper settings
N = 5
MU = 0.0
KAPPA = 10.0
ALPHA = 10.0
BETA = 10.0

NUM_WARMUP = 1000
NUM_SAMPLES = 1000
THIN = 5

# sanity thresholds
MAX_D_PK = 0.25
MIN_D_AB = 0.80


def main():
    print("=== bnn (NumPyro) VRW NUTS demo: PK vs ablation ===")

    cfg_pk = VRWNUTSConfig(
        N=N,
        mu=MU,
        kappa=KAPPA,
        alpha=ALPHA,
        beta=BETA,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        use_reference=True,
    )
    cfg_ab = VRWNUTSConfig(
        N=N,
        mu=MU,
        kappa=KAPPA,
        alpha=ALPHA,
        beta=BETA,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        use_reference=False,
    )

    r_pk = run_nuts_vrw_pk(jr.PRNGKey(0), cfg_pk)["r"]
    r_ab = run_nuts_vrw_pk(jr.PRNGKey(1), cfg_ab)["r"]

    D_pk, p_pk = kstest((r_pk[::THIN] / N), "beta", args=(ALPHA, BETA))
    D_ab, p_ab = kstest((r_ab[::THIN] / N), "beta", args=(ALPHA, BETA))

    print(f"PK (with ref):   KS D={D_pk:.3f}, p={p_pk:.3g}")
    print(f"No-ref ablation: KS D={D_ab:.3f}, p={p_ab:.3g}")

    assert D_pk < MAX_D_PK, f"PK KS D too large: {D_pk:.3f} (threshold {MAX_D_PK})"
    assert (
        D_ab > MIN_D_AB
    ), f"Ablation KS D too small: {D_ab:.3f} (threshold {MIN_D_AB})"
    print("PASS: KS sanity thresholds satisfied.")

    # quick plot for slides
    plt.figure(figsize=(8, 4))
    plt.hist(r_pk, bins=60, density=True, alpha=0.5, label="PK (with ref)")
    plt.hist(r_ab, bins=60, density=True, alpha=0.5, label="Ablation (no ref)")
    plt.xlabel("r")
    plt.ylabel("density")
    plt.title("VRW NUTS: PK vs no-ref ablation")
    plt.show()
    plt.legend()

    fig_path = OUT_DIR / "bnn_vrw_nuts_pk_vs_ablation.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {fig_path}")

    summary_path = OUT_DIR / "summary.txt"
    summary_path.write_text(
        "=== bnn (NumPyro) VRW NUTS demo ===\n"
        f"PK (with ref):   KS D={D_pk:.6f}, p={p_pk:.6g}\n"
        f"No-ref ablation: KS D={D_ab:.6f}, p={p_ab:.6g}\n"
    )
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
