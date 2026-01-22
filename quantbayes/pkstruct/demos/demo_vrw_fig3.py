from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as sp_beta, kstest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import numpyro.distributions as dist

from quantbayes.pkstruct.toy.vrw import stephens_logpdf_r, StephensConfig
from quantbayes.bnn.pkstruct.vrw_numpyro import run_nuts_vrw_pk, VRWNUTSConfig


# =========================
# USER SETTINGS (edit here)
# =========================
OUT_DIR = Path("reports/vrw_fig3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paper settings
N = 5
MU = 0.0
KAPPA = 10.0
ALPHA = 10.0
BETA = 10.0

# Inference
NUM_WARMUP = 1000
NUM_SAMPLES = 1000
THIN = 5

# Prior sample size for histogram
PRIOR_DRAWS = 50_000

# Trajectory panel
NUM_TRAJ = 500
TRAJ_SEED = 123  # for selecting which posterior samples to show

# Simple sanity thresholds (robust)
MAX_D_PK = 0.25
MIN_D_ABLATION = 0.80


def _theta_to_traj_batch(theta: np.ndarray) -> np.ndarray:
    """
    theta: (S, N)
    returns traj: (S, N+1, 2)
      traj[:,0] = (0,0)
      traj[:,t] = sum_{i<=t} (cos theta_i, sin theta_i)
    """
    steps = np.stack([np.cos(theta), np.sin(theta)], axis=-1)  # (S,N,2)
    pos = np.cumsum(steps, axis=1)  # (S,N,2)
    origin = np.zeros((theta.shape[0], 1, 2), dtype=pos.dtype)
    traj = np.concatenate([origin, pos], axis=1)  # (S,N+1,2)
    return traj


def main():
    # --- NUTS posterior (with reference) ---
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
    s_pk = run_nuts_vrw_pk(jr.PRNGKey(0), cfg_pk)
    r_pk = s_pk["r"]
    theta_pk = s_pk["theta"]  # (num_samples, N)

    # --- NUTS ablation (no reference) ---
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
    s_ab = run_nuts_vrw_pk(jr.PRNGKey(1), cfg_ab)
    r_ab = s_ab["r"]

    # --- Empirical prior samples (for histogram) ---
    theta_prior = np.array(
        dist.VonMises(MU, KAPPA)
        .expand([N])
        .to_event(1)
        .sample(jr.PRNGKey(2), (PRIOR_DRAWS,))
    )
    r_prior = np.linalg.norm(
        np.stack([np.cos(theta_prior), np.sin(theta_prior)], axis=-1).sum(axis=1),
        axis=-1,
    )

    # --- KS test on r/N ~ Beta(alpha,beta) ---
    r_pk_s = r_pk[::THIN] / N
    r_ab_s = r_ab[::THIN] / N

    D_pk, p_pk = kstest(r_pk_s, "beta", args=(ALPHA, BETA))
    D_ab, p_ab = kstest(r_ab_s, "beta", args=(ALPHA, BETA))

    print("=== VRW PK Fig.3 demo ===")
    print(f"PK (with ref):   KS D={D_pk:.3f}, p={p_pk:.3g}")
    print(f"No-ref ablation: KS D={D_ab:.3f}, p={p_ab:.3g}")

    # Assertions (fail loudly if something breaks)
    assert D_pk < MAX_D_PK, f"PK KS D too large: {D_pk:.3f} (threshold {MAX_D_PK})"
    assert (
        D_ab > MIN_D_ABLATION
    ), f"Ablation KS D too small: {D_ab:.3f} (threshold {MIN_D_ABLATION})"
    print("PASS: KS sanity thresholds satisfied.")

    # --- Theoretical curves for the left panel ---
    r_grid = np.linspace(1e-4, N - 1e-4, 400)
    x_grid = r_grid / N
    scaledbeta_pdf = (1.0 / N) * sp_beta.pdf(x_grid, ALPHA, BETA)

    steph_cfg = StephensConfig(kappa=KAPPA, N=N)
    stephens_lp = jax.vmap(lambda rr: stephens_logpdf_r(rr, cfg=steph_cfg))(
        jnp.asarray(r_grid)
    )
    stephens_pdf = np.exp(np.array(stephens_lp))

    # --- Trajectory samples for right panel ---
    # Prior trajectories: sample theta ~ prior
    theta_prior_traj = np.array(
        dist.VonMises(MU, KAPPA)
        .expand([N])
        .to_event(1)
        .sample(jr.PRNGKey(3), (NUM_TRAJ,))
    )

    # Posterior trajectories: choose NUM_TRAJ theta samples from NUTS output
    rng = np.random.default_rng(TRAJ_SEED)
    if theta_pk.shape[0] < NUM_TRAJ:
        raise ValueError(
            f"Need at least {NUM_TRAJ} posterior samples, got {theta_pk.shape[0]}."
        )
    idx = rng.choice(theta_pk.shape[0], size=NUM_TRAJ, replace=False)
    theta_post_traj = theta_pk[idx]

    traj_prior = _theta_to_traj_batch(theta_prior_traj)  # (S,N+1,2)
    traj_post = _theta_to_traj_batch(theta_post_traj)

    # --- Make a single two-panel figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    # Left: histogram + theoretical curves
    ax = axes[0]
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)  # grid behind bars/lines
    ax.hist(
        r_prior,
        bins=80,
        density=True,
        alpha=0.35,
        color="tab:blue",
        label="Empirical prior (VRW)",
    )
    ax.hist(
        r_pk,
        bins=80,
        density=True,
        alpha=0.35,
        color="tab:red",
        label="Empirical posterior (PK)",
    )
    ax.hist(
        r_ab, bins=80, density=True, alpha=0.35, color="0.6", label="Naive (no ref)"
    )

    ax.plot(r_grid, scaledbeta_pdf, color="k", label="Target (ScaledBeta)", linewidth=2)
    ax.plot(
        r_grid,
        stephens_pdf,
        color="k",
        linestyle="--",
        label="Reference (Stephens)",
        linewidth=2,
    )

    ax.set_xlabel("r (endpoint distance)")
    ax.set_ylabel("density")
    ax.set_title("Marginal over r (numpyro/JAX) reproduction")
    ax.legend()

    c_prior = "tab:blue"
    c_post = "tab:red"
    # Right: 500 trajectories (prior vs PK)
    ax = axes[1]

    # Plot many faint trajectories + endpoints.
    # Prior
    for t in range(NUM_TRAJ):
        ax.plot(
            traj_prior[t, :, 0],
            traj_prior[t, :, 1],
            color=c_prior,
            alpha=0.06,
            linewidth=0.9,
        )

    for t in range(NUM_TRAJ):
        ax.plot(
            traj_post[t, :, 0],
            traj_post[t, :, 1],
            color=c_post,
            alpha=0.06,
            linewidth=0.9,
        )

    # Posterior
    for t in range(NUM_TRAJ):
        ax.plot(traj_post[t, :, 0], traj_post[t, :, 1], alpha=0.08, linewidth=0.8)
    ax.scatter(
        traj_prior[:, -1, 0],
        traj_prior[:, -1, 1],
        s=14,
        alpha=0.85,
        color=c_prior,
        marker="o",
        edgecolors="none",
        zorder=5,
        label="Prior endpoints",
    )

    ax.scatter(
        traj_post[:, -1, 0],
        traj_post[:, -1, 1],
        s=14,
        alpha=0.85,
        color=c_post,
        marker="o",
        edgecolors="none",
        zorder=6,
        label="PK endpoints",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{NUM_TRAJ} trajectories (prior vs PK) (numpyro/JAX) reproduction")
    ax.legend(loc="upper right")

    fig.suptitle(
        "VRW PK reproduction (Figure 3-style: marginal + trajectories)", y=1.02
    )
    fig.tight_layout()

    fig_path = OUT_DIR / "vrw_fig3.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {fig_path}")

    summary_path = OUT_DIR / "summary.txt"
    summary_path.write_text(
        "=== VRW PK Fig.3 demo ===\n"
        f"N={N} mu={MU} kappa={KAPPA} alpha={ALPHA} beta={BETA}\n"
        f"PK (with ref):   KS D={D_pk:.6f}, p={p_pk:.6g}\n"
        f"No-ref ablation: KS D={D_ab:.6f}, p={p_ab:.6g}\n"
        f"Trajectories plotted: {NUM_TRAJ}\n"
    )
    print(f"Saved summary: {summary_path}")

    # Optional: show interactively
    plt.show()


if __name__ == "__main__":
    main()
