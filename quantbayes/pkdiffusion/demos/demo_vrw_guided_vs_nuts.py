# quantbayes/pkdiffusion/demos/demo_vrw_guided_vs_nuts.py
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as sp_beta, kstest, ks_2samp

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx

from quantbayes.stochax.diffusion.schedules.vp import make_vp_int_beta
from quantbayes.pkdiffusion.models import ScoreMLP
from quantbayes.pkdiffusion.samplers import sample_many_reverse_vp_sde_euler

from quantbayes.pkdiffusion.guidance import (
    VRWRadialRRGuidanceConfig,
    make_vrw_radial_rr_guidance,
)
from quantbayes.pkdiffusion.metrics import w2_empirical_1d, sliced_w2_empirical

from quantbayes.bnn.pkstruct.vrw_numpyro import run_nuts_vrw_pk, VRWNUTSConfig


# =========================
# USER SETTINGS (edit here)
# =========================
OUT_DIR = Path("reports/pkdiffusion/vrw_guided_vs_nuts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Where the trained diffusion prior lives
MODEL_DIR = Path("reports/pkdiffusion/vrw_endpoint_score")
MODEL_CFG = MODEL_DIR / "config.json"

# Evidence q(u) where u=r/N
Q_ALPHA = 20.0
Q_BETA = 2.0

# Guidance settings (Stephens reference)
REF_KIND = "stephens"
GUIDANCE_SCALE_DEFAULT = 1.0
NOISE_AWARE = True
NOISE_POWER = 1.0
SNR_GAMMA = 1.0

# Option: automatically pick scale from the sweep file if it exists
AUTO_PICK_SCALE_FROM_SWEEP = True
SWEEP_JSON = Path("reports/pkdiffusion/vrw_endpoint_guidance_sweep/results.json")
AUTO_PICK_BY = "SW2_x_vs_IS"  # or "W2_r_vs_IS" if you prefer

# Diffusion sampling
DIFF_NUM_SAMPLES = 3000
DIFF_NUM_STEPS = 600
DIFF_SEED_GUIDED = 0
DIFF_SEED_UNCOND = 1  # baseline

# NUTS settings
NUTS_SEED = 42
NUM_WARMUP = 1000
NUM_SAMPLES = 1500
THIN = 1  # set to 2 or 5 if you want, but increases MC error for comparisons


def _load_score_model(cfg: dict) -> eqx.Module:
    arch = cfg["model_arch"]
    template = ScoreMLP(
        dim=arch["dim"],
        time_dim=arch["time_dim"],
        width_size=arch["width_size"],
        depth=arch["depth"],
        key=jr.PRNGKey(0),
    )
    with open(cfg["model_path"], "rb") as f:
        model = eqx.tree_deserialise_leaves(f, template)
    return model


def _theta_to_endpoints(theta: np.ndarray) -> np.ndarray:
    """
    theta: (S, N)
    return endpoints: (S, 2)
    """
    steps = np.stack([np.cos(theta), np.sin(theta)], axis=-1)  # (S,N,2)
    end = np.sum(steps, axis=1)  # (S,2)
    return end


def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(x))
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def _pick_scale_from_sweep(default_scale: float) -> float:
    if not (AUTO_PICK_SCALE_FROM_SWEEP and SWEEP_JSON.exists()):
        return float(default_scale)

    try:
        obj = json.loads(SWEEP_JSON.read_text())
    except Exception:
        return float(default_scale)

    rows = obj.get("results", None)
    if rows is None and isinstance(obj, list):
        rows = obj
    if not rows:
        return float(default_scale)

    key = AUTO_PICK_BY
    if key not in rows[0]:
        return float(default_scale)

    best = min(rows, key=lambda r: float(r.get(key, np.inf)))
    return float(best["scale"])


def main():
    if not MODEL_CFG.exists():
        raise FileNotFoundError(
            "Missing trained diffusion config. Run:\n"
            "  python -m quantbayes.pkdiffusion.demos.demo_vrw_endpoint_train_score\n"
            "first."
        )

    cfg = json.loads(MODEL_CFG.read_text())
    score_model = _load_score_model(cfg)

    # Prior params from training config
    N = int(cfg["N"])
    MU = float(cfg["MU"])
    KAPPA = float(cfg["KAPPA"])

    # VP schedule (must match training)
    t1 = float(cfg["t1"])
    beta_min = float(cfg["beta_min"])
    beta_max = float(cfg["beta_max"])
    int_beta_fn = make_vp_int_beta(
        "linear", beta_min=beta_min, beta_max=beta_max, t1=t1
    )

    # Pick guidance scale
    guidance_scale = _pick_scale_from_sweep(GUIDANCE_SCALE_DEFAULT)

    print("=== VRW guided diffusion vs NUTS (PK posterior) ===")
    print(f"N={N} MU={MU} KAPPA={KAPPA}")
    print(f"Evidence on u=r/N: Beta({Q_ALPHA},{Q_BETA})")
    print(f"Reference: {REF_KIND}")
    print(
        f"Guidance: scale={guidance_scale} noise_aware={NOISE_AWARE} noise_power={NOISE_POWER} snr_gamma={SNR_GAMMA}"
    )
    print(f"Diffusion: num_samples={DIFF_NUM_SAMPLES} num_steps={DIFF_NUM_STEPS}")
    print(f"NUTS: warmup={NUM_WARMUP} samples={NUM_SAMPLES} thin={THIN}")

    # -------------------------
    # NUTS ground truth in theta space
    # -------------------------
    nuts_cfg = VRWNUTSConfig(
        N=N,
        mu=MU,
        kappa=KAPPA,
        alpha=float(Q_ALPHA),
        beta=float(Q_BETA),
        num_warmup=int(NUM_WARMUP),
        num_samples=int(NUM_SAMPLES),
        use_reference=True,  # Stephens reference ON (PK posterior)
        use_circular_reparam=True,
    )
    nuts = run_nuts_vrw_pk(jr.PRNGKey(int(NUTS_SEED)), nuts_cfg)

    theta_nuts = np.asarray(nuts["theta"])  # (S,N)
    r_nuts = np.asarray(nuts["r"])  # (S,)

    if THIN > 1:
        theta_nuts = theta_nuts[::THIN]
        r_nuts = r_nuts[::THIN]

    X_nuts = _theta_to_endpoints(theta_nuts)
    r_nuts_check = np.linalg.norm(X_nuts, axis=-1)

    # sanity: r returned by numpyro should match computed endpoint norm
    max_abs_r = float(np.max(np.abs(r_nuts_check - r_nuts)))
    print(f"NUTS sanity: max |r(theta)-r_saved| = {max_abs_r:.3e}")

    # -------------------------
    # Guided diffusion sampling in endpoint space
    # -------------------------
    gcfg = VRWRadialRRGuidanceConfig(
        N=N,
        kappa=KAPPA,
        alpha=float(Q_ALPHA),
        beta=float(Q_BETA),
        ref_kind="stephens",
        guidance_scale=float(guidance_scale),
        noise_aware=bool(NOISE_AWARE),
        noise_power=float(NOISE_POWER),
        snr_gamma=float(SNR_GAMMA),
        eps=1e-6,
        min_alpha=0.05,
        max_guidance_norm=25.0,
    )
    guidance_fn = make_vrw_radial_rr_guidance(gcfg)

    key_g = jr.PRNGKey(int(DIFF_SEED_GUIDED))
    X_guided = np.array(
        sample_many_reverse_vp_sde_euler(
            score_model,
            int_beta_fn,
            sample_shape=(2,),
            key=key_g,
            num_samples=int(DIFF_NUM_SAMPLES),
            t1=float(t1),
            num_steps=int(DIFF_NUM_STEPS),
            guidance_fn=guidance_fn,
        )
    )
    r_guided = np.linalg.norm(X_guided, axis=-1)

    # Baseline (unconditional diffusion)
    key_u = jr.PRNGKey(int(DIFF_SEED_UNCOND))
    X_uncond = np.array(
        sample_many_reverse_vp_sde_euler(
            score_model,
            int_beta_fn,
            sample_shape=(2,),
            key=key_u,
            num_samples=int(DIFF_NUM_SAMPLES),
            t1=float(t1),
            num_steps=int(DIFF_NUM_STEPS),
            guidance_fn=None,
        )
    )
    r_uncond = np.linalg.norm(X_uncond, axis=-1)

    # -------------------------
    # Metrics: guided vs NUTS
    # -------------------------
    u_nuts = r_nuts / N
    u_guided = r_guided / N
    u_uncond = r_uncond / N

    # Two-sample KS (guided vs NUTS)
    D_2s, p_2s = ks_2samp(u_guided, u_nuts)

    # One-sample KS to target Beta (helps sanity)
    D_nuts, p_nuts = kstest(u_nuts, "beta", args=(Q_ALPHA, Q_BETA))
    D_guided, p_guided = kstest(u_guided, "beta", args=(Q_ALPHA, Q_BETA))
    D_uncond, p_uncond = kstest(u_uncond, "beta", args=(Q_ALPHA, Q_BETA))

    # Distances guided <-> NUTS
    w2_r_gn = float(w2_empirical_1d(r_guided, r_nuts))
    sw2_x_gn = float(sliced_w2_empirical(X_guided, X_nuts, num_projections=256, seed=0))

    # Baseline distances uncond <-> NUTS
    w2_r_un = float(w2_empirical_1d(r_uncond, r_nuts))
    sw2_x_un = float(sliced_w2_empirical(X_uncond, X_nuts, num_projections=256, seed=0))

    frac_guided_gt1 = float(np.mean(u_guided > 1.0))
    frac_uncond_gt1 = float(np.mean(u_uncond > 1.0))
    frac_nuts_gt1 = float(np.mean(u_nuts > 1.0))

    print("\n=== Metrics ===")
    print(f"Two-sample KS (guided vs NUTS) on u=r/N: D={D_2s:.4f}, p={p_2s:.3g}")
    print(f"KS to target Beta on u=r/N:")
    print(f"  NUTS:   D={D_nuts:.4f}, p={p_nuts:.3g}, u>1={frac_nuts_gt1:.4f}")
    print(f"  guided: D={D_guided:.4f}, p={p_guided:.3g}, u>1={frac_guided_gt1:.4f}")
    print(f"  uncond: D={D_uncond:.4f}, p={p_uncond:.3g}, u>1={frac_uncond_gt1:.4f}")
    print(f"\nDistances vs NUTS:")
    print(f"  guided: W2(r)={w2_r_gn:.6f}, SW2(x)={sw2_x_gn:.6f}")
    print(f"  uncond: W2(r)={w2_r_un:.6f}, SW2(x)={sw2_x_un:.6f}")

    # -------------------------
    # Plots
    # -------------------------
    # Scatter endpoints
    plt.figure(figsize=(7, 7))
    plt.scatter(
        X_uncond[::10, 0], X_uncond[::10, 1], s=6, alpha=0.12, label="uncond diffusion"
    )
    plt.scatter(
        X_guided[::10, 0], X_guided[::10, 1], s=6, alpha=0.12, label="guided diffusion"
    )
    plt.scatter(
        X_nuts[::2, 0], X_nuts[::2, 1], s=8, alpha=0.25, label="NUTS posterior (theta)"
    )
    plt.gca().set_aspect("equal", "box")
    plt.title("VRW endpoints: uncond vs guided diffusion vs NUTS posterior")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    p_scatter = OUT_DIR / "scatter_endpoints.png"
    plt.savefig(p_scatter, dpi=200, bbox_inches="tight")
    plt.close()

    # Radial histogram + target curve
    bins = np.linspace(0.0, float(N), 81)
    plt.figure(figsize=(10, 4.5))
    plt.hist(
        r_uncond,
        bins=bins,
        density=True,
        alpha=0.30,
        label=f"uncond (u>1: {frac_uncond_gt1:.3f})",
    )
    plt.hist(
        r_guided,
        bins=bins,
        density=True,
        alpha=0.30,
        label=f"guided (u>1: {frac_guided_gt1:.3f})",
    )
    plt.hist(
        r_nuts,
        bins=bins,
        density=True,
        alpha=0.30,
        label=f"NUTS (u>1: {frac_nuts_gt1:.3f})",
    )

    rg = np.linspace(1e-4, float(N) - 1e-4, 500)
    ug = rg / float(N)
    target = (1.0 / float(N)) * sp_beta.pdf(ug, Q_ALPHA, Q_BETA)
    plt.plot(
        rg, target, linewidth=2, label=f"target q(r): Beta({Q_ALPHA},{Q_BETA}) on r/N"
    )

    plt.title("Radial distribution r=||endpoint||")
    plt.xlabel("r")
    plt.ylabel("density")
    plt.legend()
    p_hist = OUT_DIR / "radial_hist.png"
    plt.savefig(p_hist, dpi=200, bbox_inches="tight")
    plt.close()

    # CDF plot in u=r/N
    plt.figure(figsize=(8, 5))
    for u, name in [(u_uncond, "uncond"), (u_guided, "guided"), (u_nuts, "NUTS")]:
        xs, ys = _ecdf(np.clip(u, 0.0, 1.0))
        plt.plot(xs, ys, linewidth=2, label=name)
    ugrid = np.linspace(0.0, 1.0, 500)
    plt.plot(
        ugrid, sp_beta.cdf(ugrid, Q_ALPHA, Q_BETA), linewidth=2, label="target Beta CDF"
    )
    plt.title("CDFs of u=r/N (clipped to [0,1])")
    plt.xlabel("u")
    plt.ylabel("CDF")
    plt.legend()
    p_cdf = OUT_DIR / "u_cdf.png"
    plt.savefig(p_cdf, dpi=200, bbox_inches="tight")
    plt.close()

    # -------------------------
    # Save summary JSON
    # -------------------------
    metrics = dict(
        cfg=dict(
            N=N,
            MU=MU,
            KAPPA=KAPPA,
            q_alpha=float(Q_ALPHA),
            q_beta=float(Q_BETA),
            ref_kind=REF_KIND,
            guidance_scale=float(guidance_scale),
            noise_aware=bool(NOISE_AWARE),
            noise_power=float(NOISE_POWER),
            snr_gamma=float(SNR_GAMMA),
            diffusion=dict(
                num_samples=int(DIFF_NUM_SAMPLES),
                num_steps=int(DIFF_NUM_STEPS),
                t1=float(t1),
                beta_min=float(beta_min),
                beta_max=float(beta_max),
            ),
            nuts=dict(
                num_warmup=int(NUM_WARMUP),
                num_samples=int(NUM_SAMPLES),
                thin=int(THIN),
                seed=int(NUTS_SEED),
            ),
        ),
        ks=dict(
            two_sample_guided_vs_nuts=dict(D=float(D_2s), p=float(p_2s)),
            one_sample_to_beta=dict(
                nuts=dict(D=float(D_nuts), p=float(p_nuts)),
                guided=dict(D=float(D_guided), p=float(p_guided)),
                uncond=dict(D=float(D_uncond), p=float(p_uncond)),
            ),
        ),
        support_mass_u_gt_1=dict(
            nuts=float(frac_nuts_gt1),
            guided=float(frac_guided_gt1),
            uncond=float(frac_uncond_gt1),
        ),
        distances_vs_nuts=dict(
            guided=dict(W2_r=float(w2_r_gn), SW2_x=float(sw2_x_gn)),
            uncond=dict(W2_r=float(w2_r_un), SW2_x=float(sw2_x_un)),
        ),
        sanity=dict(max_abs_r_theta_vs_saved=float(max_abs_r)),
        files=dict(scatter=str(p_scatter), radial_hist=str(p_hist), u_cdf=str(p_cdf)),
    )

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nSaved:")
    print(" ", OUT_DIR / "metrics.json")
    print(" ", p_scatter)
    print(" ", p_hist)
    print(" ", p_cdf)


if __name__ == "__main__":
    main()
