# quantbayes/pkdiffusion/demos/demo_vrw_endpoint_guided_sampling.py
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as sp_beta, kstest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx
import numpyro.distributions as dist

from quantbayes.pkstruct.toy.vrw import vrw_endpoint
from quantbayes.pkstruct.utils.stats import log_scaled_beta_pdf
from quantbayes.pkstruct.toy.vrw import stephens_logpdf_r, StephensConfig

from quantbayes.stochax.diffusion.sde import make_weight_fn
from quantbayes.stochax.diffusion.schedules.vp import make_vp_int_beta

from quantbayes.pkdiffusion.models import ScoreMLP
from quantbayes.pkdiffusion.samplers import (
    sample_many_reverse_vp_sde_euler,
    VPSDESamplerConfig,
)
from quantbayes.pkdiffusion.guidance import (
    VRWRadialRRGuidanceConfig,
    make_vrw_radial_rr_guidance,
)
from quantbayes.pkdiffusion.metrics import (
    w2_empirical_1d,
    sliced_w2_empirical,
    w2_empirical_2d_hungarian,
)


OUT_DIR = Path("reports/pkdiffusion/vrw_endpoint_guided")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path("reports/pkdiffusion/vrw_endpoint_score")
MODEL_CFG = MODEL_DIR / "config.json"
MODEL_FILE = MODEL_DIR / "ema_score_model.eqx"


def _load_score_model(cfg: dict) -> eqx.Module:
    arch = cfg["model_arch"]
    # Template model (params overwritten by deserialise)
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


def _importance_resample_posterior_endpoints(
    *,
    N: int,
    MU: float,
    KAPPA: float,
    alpha: float,
    beta: float,
    num_samples: int,
    draws: int,
    seed: int,
) -> np.ndarray:
    """
    Baseline: sample endpoints from prior, weight by w(r)=q(r)/ref(r),
    and resample.

    This targets the same unnormalised endpoint posterior implied by
    multiplying the prior by exp(log_evidence - log_reference).
    """
    rng = np.random.default_rng(seed)
    key = jr.PRNGKey(seed)

    theta = dist.VonMises(MU, KAPPA).expand([N]).to_event(1).sample(key, (draws,))
    end = jax.vmap(vrw_endpoint)(theta)  # (draws,2)
    end_np = np.array(end)

    r = np.linalg.norm(end_np, axis=-1)
    # weights in log space
    steph_cfg = StephensConfig(kappa=KAPPA, N=N)
    logw = np.array(
        jax.vmap(lambda rr: log_scaled_beta_pdf(rr, alpha, beta, N))(jnp.asarray(r))
    )
    logw -= np.array(
        jax.vmap(lambda rr: stephens_logpdf_r(rr, cfg=steph_cfg))(jnp.asarray(r))
    )
    logw = logw - np.max(logw)
    w = np.exp(logw)
    w = w / np.sum(w)

    idx = rng.choice(end_np.shape[0], size=num_samples, replace=True, p=w)
    return end_np[idx]


def main():
    # -------------------------
    # Load trained model config
    # -------------------------
    if not MODEL_CFG.exists() or not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Missing trained model. Run:\n"
            "  python -m quantbayes.pkdiffusion.demos.demo_vrw_endpoint_train_score\n"
            "first."
        )

    cfg = json.loads(MODEL_CFG.read_text())
    score_model = _load_score_model(cfg)

    # -------------------------
    # Experiment config
    # -------------------------
    N = int(cfg["N"])
    MU = float(cfg["MU"])
    KAPPA = float(cfg["KAPPA"])

    # Evidence q(r): ScaledBeta(alpha,beta) on r in (0,N)
    ALPHA = 10.0
    BETA = 10.0

    # Diffusion sampler config (must match training t1 / betas)
    t1 = float(cfg["t1"])
    beta_min = float(cfg["beta_min"])
    beta_max = float(cfg["beta_max"])

    sampler_cfg = VPSDESamplerConfig(
        t1=t1,
        num_steps=600,
        num_samples=3000,
        seed=0,
    )

    # Build int_beta_fn (must match training)
    int_beta_fn = make_vp_int_beta(
        "linear", beta_min=beta_min, beta_max=beta_max, t1=t1
    )
    _ = make_weight_fn(
        int_beta_fn, name="likelihood"
    )  # not used, but kept for symmetry

    # Guidance
    gcfg = VRWRadialRRGuidanceConfig(
        N=N,
        kappa=KAPPA,
        alpha=ALPHA,
        beta=BETA,
        guidance_scale=1.0,  # start safer; increase to 2.0 after it’s stable
        eps=1e-6,
        min_alpha=0.05,  # IMPORTANT for t1=10.0
        max_guidance_norm=25.0,  # safety
        snr_gamma=1.0,
    )
    guidance_fn = make_vrw_radial_rr_guidance(gcfg)

    # -------------------------
    # Sample: unconditional (prior) diffusion vs guided diffusion
    # -------------------------
    key = jr.PRNGKey(sampler_cfg.seed)

    # unconditional
    key_u, key_g = jr.split(key)
    X_uncond = sample_many_reverse_vp_sde_euler(
        score_model,
        int_beta_fn,
        sample_shape=(2,),
        key=key_u,
        num_samples=sampler_cfg.num_samples,
        t1=sampler_cfg.t1,
        num_steps=sampler_cfg.num_steps,
        guidance_fn=None,
    )
    X_uncond_np = np.array(X_uncond)

    # guided
    X_guided = sample_many_reverse_vp_sde_euler(
        score_model,
        int_beta_fn,
        sample_shape=(2,),
        key=key_g,
        num_samples=sampler_cfg.num_samples,
        t1=sampler_cfg.t1,
        num_steps=sampler_cfg.num_steps,
        guidance_fn=guidance_fn,
    )
    X_guided_np = np.array(X_guided)

    # -------------------------
    # Baseline: importance resampling
    # -------------------------
    X_is = _importance_resample_posterior_endpoints(
        N=N,
        MU=MU,
        KAPPA=KAPPA,
        alpha=ALPHA,
        beta=BETA,
        num_samples=sampler_cfg.num_samples,
        draws=200_000,
        seed=123,
    )

    def _finite(X):
        X = np.asarray(X)
        m = np.isfinite(X).all(axis=1)
        return X[m], float(np.mean(m))

    X_uncond_np, frac_u = _finite(X_uncond_np)
    X_guided_np, frac_g = _finite(X_guided_np)
    X_is, frac_is = _finite(X_is)

    print(
        f"Finite fraction: uncond={frac_u:.4f}, guided={frac_g:.4f}, IS={frac_is:.4f}"
    )

    # -------------------------
    # Metrics on r
    # -------------------------
    r_u = np.linalg.norm(X_uncond_np, axis=-1)
    r_g = np.linalg.norm(X_guided_np, axis=-1)
    r_is = np.linalg.norm(X_is, axis=-1)

    # KS against Beta on r/N
    D_u, p_u = kstest(r_u / N, "beta", args=(ALPHA, BETA))
    D_g, p_g = kstest(r_g / N, "beta", args=(ALPHA, BETA))
    D_is, p_is = kstest(r_is / N, "beta", args=(ALPHA, BETA))

    # Wasserstein distances vs IS baseline
    w2_r_g_is = w2_empirical_1d(r_g, r_is)
    w2_x_g_is = sliced_w2_empirical(
        X_guided_np,
        X_is,
        num_projections=256,
        seed=0,
    )
    # -------------------------
    # Plots
    # -------------------------
    # Scatter
    plt.figure(figsize=(7, 7))
    plt.scatter(
        X_uncond_np[::8, 0],
        X_uncond_np[::8, 1],
        s=6,
        alpha=0.15,
        label="uncond diffusion",
    )
    plt.scatter(
        X_guided_np[::8, 0],
        X_guided_np[::8, 1],
        s=6,
        alpha=0.15,
        label="guided diffusion",
    )
    plt.scatter(
        X_is[::8, 0], X_is[::8, 1], s=6, alpha=0.15, label="IS baseline (RR weights)"
    )
    plt.gca().set_aspect("equal", "box")
    plt.title("VRW endpoints: unconditional vs guided diffusion vs IS baseline")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    scatter_path = OUT_DIR / "scatter_endpoints.png"
    plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Radial histogram + target curve
    plt.figure(figsize=(10, 4.5))
    plt.hist(r_u, bins=80, density=True, alpha=0.35, label="uncond diffusion r")
    plt.hist(r_g, bins=80, density=True, alpha=0.35, label="guided diffusion r")
    plt.hist(r_is, bins=80, density=True, alpha=0.35, label="IS baseline r")

    rg = np.linspace(1e-4, N - 1e-4, 400)
    xg = rg / N
    target = (1.0 / N) * sp_beta.pdf(xg, ALPHA, BETA)
    plt.plot(rg, target, linewidth=2, label="target q(r) (ScaledBeta)")

    plt.title("Radial distribution r = ||endpoint||")
    plt.xlabel("r")
    plt.ylabel("density")
    plt.legend()
    hist_path = OUT_DIR / "radial_hist.png"
    plt.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close()

    # -------------------------
    # Save summary
    # -------------------------
    summary = {
        "N": N,
        "MU": MU,
        "KAPPA": KAPPA,
        "ALPHA": ALPHA,
        "BETA": BETA,
        "sampler": dict(
            t1=sampler_cfg.t1,
            num_steps=sampler_cfg.num_steps,
            num_samples=sampler_cfg.num_samples,
        ),
        "guidance": dict(
            guidance_scale=gcfg.guidance_scale,
        ),
        "ks_beta_r_over_N": {
            "uncond": {"D": float(D_u), "p": float(p_u)},
            "guided": {"D": float(D_g), "p": float(p_g)},
            "is": {"D": float(D_is), "p": float(p_is)},
        },
        "w2_vs_is": {
            "w2_r": float(w2_r_g_is),
            "w2_x_hungarian_n600": float(w2_x_g_is),
        },
        "files": {
            "scatter": str(scatter_path),
            "radial_hist": str(hist_path),
        },
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))

    txt = (
        "=== VRW endpoint guided diffusion demo ===\n"
        f"Saved scatter: {scatter_path}\n"
        f"Saved radial hist: {hist_path}\n\n"
        f"KS (r/N vs Beta({ALPHA},{BETA})):\n"
        f"  uncond: D={D_u:.4f}, p={p_u:.3g}\n"
        f"  guided: D={D_g:.4f}, p={p_g:.3g}\n"
        f"  IS:     D={D_is:.4f}, p={p_is:.3g}\n\n"
        f"W2 vs IS baseline:\n"
        f"  W2(r): {w2_r_g_is:.6f}\n"
        f"  W2(x) (Hungarian, n=600): {w2_x_g_is:.6f}\n"
    )
    (OUT_DIR / "summary.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
