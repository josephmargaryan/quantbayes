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
from quantbayes.pkdiffusion.metrics import w2_empirical_1d, sliced_w2_empirical
from quantbayes.pkstruct.toy.vrw import stephens_logpdf_r, StephensConfig


OUT_DIR = Path("reports/pkdiffusion/vrw_endpoint_guided")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path("reports/pkdiffusion/vrw_endpoint_score")
MODEL_CFG = MODEL_DIR / "config.json"
MODEL_FILE = MODEL_DIR / "ema_score_model.eqx"


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


def _fit_beta_moments(u: np.ndarray, eps: float = 1e-6) -> tuple[float, float, dict]:
    """
    Fit Beta(a,b) on [0,1] by method of moments.
    Returns (a,b,stats).
    """
    u = np.asarray(u, dtype=float)
    u = np.clip(u, eps, 1.0 - eps)
    m = float(np.mean(u))
    v = float(np.var(u))

    # Ensure feasible variance
    vmax = m * (1.0 - m) - 1e-12
    if v <= 1e-12:
        v = 1e-12
    if v >= vmax:
        v = 0.99 * vmax

    t = m * (1.0 - m) / v - 1.0
    a = max(m * t, 1e-3)
    b = max((1.0 - m) * t, 1e-3)

    stats = {"mean": m, "var": v, "std": float(np.sqrt(v))}
    return float(a), float(b), stats


def _sample_vrw_endpoints(
    *, N: int, MU: float, KAPPA: float, draws: int, seed: int
) -> np.ndarray:
    key = jr.PRNGKey(seed)
    theta = dist.VonMises(MU, KAPPA).expand([N]).to_event(1).sample(key, (int(draws),))
    end = jax.vmap(vrw_endpoint)(theta)  # (draws,2)
    return np.array(end)


def _importance_resample(
    X_prop: np.ndarray,
    *,
    N: int,
    q_alpha: float,
    q_beta: float,
    ref_kind: str,
    kappa: float | None,
    ref_alpha: float | None,
    ref_beta: float | None,
    num_samples: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """
    Importance resampling baseline:
      w(r) ∝ q(r) / ref(r)

    ref_kind:
      - "stephens": ref(r)=Stephens approximation
      - "beta":     ref(r)=ScaledBeta(ref_alpha, ref_beta) on r/N

    Returns (resampled_endpoints, diagnostics).
    """
    rng = np.random.default_rng(seed)

    r = np.linalg.norm(X_prop, axis=-1)
    rj = jnp.asarray(r)

    # log q(r)
    logq = np.array(
        jax.vmap(lambda rr: log_scaled_beta_pdf(rr, q_alpha, q_beta, N))(rj)
    )

    # log ref(r)
    if ref_kind == "stephens":
        if kappa is None:
            raise ValueError("kappa must be provided when ref_kind='stephens'")
        scfg = StephensConfig(kappa=float(kappa), N=int(N))
        logref = np.array(jax.vmap(lambda rr: stephens_logpdf_r(rr, cfg=scfg))(rj))
    elif ref_kind == "beta":
        if ref_alpha is None or ref_beta is None:
            raise ValueError("ref_alpha/ref_beta must be provided when ref_kind='beta'")
        logref = np.array(
            jax.vmap(lambda rr: log_scaled_beta_pdf(rr, ref_alpha, ref_beta, N))(rj)
        )
    else:
        raise ValueError(f"Unknown ref_kind={ref_kind!r}")

    logw = logq - logref

    # Stabilize + avoid full underflow
    logw = logw - np.max(logw)
    logw = np.clip(logw, -80.0, 0.0)

    w = np.exp(logw)
    w = w / np.sum(w)

    ess = float(1.0 / np.sum(w**2))
    w_max = float(np.max(w))

    idx = rng.choice(X_prop.shape[0], size=int(num_samples), replace=True, p=w)
    unique = int(np.unique(idx).size)

    diag = {
        "ess": ess,
        "ess_frac": float(ess / X_prop.shape[0]),
        "w_max": w_max,
        "unique_idx": unique,
        "unique_frac": float(unique / num_samples),
        "ref_kind": ref_kind,
    }
    return X_prop[idx], diag


def main():
    REF_KIND = "stephens"
    if not MODEL_CFG.exists() or not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Missing trained model. Run:\n"
            "  python -m quantbayes.pkdiffusion.demos.demo_vrw_endpoint_train_score\n"
            "first."
        )

    cfg = json.loads(MODEL_CFG.read_text())
    score_model = _load_score_model(cfg)

    # VRW prior params (from training config)
    N = int(cfg["N"])
    MU = float(cfg["MU"])
    KAPPA = float(cfg["KAPPA"])

    # -------------------------
    # Evidence q(r) on (0,N)
    # -------------------------
    # Start with a MODERATE shift first (so the demo is clean).
    # After it works, try the hard stress-test:
    #   q_alpha=10.0; q_beta=10.0
    q_alpha = 20.0
    q_beta = 2.0

    # Diffusion schedule (must match training)
    t1 = float(cfg["t1"])
    beta_min = float(cfg["beta_min"])
    beta_max = float(cfg["beta_max"])
    int_beta_fn = make_vp_int_beta(
        "linear", beta_min=beta_min, beta_max=beta_max, t1=t1
    )

    sampler_cfg = VPSDESamplerConfig(
        t1=t1,
        num_steps=600,
        num_samples=3000,
        seed=0,
    )

    # -------------------------
    # Fit ref(r) as a Beta on r/N using prior samples
    # -------------------------
    draws_prop = 250_000
    X_prop = _sample_vrw_endpoints(N=N, MU=MU, KAPPA=KAPPA, draws=draws_prop, seed=123)

    r_prop = np.linalg.norm(X_prop, axis=-1)
    u_prop = np.clip(r_prop / N, 1e-6, 1.0 - 1e-6)

    ref_alpha, ref_beta, ref_stats = _fit_beta_moments(u_prop)

    print("=== Reference fit (prior r/N) ===")
    print(f"ref_kind = beta-fit")
    print(f"ref_alpha = {ref_alpha:.6f}, ref_beta = {ref_beta:.6f}")
    print(f"ref mean(u)= {ref_stats['mean']:.6f}, std(u)= {ref_stats['std']:.6f}")
    print(f"q   mean(u)= {q_alpha/(q_alpha+q_beta):.6f}")

    # -------------------------
    # Guidance using ref_kind="beta"
    # -------------------------
    gcfg = VRWRadialRRGuidanceConfig(
        N=N,
        kappa=KAPPA,
        alpha=q_alpha,
        beta=q_beta,
        ref_kind=REF_KIND,
        guidance_scale=1.0,  # IMPORTANT: start smaller than 2.0
        noise_aware=True,
        noise_power=1.0,
        eps=1e-6,
        min_alpha=0.05,
        max_guidance_norm=25.0,
        snr_gamma=1.0,
    )
    guidance_fn = make_vrw_radial_rr_guidance(gcfg)

    # -------------------------
    # Sample: unconditional vs guided diffusion
    # -------------------------
    key = jr.PRNGKey(sampler_cfg.seed)
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
    # Baseline: IS resampling from VRW prior
    # -------------------------
    X_is, is_diag = _importance_resample(
        X_prop,
        N=N,
        q_alpha=q_alpha,
        q_beta=q_beta,
        ref_kind=REF_KIND,
        kappa=KAPPA,
        ref_alpha=ref_alpha,
        ref_beta=ref_beta,
        num_samples=sampler_cfg.num_samples,
        seed=999,
    )

    # Finite filtering (should be 100%)
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
    print("IS diagnostics:", is_diag)

    # -------------------------
    # Metrics on r
    # -------------------------
    r_u = np.linalg.norm(X_uncond_np, axis=-1)
    r_g = np.linalg.norm(X_guided_np, axis=-1)
    r_is = np.linalg.norm(X_is, axis=-1)

    u_u = r_u / N
    u_g = r_g / N
    u_is = r_is / N

    # Out-of-support mass (should be near 0)
    frac_u_gt1 = float(np.mean(u_u > 1.0))
    frac_g_gt1 = float(np.mean(u_g > 1.0))
    frac_is_gt1 = float(np.mean(u_is > 1.0))

    # KS against Beta on u=r/N
    D_u, p_u = kstest(u_u, "beta", args=(q_alpha, q_beta))
    D_g, p_g = kstest(u_g, "beta", args=(q_alpha, q_beta))
    D_is, p_is = kstest(u_is, "beta", args=(q_alpha, q_beta))

    # Wasserstein distances vs IS baseline
    w2_r_g_is = w2_empirical_1d(r_g, r_is)
    sw2_x_g_is = sliced_w2_empirical(X_guided_np, X_is, num_projections=256, seed=0)

    # -------------------------
    # Plots (with shared bins!)
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
        X_is[::8, 0], X_is[::8, 1], s=6, alpha=0.15, label="IS baseline (VRW + weights)"
    )
    plt.gca().set_aspect("equal", "box")
    plt.title("VRW endpoints: uncond vs guided diffusion vs IS baseline")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    scatter_path = OUT_DIR / "scatter_endpoints.png"
    plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Radial histogram + target curve (shared bins)
    bins = np.linspace(0.0, float(N), 81)

    def _in_support(r):
        r = np.asarray(r)
        m = (r >= 0.0) & (r <= float(N))
        return r[m]

    r_u_plot = _in_support(r_u)
    r_g_plot = _in_support(r_g)
    r_is_plot = _in_support(r_is)

    plt.figure(figsize=(10, 4.5))
    plt.hist(
        r_u_plot,
        bins=bins,
        density=True,
        alpha=0.35,
        label=f"uncond r (u>1: {frac_u_gt1:.3f})",
    )
    plt.hist(
        r_g_plot,
        bins=bins,
        density=True,
        alpha=0.35,
        label=f"guided r (u>1: {frac_g_gt1:.3f})",
    )
    plt.hist(
        r_is_plot,
        bins=bins,
        density=True,
        alpha=0.35,
        label=f"IS r (u>1: {frac_is_gt1:.3f})",
    )

    rg = np.linspace(1e-4, float(N) - 1e-4, 400)
    xg = rg / float(N)
    target = (1.0 / float(N)) * sp_beta.pdf(xg, q_alpha, q_beta)
    plt.plot(
        rg, target, linewidth=2, label=f"target q(r): Beta({q_alpha},{q_beta}) on r/N"
    )

    plt.title("Radial distribution r = ||endpoint|| (shared bins)")
    plt.xlabel("r")
    plt.ylabel("density")
    plt.legend()
    hist_path = OUT_DIR / "radial_hist.png"
    plt.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close()

    # CDF plot in u=r/N (bin-free diagnostic)
    def _ecdf(x):
        x = np.sort(np.asarray(x))
        y = np.arange(1, x.size + 1) / x.size
        return x, y

    plt.figure(figsize=(8, 5))
    for u, name in [(u_u, "uncond"), (u_g, "guided"), (u_is, "IS")]:
        xs, ys = _ecdf(np.clip(u, 0.0, 1.0))
        plt.plot(xs, ys, linewidth=2, label=name)

    ug = np.linspace(0.0, 1.0, 500)
    plt.plot(ug, sp_beta.cdf(ug, q_alpha, q_beta), linewidth=2, label="target Beta CDF")

    plt.title("CDFs of u=r/N (clipped to [0,1])")
    plt.xlabel("u")
    plt.ylabel("CDF")
    plt.legend()
    cdf_path = OUT_DIR / "radial_cdf.png"
    plt.savefig(cdf_path, dpi=200, bbox_inches="tight")
    plt.close()

    # -------------------------
    # Save summary
    # -------------------------
    summary = {
        "N": N,
        "MU": MU,
        "KAPPA": KAPPA,
        "evidence_beta_on_u": {"alpha": q_alpha, "beta": q_beta},
        "ref_beta_fit_on_u": {"alpha": ref_alpha, "beta": ref_beta, **ref_stats},
        "sampler": {
            "t1": sampler_cfg.t1,
            "num_steps": sampler_cfg.num_steps,
            "num_samples": sampler_cfg.num_samples,
        },
        "guidance": {"ref_kind": gcfg.ref_kind, "guidance_scale": gcfg.guidance_scale},
        "finite_fraction": {"uncond": frac_u, "guided": frac_g, "is": frac_is},
        "support_mass_u_gt_1": {
            "uncond": frac_u_gt1,
            "guided": frac_g_gt1,
            "is": frac_is_gt1,
        },
        "ks_beta_on_u": {
            "uncond": {"D": float(D_u), "p": float(p_u)},
            "guided": {"D": float(D_g), "p": float(p_g)},
            "is": {"D": float(D_is), "p": float(p_is)},
        },
        "is_diag": is_diag,
        "dist_vs_is": {"w2_r": float(w2_r_g_is), "sw2_x": float(sw2_x_g_is)},
        "files": {
            "scatter": str(scatter_path),
            "radial_hist": str(hist_path),
            "radial_cdf": str(cdf_path),
        },
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))

    txt = (
        "=== VRW endpoint guided diffusion demo (stephens-ref) ===\n"
        f"Saved scatter:    {scatter_path}\n"
        f"Saved radial hist:{hist_path}\n"
        f"Saved radial CDF: {cdf_path}\n\n"
        f"Ref fit on u=r/N: Beta({ref_alpha:.4f},{ref_beta:.4f}), mean={ref_stats['mean']:.4f}, std={ref_stats['std']:.4f}\n"
        f"Evidence on u:    Beta({q_alpha:.1f},{q_beta:.1f}), mean={q_alpha/(q_alpha+q_beta):.4f}\n\n"
        f"IS diagnostics: ESS={is_diag['ess']:.1f} ({is_diag['ess_frac']:.3g} of proposal), "
        f"w_max={is_diag['w_max']:.3g}, unique_frac={is_diag['unique_frac']:.3f}\n\n"
        f"KS (u=r/N vs Beta({q_alpha},{q_beta})):\n"
        f"  uncond: D={D_u:.4f}, p={p_u:.3g} (u>1: {frac_u_gt1:.3f})\n"
        f"  guided: D={D_g:.4f}, p={p_g:.3g} (u>1: {frac_g_gt1:.3f})\n"
        f"  IS:     D={D_is:.4f}, p={p_is:.3g} (u>1: {frac_is_gt1:.3f})\n\n"
        f"Distances vs IS baseline:\n"
        f"  W2(r):  {w2_r_g_is:.6f}\n"
        f"  SW2(x): {sw2_x_g_is:.6f}\n"
    )
    (OUT_DIR / "summary.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
