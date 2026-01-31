# quantbayes/pkdiffusion/demos/demo_vp_gaussian_guidance_sweep.py
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx
import numpyro.distributions as dist

from quantbayes.pkstruct.toy.vrw import vrw_endpoint
from quantbayes.pkstruct.utils.stats import log_scaled_beta_pdf
from quantbayes.pkstruct.toy.vrw import stephens_logpdf_r, StephensConfig

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


OUT_DIR = Path("reports/pkdiffusion/vrw_endpoint_guidance_sweep")
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


def _sample_vrw_endpoints(
    *, N: int, MU: float, KAPPA: float, draws: int, seed: int
) -> np.ndarray:
    key = jr.PRNGKey(seed)
    theta = dist.VonMises(MU, KAPPA).expand([N]).to_event(1).sample(key, (int(draws),))
    end = jax.vmap(vrw_endpoint)(theta)
    return np.array(end)


def _importance_resample_stephens(
    X_prop: np.ndarray,
    *,
    N: int,
    kappa: float,
    q_alpha: float,
    q_beta: float,
    num_samples: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)

    r = np.linalg.norm(X_prop, axis=-1)
    rj = jnp.asarray(r)

    logq = np.array(
        jax.vmap(lambda rr: log_scaled_beta_pdf(rr, q_alpha, q_beta, N))(rj)
    )
    scfg = StephensConfig(kappa=float(kappa), N=int(N))
    logref = np.array(jax.vmap(lambda rr: stephens_logpdf_r(rr, cfg=scfg))(rj))

    logw = logq - logref
    logw = logw - np.max(logw)
    logw = np.clip(logw, -80.0, 0.0)

    w = np.exp(logw)
    w = w / np.sum(w)

    ess = float(1.0 / np.sum(w**2))
    w_max = float(np.max(w))

    idx = rng.choice(X_prop.shape[0], size=int(num_samples), replace=True, p=w)
    unique = int(np.unique(idx).size)

    diag = dict(
        ess=ess,
        ess_frac=float(ess / X_prop.shape[0]),
        w_max=w_max,
        unique_idx=unique,
        unique_frac=float(unique / num_samples),
        ref_kind="stephens",
    )
    return X_prop[idx], diag


def main():
    if not MODEL_CFG.exists() or not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Missing trained model/config; run the VRW training demo first."
        )

    cfg = json.loads(MODEL_CFG.read_text())
    score_model = _load_score_model(cfg)

    N = int(cfg["N"])
    MU = float(cfg["MU"])
    KAPPA = float(cfg["KAPPA"])

    # Evidence on u=r/N
    q_alpha = 20.0
    q_beta = 2.0

    # VP schedule
    t1 = float(cfg["t1"])
    beta_min = float(cfg["beta_min"])
    beta_max = float(cfg["beta_max"])
    int_beta_fn = make_vp_int_beta(
        "linear", beta_min=beta_min, beta_max=beta_max, t1=t1
    )

    sampler_cfg = VPSDESamplerConfig(t1=t1, num_steps=600, num_samples=3000, seed=0)

    # Proposal prior endpoints for IS baseline
    draws_prop = 250_000
    X_prop = _sample_vrw_endpoints(N=N, MU=MU, KAPPA=KAPPA, draws=draws_prop, seed=123)

    # IS baseline (Stephens ref)
    X_is, is_diag = _importance_resample_stephens(
        X_prop,
        N=N,
        kappa=KAPPA,
        q_alpha=q_alpha,
        q_beta=q_beta,
        num_samples=sampler_cfg.num_samples,
        seed=999,
    )
    print("IS diagnostics:", is_diag)

    # Sweep knobs
    scales = [0.0, 0.1, 0.2, 0.35, 0.5, 0.8, 1.0, 1.3]
    noise_power = 1.0
    snr_gamma = 1.0

    base_key = jr.PRNGKey(2026)
    keys = jr.split(base_key, len(scales))

    results = []
    for s, key in zip(scales, keys):
        if s == 0.0:
            guidance_fn = None
        else:
            gcfg = VRWRadialRRGuidanceConfig(
                N=N,
                kappa=KAPPA,
                alpha=q_alpha,
                beta=q_beta,
                ref_kind="stephens",
                guidance_scale=float(s),
                noise_aware=True,
                noise_power=float(noise_power),
                snr_gamma=float(snr_gamma),
                eps=1e-6,
                min_alpha=0.05,
                max_guidance_norm=25.0,
            )
            guidance_fn = make_vrw_radial_rr_guidance(gcfg)

        X_g = np.array(
            sample_many_reverse_vp_sde_euler(
                score_model,
                int_beta_fn,
                sample_shape=(2,),
                key=key,
                num_samples=sampler_cfg.num_samples,
                t1=sampler_cfg.t1,
                num_steps=sampler_cfg.num_steps,
                guidance_fn=guidance_fn,
            )
        )

        r_g = np.linalg.norm(X_g, axis=-1)
        r_is = np.linalg.norm(X_is, axis=-1)

        u_g = r_g / N
        frac_gt1 = float(np.mean(u_g > 1.0))

        D_g, p_g = kstest(u_g, "beta", args=(q_alpha, q_beta))

        w2_r = float(w2_empirical_1d(r_g, r_is))
        sw2_x = float(sliced_w2_empirical(X_g, X_is, num_projections=256, seed=0))

        row = dict(
            scale=float(s),
            KS_D=float(D_g),
            KS_p=float(p_g),
            u_gt_1=float(frac_gt1),
            W2_r_vs_IS=w2_r,
            SW2_x_vs_IS=sw2_x,
        )
        results.append(row)
        print(row)

    (OUT_DIR / "results.json").write_text(
        json.dumps(
            dict(
                cfg=dict(
                    N=N,
                    MU=MU,
                    KAPPA=KAPPA,
                    q_alpha=q_alpha,
                    q_beta=q_beta,
                    t1=t1,
                    beta_min=beta_min,
                    beta_max=beta_max,
                    num_steps=sampler_cfg.num_steps,
                    num_samples=sampler_cfg.num_samples,
                    noise_power=noise_power,
                    snr_gamma=snr_gamma,
                ),
                is_diag=is_diag,
                results=results,
            ),
            indent=2,
        )
    )
    print("Saved:", OUT_DIR / "results.json")

    # Plots (separate figures, no custom colors)
    scales_np = np.array([r["scale"] for r in results])
    ks_np = np.array([r["KS_D"] for r in results])
    w2_np = np.array([r["W2_r_vs_IS"] for r in results])
    sw2_np = np.array([r["SW2_x_vs_IS"] for r in results])
    gt1_np = np.array([r["u_gt_1"] for r in results])

    plt.figure()
    plt.plot(scales_np, ks_np, marker="o")
    plt.xlabel("guidance_scale")
    plt.ylabel("KS D (u=r/N vs target Beta)")
    plt.title("VRW guidance sweep: KS vs scale")
    plt.savefig(OUT_DIR / "ks_vs_scale.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(scales_np, w2_np, marker="o")
    plt.xlabel("guidance_scale")
    plt.ylabel("W2(r) vs IS baseline")
    plt.title("VRW guidance sweep: W2(r) vs scale")
    plt.savefig(OUT_DIR / "w2r_vs_scale.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(scales_np, sw2_np, marker="o")
    plt.xlabel("guidance_scale")
    plt.ylabel("SW2(x) vs IS baseline")
    plt.title("VRW guidance sweep: SW2(x) vs scale")
    plt.savefig(OUT_DIR / "sw2x_vs_scale.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(scales_np, gt1_np, marker="o")
    plt.xlabel("guidance_scale")
    plt.ylabel("fraction u>1")
    plt.title("VRW guidance sweep: out-of-support mass vs scale")
    plt.savefig(OUT_DIR / "u_gt1_vs_scale.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved plots to:", OUT_DIR)


if __name__ == "__main__":
    main()
