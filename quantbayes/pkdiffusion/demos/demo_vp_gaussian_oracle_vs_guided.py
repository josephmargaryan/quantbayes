# quantbayes/pkdiffusion/demos/demo_vp_gaussian_oracle_vs_guided.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.diffusion.schedules.vp import make_vp_int_beta

from quantbayes.pkdiffusion.gaussian_pk import (
    GaussianND,
    pk_update_linear_gaussian,
    marginal_1d,
)
from quantbayes.pkdiffusion.metrics import (
    kl_gaussian_nd,
    w2_gaussian_nd,
    kl_gaussian_1d,
)
from quantbayes.pkdiffusion.samplers import sample_many_reverse_vp_sde_euler

from quantbayes.pkdiffusion.analytic_scores import AnalyticGaussianVPSScore
from quantbayes.pkdiffusion.guidance_linear_gaussian import (
    LinearGaussianRRGuidanceConfig,
    make_linear_gaussian_rr_guidance,
)

OUT_DIR = Path("reports/pkdiffusion/vp_gaussian_oracle_vs_guided")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def empirical_mean_cov(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.mean(X, axis=0)
    C = np.cov(X.T, bias=True)
    return m, C


def main():
    # -------------------------
    # Ground-truth prior and PK posterior (analytic)
    # -------------------------
    prior = GaussianND(mean=np.zeros(2), cov=np.eye(2))

    a = np.array([1.0, 0.7], dtype=float)
    a = a / np.linalg.norm(a)

    py_mean, py_var = marginal_1d(prior, a)

    q_mean = 1.25
    q_var = 0.20**2

    post = pk_update_linear_gaussian(prior, a, q_mean=q_mean, q_var=q_var)

    # -------------------------
    # VP schedule (must match your sampler conventions)
    # -------------------------
    t1 = 10.0
    beta_min = 0.1
    beta_max = 20.0
    int_beta_fn = make_vp_int_beta(
        "linear", beta_min=beta_min, beta_max=beta_max, t1=t1
    )

    # -------------------------
    # Analytic scores
    # -------------------------
    prior_score = AnalyticGaussianVPSScore(
        mean0=jnp.asarray(prior.mean),
        cov0=jnp.asarray(prior.cov),
        int_beta_fn=int_beta_fn,
    )
    post_score = AnalyticGaussianVPSScore(
        mean0=jnp.asarray(post.mean),
        cov0=jnp.asarray(post.cov),
        int_beta_fn=int_beta_fn,
    )

    # -------------------------
    # Guidance = PK/RR on y=a^T x0
    # -------------------------
    gcfg = LinearGaussianRRGuidanceConfig(
        a=jnp.asarray(a),
        q_mean=q_mean,
        q_var=q_var,
        py_mean=py_mean,
        py_var=py_var,
        guidance_scale=1.0,  # <-- YOU WILL SWEEP THIS LATER
        min_alpha=0.05,
        max_guidance_norm=25.0,
        snr_gamma=1.0,
    )
    guidance_fn = make_linear_gaussian_rr_guidance(gcfg)

    # -------------------------
    # Sample
    # -------------------------
    key = jr.PRNGKey(0)
    k0, k1, k2 = jr.split(key, 3)

    num_samples = 20000
    num_steps = 800

    X_oracle = np.array(
        sample_many_reverse_vp_sde_euler(
            post_score,
            int_beta_fn,
            sample_shape=(2,),
            key=k0,
            num_samples=num_samples,
            t1=t1,
            num_steps=num_steps,
            guidance_fn=None,
        )
    )

    X_guided = np.array(
        sample_many_reverse_vp_sde_euler(
            prior_score,
            int_beta_fn,
            sample_shape=(2,),
            key=k1,
            num_samples=num_samples,
            t1=t1,
            num_steps=num_steps,
            guidance_fn=guidance_fn,
        )
    )

    X_uncond = np.array(
        sample_many_reverse_vp_sde_euler(
            prior_score,
            int_beta_fn,
            sample_shape=(2,),
            key=k2,
            num_samples=num_samples,
            t1=t1,
            num_steps=num_steps,
            guidance_fn=None,
        )
    )

    # -------------------------
    # Metrics vs true posterior
    # -------------------------
    mT, CT = post.mean, post.cov

    def report(name: str, X: np.ndarray):
        m, C = empirical_mean_cov(X)
        kl = kl_gaussian_nd(m, C, mT, CT)
        w2 = w2_gaussian_nd(m, C, mT, CT)

        y = X @ a
        y_m = float(np.mean(y))
        y_v = float(np.var(y))
        kl_y = kl_gaussian_1d(y_m, y_v, q_mean, q_var)

        return dict(name=name, kl=kl, w2=w2, y_mean=y_m, y_var=y_v, kl_y=kl_y)

    R_oracle = report("oracle_post_score", X_oracle)
    R_guided = report("guided_prior+RR", X_guided)
    R_uncond = report("uncond_prior", X_uncond)

    print("=== VP Gaussian Oracle vs Guided ===")
    print("True posterior (PK) mean:", mT)
    print("True posterior (PK) cov:\n", CT)
    for R in [R_oracle, R_guided, R_uncond]:
        print("\n---", R["name"], "---")
        print(f"KL(emp || true)  = {R['kl']:.6f}")
        print(f"W2(emp, true)    = {R['w2']:.6f}")
        print(f"Y mean/var       = ({R['y_mean']:.6f}, {R['y_var']:.6f})")
        print(f"KL(Y || q)       = {R['kl_y']:.6f}")

    # -------------------------
    # Plots
    # -------------------------
    plt.figure(figsize=(7, 7))
    plt.scatter(
        X_uncond[::20, 0], X_uncond[::20, 1], s=6, alpha=0.15, label="uncond prior"
    )
    plt.scatter(
        X_guided[::20, 0], X_guided[::20, 1], s=6, alpha=0.15, label="guided prior+RR"
    )
    plt.scatter(
        X_oracle[::20, 0],
        X_oracle[::20, 1],
        s=6,
        alpha=0.15,
        label="oracle posterior score",
    )
    plt.gca().set_aspect("equal", "box")
    plt.title("VP sampling: oracle vs guided (Gaussian PK posterior)")
    plt.legend()
    p1 = OUT_DIR / "scatter.png"
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()

    # Coarse hist
    yT = np.random.default_rng(0).multivariate_normal(mT, CT, size=200000) @ a
    yU = X_uncond @ a
    yG = X_guided @ a
    yO = X_oracle @ a
    bins = 120

    plt.figure(figsize=(10, 4.5))
    plt.hist(yT, bins=bins, density=True, alpha=0.25, label="true PK posterior (MC)")
    plt.hist(yU, bins=bins, density=True, alpha=0.25, label="uncond prior")
    plt.hist(yG, bins=bins, density=True, alpha=0.25, label="guided")
    plt.hist(yO, bins=bins, density=True, alpha=0.25, label="oracle")
    plt.title("Coarse marginal y=aᵀx: should match q under PK posterior")
    plt.legend()
    p2 = OUT_DIR / "coarse_hist.png"
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nSaved:", p1)
    print("Saved:", p2)


if __name__ == "__main__":
    main()
