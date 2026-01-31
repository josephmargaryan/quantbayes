# quantbayes/pkdiffusion/demos/demo_vp_gaussian_guidance_sweep.py
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

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

OUT_DIR = Path("reports/pkdiffusion/vp_gaussian_guidance_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def empirical_mean_cov(X: np.ndarray):
    m = np.mean(X, axis=0)
    C = np.cov(X.T, bias=True)
    return m, C


def main():
    # Prior + PK posterior (analytic)
    prior = GaussianND(mean=np.zeros(2), cov=np.eye(2))
    a = np.array([1.0, 0.7], dtype=float)
    a /= np.linalg.norm(a)
    py_mean, py_var = marginal_1d(prior, a)

    q_mean = 1.25
    q_var = 0.20**2
    post = pk_update_linear_gaussian(prior, a, q_mean=q_mean, q_var=q_var)

    # VP schedule
    t1 = 10.0
    beta_min, beta_max = 0.1, 20.0
    int_beta_fn = make_vp_int_beta(
        "linear", beta_min=beta_min, beta_max=beta_max, t1=t1
    )

    # Analytic prior score
    prior_score = AnalyticGaussianVPSScore(
        mean0=jnp.asarray(prior.mean),
        cov0=jnp.asarray(prior.cov),
        int_beta_fn=int_beta_fn,
    )

    # Sweep
    scales = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
    num_samples = 20000
    num_steps = 800

    results = []
    for s in scales:
        gcfg = LinearGaussianRRGuidanceConfig(
            a=jnp.asarray(a),
            q_mean=q_mean,
            q_var=q_var,
            py_mean=py_mean,
            py_var=py_var,
            guidance_scale=float(s),
            noise_aware=True,  # <-- assumes you added this
            min_alpha=0.05,
            max_guidance_norm=25.0,
            snr_gamma=1.0,
        )
        guidance_fn = make_linear_gaussian_rr_guidance(gcfg)

        key = jr.PRNGKey(0)
        X = np.array(
            sample_many_reverse_vp_sde_euler(
                prior_score,
                int_beta_fn,
                sample_shape=(2,),
                key=key,
                num_samples=num_samples,
                t1=t1,
                num_steps=num_steps,
                guidance_fn=guidance_fn if s != 0.0 else None,
            )
        )

        m, C = empirical_mean_cov(X)

        kl = float(kl_gaussian_nd(m, C, post.mean, post.cov))
        w2 = float(w2_gaussian_nd(m, C, post.mean, post.cov))

        y = X @ a
        y_m = float(np.mean(y))
        y_v = float(np.var(y))
        kl_y = float(kl_gaussian_1d(y_m, y_v, q_mean, q_var))

        results.append(
            dict(
                scale=float(s),
                KL=float(kl),
                W2=float(w2),
                y_mean=float(y_m),
                y_var=float(y_v),
                KL_y=float(kl_y),
            )
        )
        print(results[-1])

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved:", OUT_DIR / "results.json")


if __name__ == "__main__":
    main()
