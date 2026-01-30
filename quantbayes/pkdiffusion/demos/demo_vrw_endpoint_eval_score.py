# quantbayes/pkdiffusion/demos/demo_vrw_endpoint_eval_score.py
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx
import numpyro.distributions as dist

from quantbayes.pkstruct.toy.vrw import vrw_endpoint
from quantbayes.stochax.diffusion.schedules.vp import make_vp_int_beta
from quantbayes.pkdiffusion.models import ScoreMLP
from quantbayes.pkdiffusion.samplers import sample_many_reverse_vp_sde_euler
from quantbayes.pkdiffusion.metrics import w2_empirical_1d, sliced_w2_empirical


OUT_DIR = Path("reports/pkdiffusion/vrw_endpoint_score_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path("reports/pkdiffusion/vrw_endpoint_score")
MODEL_CFG = MODEL_DIR / "config.json"


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


def main():
    if not MODEL_CFG.exists():
        raise FileNotFoundError(
            "Missing model config. Run training first:\n"
            "  python -m quantbayes.pkdiffusion.demos.demo_vrw_endpoint_train_score"
        )

    cfg = json.loads(MODEL_CFG.read_text())
    score_model = _load_score_model(cfg)

    # Ground-truth VRW endpoint prior samples
    N = int(cfg["N"])
    MU = float(cfg["MU"])
    KAPPA = float(cfg["KAPPA"])

    M_TRUE = 80_000
    key_true = jr.PRNGKey(0)
    theta = dist.VonMises(MU, KAPPA).expand([N]).to_event(1).sample(key_true, (M_TRUE,))
    X_true = np.array(jax.vmap(vrw_endpoint)(theta))

    # Diffusion samples (unconditional)
    t1 = float(cfg["t1"])
    beta_min = float(cfg["beta_min"])
    beta_max = float(cfg["beta_max"])
    int_beta_fn = make_vp_int_beta(
        "linear", beta_min=beta_min, beta_max=beta_max, t1=t1
    )

    M_SAMP = 20_000
    key_samp = jr.PRNGKey(1)
    X_samp = np.array(
        sample_many_reverse_vp_sde_euler(
            score_model,
            int_beta_fn,
            sample_shape=(2,),
            key=key_samp,
            num_samples=M_SAMP,
            t1=t1,
            num_steps=600,
            guidance_fn=None,
        )
    )

    # Finite filtering (should be 100%)
    def finite(X):
        m = np.isfinite(X).all(axis=1)
        return X[m], float(np.mean(m))

    X_true, ft = finite(X_true)
    X_samp, fs = finite(X_samp)

    r_true = np.linalg.norm(X_true, axis=-1)
    r_samp = np.linalg.norm(X_samp, axis=-1)

    w2_r = w2_empirical_1d(r_samp, r_true)
    sw2_x = sliced_w2_empirical(X_samp, X_true, num_projections=256, seed=0)

    # Plots
    plt.figure(figsize=(7, 7))
    plt.scatter(
        X_true[::20, 0],
        X_true[::20, 1],
        s=6,
        alpha=0.15,
        label="ground-truth VRW endpoints",
    )
    plt.scatter(
        X_samp[::20, 0], X_samp[::20, 1], s=6, alpha=0.15, label="diffusion samples"
    )
    plt.gca().set_aspect("equal", "box")
    plt.title("VRW endpoint prior: ground-truth vs diffusion")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    p1 = OUT_DIR / "scatter_true_vs_diffusion.png"
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 4.5))
    plt.hist(r_true, bins=80, density=True, alpha=0.5, label="ground-truth r")
    plt.hist(r_samp, bins=80, density=True, alpha=0.5, label="diffusion r")
    plt.title("Radial distribution r = ||endpoint||")
    plt.xlabel("r")
    plt.ylabel("density")
    plt.legend()
    p2 = OUT_DIR / "radial_true_vs_diffusion.png"
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()

    summary = (
        "=== VRW endpoint diffusion eval ===\n"
        f"Finite fraction: true={ft:.4f}, diffusion={fs:.4f}\n"
        f"W2(r)  (diffusion vs true) = {w2_r:.6f}\n"
        f"SW2(x) (diffusion vs true) = {sw2_x:.6f}\n"
        f"Saved: {p1}\n"
        f"Saved: {p2}\n"
    )
    (OUT_DIR / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
