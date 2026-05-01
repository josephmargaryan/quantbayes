# quantbayes/stochax/energy_based/testing/pk_mnist_ink_ebm_demo.py
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jr
import tensorflow_datasets as tfds
import equinox as eqx

from quantbayes.stochax.energy_based import (
    ConvEBM,
    PCDTrainConfig,
    train_ebm_pcd,
    make_score_fn_from_ebm,
    AnnealedLangevinConfig,
    make_annealed_langevin_sampler,
    InkFractionObservable01,
    PKEvidence,
    PKGuidanceConfig,
    PKGuidance,
    wrap_score_fn_with_pk,
)

# Reuse your existing 1D DSM score net (reference score s_pi(z))
from quantbayes.stochax.diffusion.pk.reference_score import (
    ScoreNet1D,
    ScoreNet1DConfig,
    train_or_load_score_net_dsm,
)


# ----------------------------
# Settings
# ----------------------------
OUTDIR = Path("artifacts/ebm_pk_mnist_ink")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 0
BATCH = 128
STEPS = 20_000  # bump to 50k+ for stronger results

# EBM training sampler
SGLD_STEPS = 60
SGLD_STEP_SIZE = 1e-2

# Sampling for visualization / ref stats
N_REF = 4096
N_SHOW = 64

# Evidence
TARGET_DIGIT = 3
FAT_QUANTILE = 0.80
EVID_SHARPEN = 0.60

# PK knobs
PK_LAMBDA = 6.0
PK_SIGMA_MAX = 0.50
PK_GAMMA = 0.5


def load_mnist_nchw_01():
    tr = tfds.load("mnist", split="train", as_supervised=True, batch_size=-1)
    (x_tr, y_tr) = tfds.as_numpy(tr)
    x_tr = x_tr.astype(np.float32) / 255.0
    x_tr = np.transpose(x_tr, (0, 3, 1, 2))  # NCHW
    return jnp.asarray(x_tr), jnp.asarray(y_tr.astype(np.int32))


def make_grid(x, nrow):
    x = np.asarray(x)
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    x = x[: nrow * nrow]
    x = x.reshape(nrow, nrow, 28, 28)
    rows = [
        np.concatenate([x[i, j] for j in range(nrow)], axis=-1) for i in range(nrow)
    ]
    return np.concatenate(rows, axis=-2)


def main():
    x_train, y_train = load_mnist_nchw_01()
    print("[MNIST]", x_train.shape, y_train.shape)

    # ----------------------------
    # 1) Train EBM (PCD)
    # ----------------------------
    ebm = ConvEBM(key=jr.PRNGKey(SEED), in_channels=1, hidden_channels=64)

    cfg = PCDTrainConfig(
        steps=STEPS,
        batch_size=BATCH,
        lr=2e-4,
        weight_decay=1e-5,
        grad_clip_norm=1.0,
        reinit_prob=0.05,
        init_scale=1.0,
        sgld_steps=SGLD_STEPS,
        sgld_step_size=SGLD_STEP_SIZE,
        clamp_min=0.0,
        clamp_max=1.0,
        l2_energy=0.0,
        seed=SEED,
        print_every=500,
    )
    ebm, _replay = train_ebm_pcd(ebm, x_train, cfg=cfg)
    ebm = eqx.nn.inference_mode(ebm) if hasattr(eqx.nn, "inference_mode") else ebm

    # base score
    base_score = make_score_fn_from_ebm(ebm)

    # annealed sampler for nice samples + sigma gating for PK
    ald_cfg = AnnealedLangevinConfig(
        n_sigmas=30,
        sigma_min=0.01,
        sigma_max=1.0,
        rho=7.0,
        steps_per_sigma=6,
        step_scale=0.08,
        final_denoise=True,
        clamp_min=0.0,
        clamp_max=1.0,
        max_norm=None,
    )
    sampler_base = make_annealed_langevin_sampler(
        base_score, sample_shape=(1, 28, 28), cfg=ald_cfg
    )

    # ----------------------------
    # 2) Estimate π(d) from base EBM samples and train reference score s_pi(z)
    # ----------------------------
    obs = InkFractionObservable01(thr=0.35, temp=0.08)
    x_ref = sampler_base(jr.PRNGKey(SEED + 1001), N_REF)
    d_ref, _ = obs.value_and_grad(x_ref)
    mean_d = float(jnp.mean(d_ref))
    std_d = float(jnp.std(d_ref) + 1e-6)
    z_ref = ((d_ref - mean_d) / std_d).reshape(-1, 1)

    score_path = OUTDIR / "score_pi_z_ink.eqx"
    score_cfg = ScoreNet1DConfig(
        hidden=128,
        lr=2e-3,
        weight_decay=1e-4,
        batch_size=512,
        steps=4000,
        noise_std=0.08,
        seed=SEED,
        print_every=200,
    )
    s_pi = train_or_load_score_net_dsm(z_ref, score_path, cfg=score_cfg)
    print(f"[ref score] saved/loaded: {score_path}")
    print(f"[pi(d)] mean_d={mean_d:.4f}, std_d={std_d:.4f}")

    # ----------------------------
    # 3) Evidence from real class-k digits
    # ----------------------------
    mask = y_train == int(TARGET_DIGIT)
    xk = x_train[mask]
    xk = xk[:5000] if xk.shape[0] > 5000 else xk

    dk, _ = obs.value_and_grad(xk)
    mu_d = float(jnp.quantile(dk, FAT_QUANTILE))
    tau_d_data = float(jnp.std(dk) + 1e-6)
    tau_d = float(max(1e-4, tau_d_data * EVID_SHARPEN))

    mu_z = float((mu_d - mean_d) / std_d)
    tau_z = float(tau_d / std_d)
    print(
        f"[evidence] mu_d={mu_d:.4f}, tau_d={tau_d:.4f} | mu_z={mu_z:.3f}, tau_z={tau_z:.3f}"
    )

    evidence = PKEvidence(mean_d=mean_d, std_d=std_d, mu_z=mu_z, tau_z=tau_z)

    # ----------------------------
    # 4) Build PK guidance + guided sampler
    # ----------------------------
    pk_cfg = PKGuidanceConfig(
        strength=PK_LAMBDA,
        sigma_max=PK_SIGMA_MAX,
        sigma_ref=1.0,
        gamma=PK_GAMMA,
        max_norm=50.0,
        scale_by_sigma2=True,
    )
    guidance = PKGuidance(
        observable=obs,
        evidence=evidence,
        reference_score_z=lambda z: s_pi(z),  # ScoreNet1D handles (B,) or (B,1)
        cfg=pk_cfg,
    )

    score_pk = wrap_score_fn_with_pk(base_score, guidance=guidance, mode="pk")
    sampler_pk = make_annealed_langevin_sampler(
        score_pk, sample_shape=(1, 28, 28), cfg=ald_cfg
    )

    # ----------------------------
    # 5) Sample + visualize
    # ----------------------------
    x_base = sampler_base(jr.PRNGKey(SEED + 2002), N_SHOW)
    x_pk = sampler_pk(jr.PRNGKey(SEED + 3003), N_SHOW)

    nrow = int(round(math.sqrt(N_SHOW)))
    assert nrow * nrow == N_SHOW

    g0 = make_grid(x_base, nrow)
    g1 = make_grid(x_pk, nrow)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(g0, cmap="gray")
    plt.title("EBM samples (base)")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(g1, cmap="gray")
    plt.title(f"EBM + PK(ink) λ={PK_LAMBDA}")
    plt.axis("off")
    plt.tight_layout()
    out = OUTDIR / "grid_compare.png"
    plt.savefig(out, dpi=180)
    plt.show()
    print("[saved]", out)

    # Hist in z-space
    db, _ = obs.value_and_grad(x_base)
    dp, _ = obs.value_and_grad(x_pk)
    zb = (np.asarray(db) - mean_d) / std_d
    zp = (np.asarray(dp) - mean_d) / std_d

    plt.figure(figsize=(10, 4))
    plt.hist(
        np.asarray(z_ref).reshape(-1),
        bins=60,
        density=True,
        alpha=0.25,
        label="π(z) (ref)",
    )
    plt.hist(
        zb, bins=60, density=True, histtype="step", linewidth=2, label="samples (base)"
    )
    plt.hist(
        zp, bins=60, density=True, histtype="step", linewidth=2, label="samples (PK)"
    )
    plt.title("Ink observable distribution in z-space")
    plt.legend()
    plt.tight_layout()
    out2 = OUTDIR / "hist_z.png"
    plt.savefig(out2, dpi=180)
    plt.show()
    print("[saved]", out2)

    print("\nDone. Artifacts in:", OUTDIR)


if __name__ == "__main__":
    main()
