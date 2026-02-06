# ============================================================
# PK CANCELLATION DEMO (modular, library-style)
#
# Coarse variable: ink fraction
# Evidence score set equal to reference score => PK cancels:
#   p(d) == pi(d) => p*(x) == pi(x)
#
# This script produces ONE output figure: a 3-panel grid:
#   (none) vs (evidence-only) vs (PK-cancels)
#
# Uses λ=strength=8.0 and gate 1[sigma <= 3.0]
# ============================================================

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import tensorflow_datasets as tfds

from quantbayes.stochax.diffusion.models.mixer_2d import Mixer2d

from quantbayes.stochax.diffusion.pk import (
    InkFractionObservable,
    ScoreNet1DConfig,
    train_or_load_score_net_dsm,
    PKGuidanceConfig,
    PKGuidance,
    wrap_edm_denoise_fn_with_pk,
    make_preconditioned_edm_denoise_fn,
    EDMHeunConfig,
    make_edm_heun_sampler,
    EDMTrainConfig,
    train_or_load_edm_unconditional,
    make_image_grid,
)

# ------------------------
# USER CONTROLS
# ------------------------
SEED = 0
OUTDIR = Path("artifacts/pk_mnist_cancellation")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Try to reuse an existing diffusion ckpt if you already trained one.
CANDIDATE_CKPTS = [
    Path("artifacts/pk_mnist_demo/diffusion_ckpt"),
    Path("artifacts/pk_mnist_ink_only/diffusion_ckpt"),
    OUTDIR / "diffusion_ckpt",
]
DIFF_CKPT_DIR = next(
    (p for p in CANDIDATE_CKPTS if (p / "latest.txt").exists()),
    OUTDIR / "diffusion_ckpt",
)
DIFF_CKPT_DIR.mkdir(parents=True, exist_ok=True)

SCORE_PATH = OUTDIR / "score_pi_z_ink.eqx"

# Coarse observable params
INK_THRESH = 0.35
INK_TEMP = 0.08

# Diffusion training params (kept aligned with your notebook)
train_cfg = EDMTrainConfig(
    lr=2e-4,
    weight_decay=1e-4,
    batch_size=128,
    num_steps=40000,
    ema_decay=0.999,
    grad_clip_norm=1.0,
    print_every=500,
    checkpoint_every=5000,
    keep_last=3,
    sigma_data=0.5,
    sigma_min_train=0.002,
    sigma_max_train=80.0,
    seed=SEED,
)

# Sampling params (EDM-Heun with Karras schedule)
sample_cfg = EDMHeunConfig(
    steps=40,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    sigma_data=train_cfg.sigma_data,
)

# Guidance params (λ + indicator trick)
GUIDE_SIGMA_MAX = 3.0
LAMBDA = 8.0  # <-- as requested
MAX_GUIDE_NORM = 10.0

# Counts
REF_NUM_SAMPLES = 6000
NUM_HIST = 1024
NUM_SHOW = 64


# ------------------------
# DATA (only needed if training)
# ------------------------
def load_mnist_train_nchw_scaled():
    ds = tfds.load("mnist", split="train", as_supervised=True, batch_size=-1)
    x_tr, _ = tfds.as_numpy(ds)

    x_tr = x_tr.astype(np.float32) / 255.0
    if x_tr.ndim == 4 and x_tr.shape[-1] == 1:
        x_tr = np.transpose(x_tr, (0, 3, 1, 2))  # (N,1,28,28)
    x_tr = x_tr * 2.0 - 1.0
    return jnp.asarray(x_tr)


# ------------------------
# Diffusion model wrapper
# ------------------------
class EDMNet(eqx.Module):
    net: eqx.Module

    def __call__(self, log_sigma, x, *, key=None, train=False, **kwargs):
        return self.net(log_sigma, x, key=key)


def build_diffusion_model():
    k = jr.PRNGKey(SEED + 2)
    base = Mixer2d(
        img_size=(1, 28, 28),
        patch_size=4,
        hidden_size=96,
        mix_patch_size=512,
        mix_hidden_size=512,
        num_blocks=4,
        t1=1.0,
        key=k,
    )
    return EDMNet(base)


def main():
    # If we need to train, load data; otherwise dataset is unused.
    dataset = load_mnist_train_nchw_scaled()

    # Train or resume EDM prior
    _, ema_model = train_or_load_edm_unconditional(
        ckpt_dir=DIFF_CKPT_DIR,
        build_model_fn=build_diffusion_model,
        dataset=dataset,
        cfg=train_cfg,
    )

    # Base denoiser (correct EDM preconditioning)
    denoise_base = make_preconditioned_edm_denoise_fn(
        ema_model, sigma_data=train_cfg.sigma_data
    )

    # Observable
    obs = InkFractionObservable(thr=INK_THRESH, temp=INK_TEMP)

    # Sampler (NONE) used to estimate pi(d) and create z_ref for DSM
    sampler_none_tmp = make_edm_heun_sampler(
        denoise_base,
        sample_shape=(1, 28, 28),
        cfg=sample_cfg,
    )

    print(
        f"\nSampling {REF_NUM_SAMPLES} prior images to estimate π(d) and train reference score..."
    )
    x_ref = sampler_none_tmp(jr.PRNGKey(SEED + 777), REF_NUM_SAMPLES)
    d_ref = obs.value(jnp.clip(x_ref, -1.0, 1.0))

    mean_d = float(jnp.mean(d_ref))
    std_d = float(jnp.std(d_ref) + 1e-6)
    print(f"π(d): mean_d={mean_d:.4f}, std_d={std_d:.4f}")

    z_ref = ((d_ref - mean_d) / std_d).reshape(-1, 1)

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
    score_net = train_or_load_score_net_dsm(z_ref, SCORE_PATH, cfg=score_cfg)
    print(f"Reference score net ready: {SCORE_PATH}")

    # Evidence = prior pushforward: s_p(z) := s_pi(z)
    evidence_score_z = lambda z: score_net(z)
    reference_score_z = lambda z: score_net(z)

    guide_cfg = PKGuidanceConfig(
        strength=LAMBDA,
        sigma_max=GUIDE_SIGMA_MAX,
        max_norm=MAX_GUIDE_NORM,
        x0_clip_min=-1.0,
        x0_clip_max=1.0,
        scale_by_sigma2=True,
    )
    guidance = PKGuidance(
        observable=obs,
        mean_d=mean_d,
        std_d=std_d,
        evidence_score_z=evidence_score_z,
        reference_score_z=reference_score_z,
        cfg=guide_cfg,
    )

    # Wrap denoisers
    denoise_evidence = wrap_edm_denoise_fn_with_pk(
        denoise_base,
        sigma_data=train_cfg.sigma_data,
        guidance=guidance,
        mode="evidence",
    )
    denoise_pk = wrap_edm_denoise_fn_with_pk(
        denoise_base, sigma_data=train_cfg.sigma_data, guidance=guidance, mode="pk"
    )

    # Build samplers for each denoiser
    sampler_none = make_edm_heun_sampler(
        denoise_base, sample_shape=(1, 28, 28), cfg=sample_cfg
    )
    sampler_evd = make_edm_heun_sampler(
        denoise_evidence, sample_shape=(1, 28, 28), cfg=sample_cfg
    )
    sampler_pk = make_edm_heun_sampler(
        denoise_pk, sample_shape=(1, 28, 28), cfg=sample_cfg
    )

    base_key = jr.PRNGKey(SEED + 2026)

    print("\nSampling (none)...")
    x_none = sampler_none(base_key, NUM_HIST)

    print("Sampling (evidence-only, evidence=prior)...")
    x_evd = sampler_evd(base_key, NUM_HIST)

    print("Sampling (PK, evidence=prior -> should cancel)...")
    x_pk = sampler_pk(base_key, NUM_HIST)

    max_abs_diff = float(jnp.max(jnp.abs(x_none - x_pk)))
    print(f"\nSanity: max |x_none - x_pk| = {max_abs_diff:.6e}  (should be ~0)")

    # ---- ONE OUTPUT: 3-panel grid figure ----
    nrow = int(round(math.sqrt(NUM_SHOW)))
    assert nrow * nrow == NUM_SHOW, "NUM_SHOW must be a perfect square (e.g. 64)."

    g_none = make_image_grid(np.asarray(x_none[:NUM_SHOW]), nrow=nrow)
    g_evd = make_image_grid(np.asarray(x_evd[:NUM_SHOW]), nrow=nrow)
    g_pk = make_image_grid(np.asarray(x_pk[:NUM_SHOW]), nrow=nrow)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(g_none, cmap="gray")
    plt.title("none (prior)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(g_evd, cmap="gray")
    plt.title("evidence-only (evidence=prior)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(g_pk, cmap="gray")
    plt.title("PK (should equal none)")
    plt.axis("off")

    plt.tight_layout()
    out_png = OUTDIR / "pk_cancellation_grid.png"
    plt.savefig(out_png, dpi=150)
    plt.show()

    print(f"\nSaved: {out_png}")
    print(f"Artifacts dir: {OUTDIR}")
    print(f"Diffusion ckpt: {DIFF_CKPT_DIR}")
    print(f"Score net: {SCORE_PATH}")


if __name__ == "__main__":
    main()
