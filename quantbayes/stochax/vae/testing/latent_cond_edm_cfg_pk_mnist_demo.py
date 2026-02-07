# quantbayes/stochax/vae/testing/latent_cond_edm_cfg_pk_mnist_demo.py
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

from quantbayes.stochax.vae.components import ConvVAE
from quantbayes.stochax.vae.train_vae import TrainConfig, train_vae

from quantbayes.stochax.vae.latent_diffusion import (
    EDMNet,
    LatentEDMCondMLP,
    LatentEDMCondConfig,
    LatentEDMCondTrainConfig,
    train_or_load_latent_edm_prior_conditional,
    LatentEDMCondSampleConfig,
    sample_latent_edm_conditional_cfg,
    collect_latents_with_labels_from_vae,
    DecodedInkPKConfig,
    InkEvidence,
    compute_ink_evidence_from_real_data,
    train_reference_score_for_decoded_ink,
    DecodedInkPKGuidance,
)
from quantbayes.stochax.vae.latent_diffusion.coarse import ink_fraction_01
from quantbayes.stochax.diffusion.pk.calibration import w1_to_gaussian

from quantbayes.stochax.diffusion.pk.reference_score import ScoreNet1DConfig


OUTDIR = Path("artifacts/vae_latent_cond_edm_cfg_pk_mnist")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 0
NUM_CLASSES = 10
LATENT_DIM = 16

# ---- VAE ----
VAE_PATH = OUTDIR / "vae.eqx"
VAE_EPOCHS = 25
VAE_BATCH = 128

# ---- Conditional latent EDM prior ----
PRIOR_CKPT = OUTDIR / "latent_edm_cond_ckpt"
PRIOR_TRAIN = LatentEDMCondTrainConfig(
    batch_size=512,
    num_steps=50_000,
    lr=2e-4,
    weight_decay=1e-5,
    grad_clip_norm=1.0,
    ema_decay=0.999,
    print_every=500,
    checkpoint_every=5000,
    keep_last=3,
    seed=SEED,
    sigma_data=0.5,
    sigma_min_train=0.002,
    sigma_max_train=80.0,
    p_uncond=0.10,  # CFG dropout
)

PRIOR_SAMPLE = LatentEDMCondSampleConfig(
    steps=40,
    sigma_min=0.002,
    sigma_max=80.0,
    sigma_data=0.5,
    rho=7.0,
    cfg_scale=3.0,
)

# ---- PK decoded-ink steering ----
TARGET_DIGIT = 3
FAT_QUANTILE = 0.80
EVID_SHARPEN = 0.60

PK_CFG = DecodedInkPKConfig(
    ink_thr=0.35,
    ink_temp=0.08,
    guide_strength=8.0,
    guide_sigma_max=1.5,
    max_guide_norm=50.0,
    sigma_data=0.5,
    w_gamma=0.5,
)

REF_SCORE_PATH = OUTDIR / f"score_pi_z_decoded_ink_label{TARGET_DIGIT}.eqx"
REF_SCORE_CFG = ScoreNet1DConfig(
    hidden=128,
    lr=2e-3,
    weight_decay=1e-4,
    batch_size=512,
    steps=4000,
    noise_std=0.08,
    seed=SEED,
    print_every=200,
)

# visuals
PER_CLASS = 16  # 4x4 per class
GRID_SHOW = 64  # 8x8 for comparisons
REF_SAMPLES = 4096  # reference π(d|y) estimation


def load_mnist_nchw_01():
    tr = tfds.load("mnist", split="train", as_supervised=True, batch_size=-1)
    (x_tr, y_tr) = tfds.as_numpy(tr)
    x_tr = x_tr.astype(np.float32) / 255.0
    x_tr = np.transpose(x_tr, (0, 3, 1, 2))  # NCHW
    return jnp.asarray(x_tr), jnp.asarray(y_tr.astype(np.int32))


def save_eqx(path: Path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load_eqx(path: Path, template):
    with open(path, "rb") as f:
        return eqx.tree_deserialise_leaves(f, template)


def make_grid(x, nrow, ncol):
    x = np.asarray(x)
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    x = x[: nrow * ncol]
    x = x.reshape(nrow, ncol, 28, 28)
    rows = [
        np.concatenate([x[i, j] for j in range(ncol)], axis=-1) for i in range(nrow)
    ]
    return np.concatenate(rows, axis=-2)


def main():
    x_train, y_train = load_mnist_nchw_01()
    print("[MNIST]", x_train.shape, y_train.shape)

    # ----------------------------
    # 1) Train/load VAE
    # ----------------------------
    vae_template = ConvVAE(
        image_size=28,
        channels=1,
        hidden_channels=64,
        latent_dim=LATENT_DIM,
        key=jr.PRNGKey(SEED),
    )
    if VAE_PATH.exists():
        vae = load_eqx(VAE_PATH, vae_template)
        print("[VAE] loaded:", VAE_PATH)
    else:
        vae_cfg = TrainConfig(
            epochs=VAE_EPOCHS,
            batch_size=VAE_BATCH,
            learning_rate=1e-3,
            weight_decay=1e-4,
            grad_clip_norm=1.0,
            beta_schedule="linear",
            beta_warmup_steps=10_000,
            free_bits=0.0,
            likelihood="bernoulli",
            gaussian_learn_logvar=False,
            seed=SEED,
            verbose=True,
        )
        vae = train_vae(vae_template, x_train, vae_cfg)
        save_eqx(VAE_PATH, vae)
        print("[VAE] saved:", VAE_PATH)

    # ----------------------------
    # 2) Collect (z,y) and train/load conditional latent EDM with CFG dropout
    # ----------------------------
    z_data, y_data = collect_latents_with_labels_from_vae(
        vae,
        x_train,
        y_train,
        key=jr.PRNGKey(SEED + 111),
        batch_size=512,
        num_samples=None,
        use_mu=False,
    )
    print("[latents]", z_data.shape, y_data.shape)

    cond_cfg = LatentEDMCondConfig(
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        hidden=256,
        depth=3,
        time_emb_dim=64,
        label_emb_dim=64,
    )
    model_template = EDMNet(LatentEDMCondMLP(cond_cfg, key=jr.PRNGKey(SEED + 222)))
    null_label = cond_cfg.num_classes  # index of null token

    _, ema_model = train_or_load_latent_edm_prior_conditional(
        ckpt_dir=PRIOR_CKPT,
        model_template=model_template,
        z_dataset=z_data,
        y_dataset=y_data,
        null_label=null_label,
        cfg=PRIOR_TRAIN,
    )

    # ----------------------------
    # 3) Class-conditional CFG sampling: show digits 0..9
    # ----------------------------
    imgs_by_label = []
    for lbl in range(NUM_CLASSES):
        z = sample_latent_edm_conditional_cfg(
            ema_model=ema_model,
            key=jr.PRNGKey(SEED + 10_000 + lbl),
            label=lbl,
            num_samples=PER_CLASS,
            latent_dim=LATENT_DIM,
            cfg=PRIOR_SAMPLE,
            null_label=null_label,
            pk_guidance=None,
        )
        logits = vae.decoder(z, rng=None, train=False)
        x01 = jax.nn.sigmoid(logits)
        imgs_by_label.append(np.asarray(x01))

    # build a (10 rows) x (PER_CLASS cols) mosaic
    row_imgs = []
    for lbl in range(NUM_CLASSES):
        row_imgs.append(make_grid(imgs_by_label[lbl], nrow=1, ncol=PER_CLASS))
    mosaic = np.concatenate(row_imgs, axis=0)

    plt.figure(figsize=(18, 8))
    plt.imshow(mosaic, cmap="gray")
    plt.title(
        f"Class-conditional latent EDM prior + CFG (scale={PRIOR_SAMPLE.cfg_scale})"
    )
    plt.axis("off")
    out = OUTDIR / "cfg_class_mosaic.png"
    plt.savefig(out, dpi=180)
    plt.show()
    print("[saved]", out)

    # ----------------------------
    # 4) PK within one class: train π(d|y) ref score on decoded ink, then steer
    # ----------------------------
    # Reference: sample latents from the SAME base sampler (same label & cfg_scale) without PK
    z_ref = sample_latent_edm_conditional_cfg(
        ema_model=ema_model,
        key=jr.PRNGKey(SEED + 5555),
        label=TARGET_DIGIT,
        num_samples=REF_SAMPLES,
        latent_dim=LATENT_DIM,
        cfg=PRIOR_SAMPLE,
        null_label=null_label,
        pk_guidance=None,
    )

    def decode_logits_fn(z):
        return vae.decoder(z, rng=None, train=False)

    ref_score_net, mean_d, std_d = train_reference_score_for_decoded_ink(
        vae=vae,
        decode_logits_fn=decode_logits_fn,
        z_samples=z_ref,
        score_path=REF_SCORE_PATH,
        cfg=REF_SCORE_CFG,
        ink_thr=PK_CFG.ink_thr,
        ink_temp=PK_CFG.ink_temp,
    )
    print(f"[ref π(d|y)] mean_d={mean_d:.4f}, std_d={std_d:.4f} (label={TARGET_DIGIT})")

    # Evidence from real data: “fat” class-k digits
    mu_d, tau_d = compute_ink_evidence_from_real_data(
        x_train,
        digit_labels=y_train,
        target_digit=TARGET_DIGIT,
        fat_quantile=FAT_QUANTILE,
        sharpen=EVID_SHARPEN,
        ink_thr=PK_CFG.ink_thr,
        ink_temp=PK_CFG.ink_temp,
    )
    mu_z = float((mu_d - mean_d) / std_d)
    tau_z = float(tau_d / std_d)
    evidence = InkEvidence(mean_d=mean_d, std_d=std_d, mu_z=mu_z, tau_z=tau_z)
    print(
        f"[evidence] mu_d={mu_d:.4f}, tau_d={tau_d:.4f} | mu_z={mu_z:.3f}, tau_z={tau_z:.3f}"
    )

    # Guidance object
    guidance = DecodedInkPKGuidance(
        vae_decoder=vae.decoder,
        evidence=evidence,
        ref_score_net=ref_score_net,
        cfg=PK_CFG,
        mode="pk",
    )

    # Baseline samples for label
    z_base = sample_latent_edm_conditional_cfg(
        ema_model=ema_model,
        key=jr.PRNGKey(SEED + 7777),
        label=TARGET_DIGIT,
        num_samples=GRID_SHOW,
        latent_dim=LATENT_DIM,
        cfg=PRIOR_SAMPLE,
        null_label=null_label,
        pk_guidance=None,
    )
    x_base = jax.nn.sigmoid(vae.decoder(z_base, rng=None, train=False))

    # PK-steered samples for same label
    z_pk = sample_latent_edm_conditional_cfg(
        ema_model=ema_model,
        key=jr.PRNGKey(SEED + 8888),
        label=TARGET_DIGIT,
        num_samples=GRID_SHOW,
        latent_dim=LATENT_DIM,
        cfg=PRIOR_SAMPLE,
        null_label=null_label,
        pk_guidance=guidance,
    )
    x_pk = jax.nn.sigmoid(vae.decoder(z_pk, rng=None, train=False))

    nrow = int(round(math.sqrt(GRID_SHOW)))
    assert nrow * nrow == GRID_SHOW
    g0 = make_grid(np.asarray(x_base), nrow=nrow, ncol=nrow)
    g1 = make_grid(np.asarray(x_pk), nrow=nrow, ncol=nrow)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(g0, cmap="gray")
    plt.title(f"CFG only (label={TARGET_DIGIT})")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(g1, cmap="gray")
    plt.title(f"CFG + PK(ink) (λ={PK_CFG.guide_strength})")
    plt.axis("off")
    plt.tight_layout()
    out2 = OUTDIR / "pk_within_class_compare.png"
    plt.savefig(out2, dpi=180)
    plt.show()
    print("[saved]", out2)

    print("\nDone. Artifacts:", OUTDIR)

    def summarize_ink(x01, mean_d, std_d, mu_z, tau_z):
        d = np.asarray(
            ink_fraction_01(jnp.asarray(x01), thr=PK_CFG.ink_thr, temp=PK_CFG.ink_temp)
        )
        z = (d - mean_d) / std_d
        return (
            float(z.mean()),
            float(z.std() + 1e-12),
            float(w1_to_gaussian(z, mu=mu_z, tau=tau_z, seed=SEED + 777)),
        )

    m0, s0, w10 = summarize_ink(x_base, mean_d, std_d, mu_z, tau_z)
    m1, s1, w11 = summarize_ink(x_pk, mean_d, std_d, mu_z, tau_z)

    print("[ink z-space]")
    print(f" target mu_z={mu_z:.3f}, tau_z={tau_z:.3f}")
    print(f"  base  mean={m0:.3f}, std={s0:.3f}, W1={w10:.4f}")
    print(f"  PK    mean={m1:.3f}, std={s1:.3f}, W1={w11:.4f}")


if __name__ == "__main__":
    main()
