# quantbayes/stochax/vae/testing/latent_diffusion_pk_mnist_demo.py
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
    LatentEDMMLP,
    LatentEDMConfig,
    EDMNet,
    LatentEDMTrainConfig,
    train_or_load_latent_edm_prior,
    LatentEDMSampleConfig,
    make_latent_denoise_fn,
    sample_latent_edm,
    collect_latents_from_vae,
    DecodedInkPKConfig,
    InkEvidence,
    train_reference_score_for_decoded_ink,
    compute_ink_evidence_from_real_data,
    DecodedInkPKGuidance,
    wrap_denoise_fn_with_x0_guidance,
)
from quantbayes.stochax.diffusion.pk.reference_score import ScoreNet1DConfig


OUTDIR = Path("artifacts/vae_latent_edm_pk_mnist")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 0

# VAE
LATENT_DIM = 16
VAE_EPOCHS = 25
VAE_BATCH = 128

# Latent EDM prior
PRIOR_CKPT = OUTDIR / "latent_edm_ckpt"
PRIOR_TRAIN = LatentEDMTrainConfig(
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
)

PRIOR_SAMPLE = LatentEDMSampleConfig(
    steps=40,
    sigma_min=0.002,
    sigma_max=80.0,
    sigma_data=0.5,
    rho=7.0,
    sampler="dpmpp_3m",
)

# PK steering target
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

# Ref score net for π(d)
REF_SCORE_PATH = OUTDIR / "score_pi_z_decoded_ink.eqx"
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

# Visuals
NUM_SAMPLES = 1024
NUM_SHOW = 64


def load_mnist_nchw_01():
    tr = tfds.load("mnist", split="train", as_supervised=True, batch_size=-1)
    te = tfds.load("mnist", split="test", as_supervised=True, batch_size=-1)
    (x_tr, y_tr) = tfds.as_numpy(tr)
    (x_te, y_te) = tfds.as_numpy(te)

    x_tr = x_tr.astype(np.float32) / 255.0
    x_te = x_te.astype(np.float32) / 255.0
    # NHWC->NCHW
    x_tr = np.transpose(x_tr, (0, 3, 1, 2))
    x_te = np.transpose(x_te, (0, 3, 1, 2))
    return (
        jnp.asarray(x_tr),
        jnp.asarray(y_tr.astype(np.int32)),
        jnp.asarray(x_te),
        jnp.asarray(y_te.astype(np.int32)),
    )


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
    x_train, y_train, _, _ = load_mnist_nchw_01()
    print("[MNIST]", x_train.shape, y_train.shape)

    # ----------------------------
    # 1) Train VAE
    # ----------------------------
    vae = ConvVAE(
        image_size=28,
        channels=1,
        hidden_channels=64,
        latent_dim=LATENT_DIM,
        key=jr.PRNGKey(SEED),
    )
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
    vae = train_vae(vae, x_train, vae_cfg)
    vae_eval = eqx.nn.inference_mode(vae) if hasattr(eqx.nn, "inference_mode") else vae

    # ----------------------------
    # 2) Collect aggregated posterior latents and train latent EDM prior
    # ----------------------------
    z_data = collect_latents_from_vae(
        vae,
        x_train,
        key=jr.PRNGKey(SEED + 111),
        batch_size=512,
        num_samples=None,
        use_mu=False,
    )
    print("[latents]", z_data.shape)

    edm_cfg = LatentEDMConfig(
        latent_dim=LATENT_DIM, hidden=256, depth=3, time_emb_dim=64
    )
    model_template = EDMNet(LatentEDMMLP(edm_cfg, key=jr.PRNGKey(SEED + 222)))

    _, ema_model = train_or_load_latent_edm_prior(
        ckpt_dir=PRIOR_CKPT,
        model_template=model_template,
        z_dataset=z_data,
        cfg=PRIOR_TRAIN,
    )

    # Base latent denoiser
    denoise_base = make_latent_denoise_fn(ema_model, sigma_data=PRIOR_SAMPLE.sigma_data)

    # ----------------------------
    # 3) Baseline samples: z~N(0,1) vs z~latent-EDM
    # ----------------------------
    key = jr.PRNGKey(SEED + 333)
    k0, k1, kref = jr.split(key, 3)

    z_std = jr.normal(k0, (NUM_SAMPLES, LATENT_DIM))
    logits_std = vae.decoder(z_std, rng=None, train=False)
    x_std = jax.nn.sigmoid(logits_std)

    z_prior = sample_latent_edm(
        denoise_base,
        key=k1,
        num_samples=NUM_SAMPLES,
        latent_dim=LATENT_DIM,
        cfg=PRIOR_SAMPLE,
    )
    logits_prior = vae.decoder(z_prior, rng=None, train=False)
    x_prior = jax.nn.sigmoid(logits_prior)

    # ----------------------------
    # 4) Train reference score for decoded ink under latent prior π(d)
    # ----------------------------
    # sample some latents from the latent prior to estimate π(d)
    z_ref = sample_latent_edm(
        denoise_base,
        key=kref,
        num_samples=4096,
        latent_dim=LATENT_DIM,
        cfg=PRIOR_SAMPLE,
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
    print(f"[ref] π(d): mean_d={mean_d:.4f}, std_d={std_d:.4f}")

    # ----------------------------
    # 5) Evidence p(d) from real MNIST class-k (fat digits)
    # ----------------------------
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

    # ----------------------------
    # 6) PK guidance during latent diffusion sampling
    # ----------------------------
    guidance = DecodedInkPKGuidance(
        vae_decoder=vae.decoder,
        evidence=evidence,
        ref_score_net=ref_score_net,
        cfg=PK_CFG,
        mode="pk",
    )
    denoise_pk = wrap_denoise_fn_with_x0_guidance(
        denoise_base,
        sigma_data=PRIOR_SAMPLE.sigma_data,
        guidance=guidance,
    )

    z_pk = sample_latent_edm(
        denoise_pk,
        key=jr.PRNGKey(SEED + 444),
        num_samples=NUM_SAMPLES,
        latent_dim=LATENT_DIM,
        cfg=PRIOR_SAMPLE,
    )
    logits_pk = vae.decoder(z_pk, rng=None, train=False)
    x_pk = jax.nn.sigmoid(logits_pk)

    # ----------------------------
    # 7) Visual comparison
    # ----------------------------
    nrow = int(round(math.sqrt(NUM_SHOW)))
    assert nrow * nrow == NUM_SHOW

    g_std = make_grid(np.asarray(x_std[:NUM_SHOW]), nrow)
    g_prior = make_grid(np.asarray(x_prior[:NUM_SHOW]), nrow)
    g_pk = make_grid(np.asarray(x_pk[:NUM_SHOW]), nrow)

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(g_std, cmap="gray")
    plt.title("VAE: z ~ N(0,1)")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(g_prior, cmap="gray")
    plt.title("VAE: z ~ Latent EDM Prior")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(g_pk, cmap="gray")
    plt.title("Latent EDM + PK(ink)")
    plt.axis("off")
    plt.tight_layout()
    out = OUTDIR / "compare_grid.png"
    plt.savefig(out, dpi=180)
    plt.show()
    print("[saved]", out)

    print("\nDone. Artifacts:", OUTDIR)


if __name__ == "__main__":
    main()
