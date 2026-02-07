from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import tensorflow_datasets as tfds

from quantbayes.stochax.vae.components import ConvVAE, MLP_VAE, ViT_VAE
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
    train_reference_score_for_decoded_ink,
    compute_ink_evidence_from_real_data,
    DecodedInkPKGuidance,
    wrap_denoise_fn_with_x0_guidance,
)
from quantbayes.stochax.diffusion.pk.reference_score import ScoreNet1DConfig
from quantbayes.stochax.vae.latent_diffusion.pk_guidance import InkEvidence


# ============================================================
# USER CONTROLS / CONFIG
# ============================================================

SEED = 0

ARTIFACT_ROOT = Path("artifacts/vae_latent_edm_pk_mnist")
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

# ---- model switchboards
VAE_NAME = "conv"  # "conv" | "mlp" | "vit"
PRIOR_NAME = "latent_mlp"  # currently only "latent_mlp" here (easy to extend)

LATENT_DIM = 16

# ---- optional external paths (if set and exists, overrides internal defaults)
VAE_LOAD_PATH: Optional[Path] = None  # e.g. Path(".../vae.eqx")
LATENT_PRIOR_CKPT_DIR: Optional[Path] = None  # e.g. Path(".../latent_edm_ckpt")

# ---- training flags
TRAIN_IF_MISSING_VAE = True
TRAIN_IF_MISSING_PRIOR = True

# ---- VAE training
VAE_EPOCHS = 25
VAE_BATCH = 128

# ---- latent EDM prior training
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

# ---- latent EDM prior sampling
PRIOR_SAMPLE = LatentEDMSampleConfig(
    steps=40,
    sigma_min=0.002,
    sigma_max=80.0,
    sigma_data=0.5,
    rho=7.0,
    sampler="dpmpp_3m",
)

# ---- PK steering target
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

# ---- ref score net π(d) for decoded ink under latent prior
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

# ---- visuals
NUM_SAMPLES = 1024
NUM_SHOW = 64


# ============================================================
# Helpers
# ============================================================


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, obj: dict) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))


def _save_eqx(path: Path, model) -> None:
    _ensure_dir(path.parent)
    with open(path, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def _load_eqx(path: Path, template):
    with open(path, "rb") as f:
        return eqx.tree_deserialise_leaves(f, template)


def _pick_first_existing(candidates: list[Path], default: Path) -> Path:
    for p in candidates:
        if p is not None and p.exists():
            return p
    return default


def inference_copy(model):
    maybe = eqx.nn.inference_mode(model)
    enter = getattr(maybe, "__enter__", None)
    exit_ = getattr(maybe, "__exit__", None)
    if callable(enter) and callable(exit_):
        try:
            m = enter()
            return m
        finally:
            exit_(None, None, None)
    return maybe


def load_mnist_nchw_01():
    tr = tfds.load("mnist", split="train", as_supervised=True, batch_size=-1)
    te = tfds.load("mnist", split="test", as_supervised=True, batch_size=-1)
    (x_tr, y_tr) = tfds.as_numpy(tr)
    (x_te, y_te) = tfds.as_numpy(te)

    x_tr = x_tr.astype(np.float32) / 255.0
    x_te = x_te.astype(np.float32) / 255.0

    # NHWC -> NCHW
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


# ============================================================
# Model builders (easy swap)
# ============================================================


def build_vae(key: jr.PRNGKey):
    name = VAE_NAME.lower().replace("-", "").replace("_", "").strip()

    if name == "conv":
        return ConvVAE(
            image_size=28,
            channels=1,
            hidden_channels=64,
            latent_dim=LATENT_DIM,
            key=key,
        )

    if name == "mlp":
        # MNIST flattened: 28*28=784
        return MLP_VAE(
            input_dim=784,
            hidden_dim=512,
            latent_dim=LATENT_DIM,
            output_dim=784,
            key=key,
        )

    if name == "vit":
        return ViT_VAE(
            image_size=28,
            channels=1,
            patch_size=4,
            embedding_dim=128,
            num_layers=4,
            num_heads=4,
            latent_dim=LATENT_DIM,
            dropout_rate=0.1,
            key=key,
        )

    raise ValueError(f"Unknown VAE_NAME={VAE_NAME!r}")


def build_latent_prior_model(key: jr.PRNGKey):
    name = PRIOR_NAME.lower().replace("-", "").replace("_", "").strip()
    if name == "latentmlp" or name == "latent_mlp":
        edm_cfg = LatentEDMConfig(
            latent_dim=LATENT_DIM, hidden=256, depth=3, time_emb_dim=64
        )
        return EDMNet(LatentEDMMLP(edm_cfg, key=key))
    raise ValueError(f"Unknown PRIOR_NAME={PRIOR_NAME!r}")


# ============================================================
# Train-or-load routines
# ============================================================


@dataclass(frozen=True)
class Paths:
    run_dir: Path
    vae_path: Path
    vae_meta: Path
    z_cache: Path
    prior_ckpt: Path
    ref_score_path: Path


def make_paths() -> Paths:
    run_dir = ARTIFACT_ROOT / f"vae_{VAE_NAME}_z{LATENT_DIM}__prior_{PRIOR_NAME}"
    _ensure_dir(run_dir)

    return Paths(
        run_dir=run_dir,
        vae_path=run_dir / "vae.eqx",
        vae_meta=run_dir / "vae_meta.json",
        z_cache=run_dir / "z_agg.npy",
        prior_ckpt=(
            LATENT_PRIOR_CKPT_DIR
            if LATENT_PRIOR_CKPT_DIR is not None
            else (run_dir / "latent_edm_ckpt")
        ),
        ref_score_path=run_dir / "score_pi_z_decoded_ink.eqx",
    )


def get_or_train_vae(x_train: jnp.ndarray, paths: Paths):
    # external override
    if VAE_LOAD_PATH is not None and Path(VAE_LOAD_PATH).exists():
        template = build_vae(jr.PRNGKey(SEED))
        vae = _load_eqx(Path(VAE_LOAD_PATH), template)
        return vae

    # local cache
    if paths.vae_path.exists():
        template = build_vae(jr.PRNGKey(SEED))
        vae = _load_eqx(paths.vae_path, template)
        return vae

    if not TRAIN_IF_MISSING_VAE:
        raise FileNotFoundError(f"Missing VAE checkpoint at {paths.vae_path}")

    # train from scratch
    vae = build_vae(jr.PRNGKey(SEED))

    # For MLP_VAE, flatten inputs
    name = VAE_NAME.lower().replace("-", "").replace("_", "").strip()
    if name == "mlp":
        x_in = x_train.reshape((x_train.shape[0], -1))
    else:
        x_in = x_train

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
    vae = train_vae(vae, x_in, vae_cfg)

    _save_eqx(paths.vae_path, vae)
    _save_json(
        paths.vae_meta,
        dict(
            vae_name=VAE_NAME,
            latent_dim=LATENT_DIM,
            epochs=VAE_EPOCHS,
            batch=VAE_BATCH,
            seed=SEED,
        ),
    )
    return vae


def get_or_compute_z_agg(vae, x_train: jnp.ndarray, paths: Paths) -> jnp.ndarray:
    if paths.z_cache.exists():
        z_np = np.load(paths.z_cache)
        return jnp.asarray(z_np)

    vae_eval = inference_copy(vae)

    # For MLP_VAE, flatten
    name = VAE_NAME.lower().replace("-", "").replace("_", "").strip()
    if name == "mlp":
        x_in = x_train.reshape((x_train.shape[0], -1))
    else:
        x_in = x_train

    z = collect_latents_from_vae(
        vae_eval,
        x_in,
        key=jr.PRNGKey(SEED + 111),
        batch_size=512,
        num_samples=None,
        use_mu=False,
    )

    np.save(paths.z_cache, np.asarray(z))
    return z


def get_or_train_latent_prior(z_data: jnp.ndarray, paths: Paths):
    _ensure_dir(paths.prior_ckpt)

    if not TRAIN_IF_MISSING_PRIOR and not (paths.prior_ckpt / "latest.txt").exists():
        raise FileNotFoundError(f"Missing latent prior ckpt at {paths.prior_ckpt}")

    model_template = build_latent_prior_model(jr.PRNGKey(SEED + 222))
    _, ema_model = train_or_load_latent_edm_prior(
        ckpt_dir=paths.prior_ckpt,
        model_template=model_template,
        z_dataset=z_data,
        cfg=PRIOR_TRAIN,
    )
    return ema_model


# ============================================================
# Main demo
# ============================================================


def main():
    paths = make_paths()
    print("[run_dir]", paths.run_dir)

    x_train, y_train, _, _ = load_mnist_nchw_01()
    print("[MNIST]", x_train.shape, y_train.shape)

    # 1) VAE: train-or-load
    vae = get_or_train_vae(x_train, paths)
    vae_eval = inference_copy(vae)

    # 2) Aggregate posterior latents z~q(z|x): cacheable
    z_data = get_or_compute_z_agg(vae_eval, x_train, paths)
    print("[z_agg]", z_data.shape)

    # 3) Latent EDM prior: train-or-load
    ema_prior = get_or_train_latent_prior(z_data, paths)
    denoise_base = make_latent_denoise_fn(ema_prior, sigma_data=PRIOR_SAMPLE.sigma_data)

    # 4) Baseline samples: z~N(0,1) vs z~latent-EDM
    key = jr.PRNGKey(SEED + 333)
    k0, k1, kref = jr.split(key, 3)

    z_std = jr.normal(k0, (NUM_SAMPLES, LATENT_DIM))
    logits_std = vae_eval.decoder(z_std, rng=None, train=False)
    x_std = jax.nn.sigmoid(logits_std)

    z_prior = sample_latent_edm(
        denoise_base,
        key=k1,
        num_samples=NUM_SAMPLES,
        latent_dim=LATENT_DIM,
        cfg=PRIOR_SAMPLE,
    )
    logits_prior = vae_eval.decoder(z_prior, rng=None, train=False)
    x_prior = jax.nn.sigmoid(logits_prior)

    # 5) Reference score π(d) for decoded ink under latent prior
    z_ref = sample_latent_edm(
        denoise_base,
        key=kref,
        num_samples=4096,
        latent_dim=LATENT_DIM,
        cfg=PRIOR_SAMPLE,
    )

    def decode_logits_fn(z):
        return vae_eval.decoder(z, rng=None, train=False)

    ref_score_net, mean_d, std_d = train_reference_score_for_decoded_ink(
        vae=vae_eval,
        decode_logits_fn=decode_logits_fn,
        z_samples=z_ref,
        score_path=paths.ref_score_path,
        cfg=REF_SCORE_CFG,
        ink_thr=PK_CFG.ink_thr,
        ink_temp=PK_CFG.ink_temp,
    )
    print(f"[ref] π(d): mean_d={mean_d:.4f}, std_d={std_d:.4f}")

    # 6) Evidence p(d) from real MNIST class-k
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

    # 7) PK guidance during latent diffusion sampling
    guidance = DecodedInkPKGuidance(
        vae_decoder=vae_eval.decoder,
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
    logits_pk = vae_eval.decoder(z_pk, rng=None, train=False)
    x_pk = jax.nn.sigmoid(logits_pk)

    # 8) Visual comparison
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

    out = paths.run_dir / "compare_grid.png"
    plt.savefig(out, dpi=180)
    plt.show()
    print("[saved]", out)
    print("[artifacts]", paths.run_dir)
    print("[prior_ckpt]", paths.prior_ckpt)
    print("[vae_ckpt]", paths.vae_path)
    print("[z_cache]", paths.z_cache)
    print("[ref_score]", paths.ref_score_path)


if __name__ == "__main__":
    main()
