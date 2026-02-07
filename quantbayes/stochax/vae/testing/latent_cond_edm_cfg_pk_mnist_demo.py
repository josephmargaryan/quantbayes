# ============================================================
# VAE + Conditional Latent EDM Prior (CFG) + PK(Decoded Ink) Demo
#
# Production-ready:
#  - Train-or-load VAE (eqx file)
#  - Cache (z,y) aggregated posterior latents (z_agg.npy, y_agg.npy)
#  - Train-or-load class-conditional latent EDM prior (checkpoint dir)
#  - Train-or-load reference score net for decoded ink π(d|y) (label-conditional reference)
#  - PK guidance inside class-conditional CFG sampling
#
# Visuals:
#  1) cfg_class_mosaic.png        : 10 rows (0..9), PER_CLASS columns
#  2) pk_within_class_compare.png : CFG-only vs CFG+PK (one label)
#  3) ink_z_hist_labelK.png       : π(z|y) vs base vs PK + target Gaussian
#  4) ref_score_check_labelK.png  : learned s_pi(z|y) vs smoothed finite-diff estimate
# ============================================================

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
import optax
import tensorflow_datasets as tfds

from quantbayes.stochax.vae.components import ConvVAE, MLP_VAE, ViT_VAE
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
from quantbayes.stochax.diffusion.checkpoint import load_checkpoint
from quantbayes.stochax.diffusion.pk.reference_score import ScoreNet1DConfig


# ----------------------------
# USER CONTROLS
# ----------------------------
SEED = 0
NUM_CLASSES = 10
LATENT_DIM = 16

ARTIFACT_ROOT = Path("artifacts/vae_latent_cond_edm_cfg_pk_mnist")

# Model switchboards
VAE_NAME = "conv"  # "conv" | "mlp" | "vit"
COND_PRIOR_NAME = "cond_mlp"  # extend if you add more conditional priors

# Optional pretrained overrides
VAE_LOAD_PATH: Optional[Path] = None
COND_PRIOR_CKPT_DIR: Optional[Path] = None

TRAIN_IF_MISSING_VAE = True
TRAIN_IF_MISSING_PRIOR = True
RESUME_TRAINING_PRIOR = True  # if False: load-only when ckpt exists

# VAE training (if missing)
VAE_EPOCHS = 25
VAE_BATCH = 128

# Conditional latent EDM prior training
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

# Conditional sampling with CFG
PRIOR_SAMPLE = LatentEDMCondSampleConfig(
    steps=40,
    sigma_min=0.002,
    sigma_max=80.0,
    sigma_data=0.5,
    rho=7.0,
    cfg_scale=3.0,
)

# PK decoded-ink steering within a class
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

# Reference score net for π(d|y)
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
PER_CLASS = 16  # 1x16 per class row in mosaic
GRID_SHOW = 64  # 8x8 compare
REF_SAMPLES = 4096


# ----------------------------
# Helpers
# ----------------------------
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


def _fmt_float(x: float) -> str:
    s = f"{float(x):.4g}"
    return s.replace(".", "p").replace("-", "m")


def load_mnist_nchw_01_train():
    tr = tfds.load("mnist", split="train", as_supervised=True, batch_size=-1)
    (x_tr, y_tr) = tfds.as_numpy(tr)
    x_tr = x_tr.astype(np.float32) / 255.0
    x_tr = np.transpose(x_tr, (0, 3, 1, 2))  # NCHW
    return jnp.asarray(x_tr), jnp.asarray(y_tr.astype(np.int32))


def make_grid(x01: np.ndarray, nrow: int, ncol: int) -> np.ndarray:
    x = np.asarray(x01)
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    x = x[: nrow * ncol]
    x = x.reshape(nrow, ncol, 28, 28)
    rows = [
        np.concatenate([x[i, j] for j in range(ncol)], axis=-1) for i in range(nrow)
    ]
    return np.concatenate(rows, axis=-2)


def gaussian_pdf(x: np.ndarray, mu: float, tau: float) -> np.ndarray:
    tau = float(max(tau, 1e-12))
    return (1.0 / (np.sqrt(2.0 * np.pi) * tau)) * np.exp(-0.5 * ((x - mu) / tau) ** 2)


def w1_to_gaussian(
    z_samples: np.ndarray, mu: float, tau: float, *, seed: int = 0
) -> float:
    z = np.sort(np.asarray(z_samples).reshape(-1))
    rng = np.random.default_rng(int(seed))
    t = np.sort(rng.normal(loc=float(mu), scale=float(tau), size=z.shape[0]))
    return float(np.mean(np.abs(z - t)))


def finite_diff_score_from_hist(
    z: np.ndarray, lo: float, hi: float, bins: int = 400, smooth_sigma_bins: float = 2.0
):
    hist, edges = np.histogram(z, bins=bins, range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ksize = int(max(5, 6 * smooth_sigma_bins)) | 1
    xs = np.arange(ksize) - ksize // 2
    kern = np.exp(-0.5 * (xs / smooth_sigma_bins) ** 2)
    kern = kern / np.sum(kern)
    smooth = np.convolve(hist, kern, mode="same")
    smooth = np.maximum(smooth, 1e-12)
    logp = np.log(smooth)
    dz = centers[1] - centers[0]
    score = np.gradient(logp, dz)
    return centers, score


# ----------------------------
# Switchboards
# ----------------------------
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


def build_cond_prior_model(key: jr.PRNGKey):
    name = COND_PRIOR_NAME.lower().replace("-", "").replace("_", "").strip()
    if name in ("condmlp", "cond_mlp"):
        cond_cfg = LatentEDMCondConfig(
            latent_dim=LATENT_DIM,
            num_classes=NUM_CLASSES,
            hidden=256,
            depth=3,
            time_emb_dim=64,
            label_emb_dim=64,
        )
        return EDMNet(LatentEDMCondMLP(cond_cfg, key=key)), cond_cfg
    raise ValueError(f"Unknown COND_PRIOR_NAME={COND_PRIOR_NAME!r}")


def prep_x_for_vae(x_nchw01: jnp.ndarray) -> jnp.ndarray:
    name = VAE_NAME.lower().replace("-", "").replace("_", "").strip()
    if name == "mlp":
        return x_nchw01.reshape((x_nchw01.shape[0], -1))
    return x_nchw01


def decode_to_x01(vae_eval, z: jnp.ndarray) -> jnp.ndarray:
    name = VAE_NAME.lower().replace("-", "").replace("_", "").strip()
    logits = vae_eval.decoder(z, rng=None, train=False)
    x01 = jax.nn.sigmoid(logits)
    if name == "mlp":
        x01 = x01.reshape((x01.shape[0], 1, 28, 28))
    return x01


# ----------------------------
# Paths
# ----------------------------
@dataclass(frozen=True)
class Paths:
    run_dir: Path
    vae_path: Path
    z_cache: Path
    y_cache: Path
    prior_ckpt: Path
    ref_score_path: Path
    summary_json: Path


def make_paths() -> Paths:
    tag = (
        f"vae_{VAE_NAME}_z{LATENT_DIM}"
        f"__cond_{COND_PRIOR_NAME}"
        f"__cfg{_fmt_float(PRIOR_SAMPLE.cfg_scale)}"
        f"__label{TARGET_DIGIT}"
        f"__sh{int(round(100 * EVID_SHARPEN)):03d}"
        f"__lam{_fmt_float(PK_CFG.guide_strength)}"
        f"__smax{_fmt_float(PK_CFG.guide_sigma_max)}"
        f"__wg{_fmt_float(PK_CFG.w_gamma)}"
    )
    run_dir = ARTIFACT_ROOT / tag
    _ensure_dir(run_dir)

    prior_dir = (
        COND_PRIOR_CKPT_DIR
        if COND_PRIOR_CKPT_DIR is not None
        else (run_dir / "latent_edm_cond_ckpt")
    )
    _ensure_dir(prior_dir)

    return Paths(
        run_dir=run_dir,
        vae_path=run_dir / "vae.eqx",
        z_cache=run_dir / "z_agg.npy",
        y_cache=run_dir / "y_agg.npy",
        prior_ckpt=prior_dir,
        ref_score_path=run_dir / f"score_pi_z_decoded_ink_label{TARGET_DIGIT}.eqx",
        summary_json=run_dir / "summary.json",
    )


# ----------------------------
# Train-or-load
# ----------------------------
def get_or_train_vae(x_train01: jnp.ndarray, paths: Paths):
    if VAE_LOAD_PATH is not None and Path(VAE_LOAD_PATH).exists():
        template = build_vae(jr.PRNGKey(SEED))
        vae = _load_eqx(Path(VAE_LOAD_PATH), template)
        print("[VAE] loaded override:", VAE_LOAD_PATH)
        return vae

    if paths.vae_path.exists():
        template = build_vae(jr.PRNGKey(SEED))
        vae = _load_eqx(paths.vae_path, template)
        print("[VAE] loaded:", paths.vae_path)
        return vae

    if not TRAIN_IF_MISSING_VAE:
        raise FileNotFoundError(f"Missing VAE checkpoint at {paths.vae_path}")

    print("[VAE] training from scratch...")
    vae = build_vae(jr.PRNGKey(SEED))
    x_in = prep_x_for_vae(x_train01)

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
    print("[VAE] saved:", paths.vae_path)
    return vae


def get_or_compute_z_y(
    vae_eval, x_train01: jnp.ndarray, y_train: jnp.ndarray, paths: Paths
):
    if paths.z_cache.exists() and paths.y_cache.exists():
        z = jnp.asarray(np.load(paths.z_cache))
        y = jnp.asarray(np.load(paths.y_cache)).astype(jnp.int32)
        print("[z,y] loaded cache:", z.shape, y.shape)
        return z, y

    x_in = prep_x_for_vae(x_train01)
    z, y = collect_latents_with_labels_from_vae(
        vae_eval,
        x_in,
        y_train,
        key=jr.PRNGKey(SEED + 111),
        batch_size=512,
        num_samples=None,
        use_mu=False,
    )
    np.save(paths.z_cache, np.asarray(z))
    np.save(paths.y_cache, np.asarray(y))
    print("[z,y] cached:", z.shape, y.shape)
    return z, y


def load_cond_prior_ema_only(
    ckpt_dir: Path, model_template, cfg: LatentEDMCondTrainConfig
):
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(cfg.lr, weight_decay=cfg.weight_decay),
    )
    opt_state_template = optimizer.init(
        eqx.filter(model_template, eqx.is_inexact_array)
    )
    model, ema_model, opt_state, step = load_checkpoint(
        ckpt_dir,
        model_template,
        model_template,
        opt_state_template,
        step=None,
    )
    return ema_model, step


def get_or_train_cond_prior(z_data: jnp.ndarray, y_data: jnp.ndarray, paths: Paths):
    ckpt_dir = paths.prior_ckpt
    has_ckpt = (ckpt_dir / "latest.txt").exists()

    model_template, cond_cfg = build_cond_prior_model(jr.PRNGKey(SEED + 222))
    null_label = cond_cfg.num_classes

    if has_ckpt and not RESUME_TRAINING_PRIOR:
        ema_model, step = load_cond_prior_ema_only(
            ckpt_dir, model_template, PRIOR_TRAIN
        )
        print(f"[cond prior] loaded (no-resume) step={step} from {ckpt_dir}")
        return ema_model, null_label

    if not has_ckpt and not TRAIN_IF_MISSING_PRIOR:
        raise FileNotFoundError(f"Missing conditional prior checkpoint at {ckpt_dir}")

    _, ema_model = train_or_load_latent_edm_prior_conditional(
        ckpt_dir=ckpt_dir,
        model_template=model_template,
        z_dataset=z_data,
        y_dataset=y_data,
        null_label=null_label,
        cfg=PRIOR_TRAIN,
    )
    print("[cond prior] ready:", ckpt_dir)
    return ema_model, null_label


# ----------------------------
# Main
# ----------------------------
def main():
    paths = make_paths()
    print("[run_dir]", paths.run_dir)

    x_train, y_train = load_mnist_nchw_01_train()
    print("[MNIST]", x_train.shape, y_train.shape)

    # 1) VAE
    vae = get_or_train_vae(x_train, paths)
    vae_eval = inference_copy(vae)

    # 2) cache z,y
    z_data, y_data = get_or_compute_z_y(vae_eval, x_train, y_train, paths)

    # 3) conditional latent prior
    ema_model, null_label = get_or_train_cond_prior(z_data, y_data, paths)

    # ----------------------------
    # Visual 1: CFG mosaic (0..9)
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
        x01 = decode_to_x01(vae_eval, z)
        imgs_by_label.append(np.asarray(x01))

    row_imgs = [
        make_grid(imgs_by_label[lbl], nrow=1, ncol=PER_CLASS)
        for lbl in range(NUM_CLASSES)
    ]
    mosaic = np.concatenate(row_imgs, axis=0)

    plt.figure(figsize=(18, 8))
    plt.imshow(mosaic, cmap="gray")
    plt.title(
        f"Class-conditional latent EDM prior + CFG (scale={PRIOR_SAMPLE.cfg_scale})"
    )
    plt.axis("off")
    out_mosaic = paths.run_dir / "cfg_class_mosaic.png"
    plt.savefig(out_mosaic, dpi=180)
    plt.show()
    print("[saved]", out_mosaic)

    # ----------------------------
    # 4) Reference π(d|y) for decoded ink under *the same conditional sampler*
    # ----------------------------
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
        # must return logits (B,1,28,28) for train_reference_score_for_decoded_ink
        name = VAE_NAME.lower().replace("-", "").replace("_", "").strip()
        logits = vae_eval.decoder(z, rng=None, train=False)
        if name == "mlp":
            logits = logits.reshape((logits.shape[0], 1, 28, 28))
        return logits

    ref_score_net, mean_d, std_d = train_reference_score_for_decoded_ink(
        vae=vae_eval,
        decode_logits_fn=decode_logits_fn,
        z_samples=z_ref,
        score_path=paths.ref_score_path,
        cfg=REF_SCORE_CFG,
        ink_thr=PK_CFG.ink_thr,
        ink_temp=PK_CFG.ink_temp,
    )
    print(f"[ref π(d|y)] mean_d={mean_d:.4f}, std_d={std_d:.4f} (label={TARGET_DIGIT})")

    # Evidence p(d) from real data (fat class-k digits)
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

    # PK guidance object
    guidance = DecodedInkPKGuidance(
        vae_decoder=vae_eval.decoder,
        evidence=evidence,
        ref_score_net=ref_score_net,
        cfg=PK_CFG,
        mode="pk",
    )

    # ----------------------------
    # Visual 2: within-class comparison (CFG only vs CFG + PK)
    # ----------------------------
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
    x_base = decode_to_x01(vae_eval, z_base)

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
    x_pk = decode_to_x01(vae_eval, z_pk)

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
    plt.title(
        f"CFG + PK(ink) | λ={PK_CFG.guide_strength}, smax={PK_CFG.guide_sigma_max}"
    )
    plt.axis("off")
    plt.tight_layout()
    out_cmp = paths.run_dir / "pk_within_class_compare.png"
    plt.savefig(out_cmp, dpi=180)
    plt.show()
    print("[saved]", out_cmp)

    # ----------------------------
    # Visual 3: 1D z-space distributions (decoded ink)
    # ----------------------------
    def ink_z(x01_np: np.ndarray) -> np.ndarray:
        d = np.asarray(
            ink_fraction_01(
                jnp.asarray(x01_np), thr=PK_CFG.ink_thr, temp=PK_CFG.ink_temp
            )
        )
        z = (d - mean_d) / std_d
        return z

    x_ref = decode_to_x01(vae_eval, z_ref)
    z_ref_d = ink_z(np.asarray(x_ref))
    z_base_d = ink_z(np.asarray(x_base))
    z_pk_d = ink_z(np.asarray(x_pk))

    lo = float(min(z_ref_d.min(), z_base_d.min(), z_pk_d.min(), mu_z - 4 * tau_z)) - 0.5
    hi = float(max(z_ref_d.max(), z_base_d.max(), z_pk_d.max(), mu_z + 4 * tau_z)) + 0.5
    grid = np.linspace(lo, hi, 600)

    plt.figure(figsize=(10, 4))
    plt.hist(
        z_ref_d, bins=60, density=True, alpha=0.25, label=f"π(z|y={TARGET_DIGIT}) (ref)"
    )
    plt.hist(
        z_base_d,
        bins=60,
        density=True,
        histtype="step",
        linewidth=2,
        label="samples (CFG only)",
    )
    plt.hist(
        z_pk_d,
        bins=60,
        density=True,
        histtype="step",
        linewidth=2,
        label="samples (CFG + PK)",
    )
    plt.plot(
        grid,
        gaussian_pdf(grid, mu_z, tau_z),
        linewidth=2,
        label="target p(z) (Gaussian)",
    )
    plt.title(f"Decoded ink distribution in z-space (label={TARGET_DIGIT})")
    plt.xlabel("z")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    out_hist = paths.run_dir / f"ink_z_hist_label{TARGET_DIGIT}.png"
    plt.savefig(out_hist, dpi=180)
    plt.show()
    print("[saved]", out_hist)

    # ----------------------------
    # Visual 4: reference score check
    # ----------------------------
    centers, fd_score = finite_diff_score_from_hist(
        z_ref_d, lo, hi, bins=400, smooth_sigma_bins=2.0
    )
    s_learn = np.asarray(ref_score_net(jnp.asarray(centers, dtype=jnp.float32)))

    plt.figure(figsize=(9, 4))
    plt.plot(centers, fd_score, linewidth=2, label="finite-diff (smoothed hist)")
    plt.plot(centers, s_learn, linewidth=2, label="learned s_pi(z|y)")
    plt.title(f"Reference score sanity check (label={TARGET_DIGIT})")
    plt.xlabel("z")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    out_score = paths.run_dir / f"ref_score_check_label{TARGET_DIGIT}.png"
    plt.savefig(out_score, dpi=180)
    plt.show()
    print("[saved]", out_score)

    # ----------------------------
    # Metrics + summary
    # ----------------------------
    def summarize(z):
        return dict(
            mean=float(np.mean(z)),
            std=float(np.std(z) + 1e-12),
            w1=float(w1_to_gaussian(z, mu=mu_z, tau=tau_z, seed=SEED + 777)),
        )

    summary = dict(
        seed=SEED,
        vae_name=VAE_NAME,
        cond_prior_name=COND_PRIOR_NAME,
        latent_dim=LATENT_DIM,
        cfg_scale=float(PRIOR_SAMPLE.cfg_scale),
        target_digit=TARGET_DIGIT,
        fat_quantile=FAT_QUANTILE,
        evid_sharpen=EVID_SHARPEN,
        pk=dict(
            lambda_strength=float(PK_CFG.guide_strength),
            sigma_max=float(PK_CFG.guide_sigma_max),
            w_gamma=float(PK_CFG.w_gamma),
            max_guide_norm=float(PK_CFG.max_guide_norm),
        ),
        ref=dict(mean_d=float(mean_d), std_d=float(std_d)),
        evidence=dict(
            mu_d=float(mu_d), tau_d=float(tau_d), mu_z=float(mu_z), tau_z=float(tau_z)
        ),
        metrics=dict(
            ref=summarize(z_ref_d),
            base=summarize(z_base_d),
            pk=summarize(z_pk_d),
        ),
        artifacts=dict(
            cfg_mosaic=str(out_mosaic),
            compare=str(out_cmp),
            ink_hist=str(out_hist),
            score_check=str(out_score),
            vae=str(paths.vae_path),
            z_cache=str(paths.z_cache),
            y_cache=str(paths.y_cache),
            prior_ckpt=str(paths.prior_ckpt),
            ref_score=str(paths.ref_score_path),
        ),
    )
    _save_json(paths.summary_json, summary)
    print("[saved]", paths.summary_json)
    print("[done] artifacts:", paths.run_dir)


if __name__ == "__main__":
    main()
