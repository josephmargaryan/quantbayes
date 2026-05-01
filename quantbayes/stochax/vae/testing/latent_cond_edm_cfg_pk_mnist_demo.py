# ============================================================
# Conditional Latent EDM (CFG) + PK(Decoded Ink) — Publication Demo
#
# Adds:
#  - Calibration sweep for PK within one class (label-conditional reference π(d|y))
#  - Evidence-only vs PK comparison
#  - Strong visuals + CSV logs
# ============================================================

from __future__ import annotations

import csv
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
)
from quantbayes.stochax.vae.latent_diffusion.coarse import ink_fraction_01
from quantbayes.stochax.diffusion.checkpoint import load_checkpoint
from quantbayes.stochax.diffusion.pk.reference_score import ScoreNet1DConfig

from quantbayes.stochax.vae.latent_diffusion.pk_guidance import (
    DecodedInkPKConfig,
    InkEvidence,
    compute_ink_evidence_from_real_data,
    train_reference_score_for_decoded_ink,
    DecodedInkPKGuidance,
)

# ----------------------------
# USER CONTROLS
# ----------------------------
SEED = 0
NUM_CLASSES = 10
LATENT_DIM = 16

ARTIFACT_ROOT = Path("artifacts/vae_latent_cond_edm_cfg_pk_mnist_pub")

VAE_NAME = "conv"  # "conv" | "mlp" | "vit"
COND_PRIOR_NAME = "cond_mlp"

VAE_LOAD_PATH: Optional[Path] = None
COND_PRIOR_CKPT_DIR: Optional[Path] = None

TRAIN_IF_MISSING_VAE = True
TRAIN_IF_MISSING_PRIOR = True
RESUME_TRAINING_PRIOR = True

VAE_EPOCHS = 25
VAE_BATCH = 128

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
    p_uncond=0.10,
)

PRIOR_SAMPLE = LatentEDMCondSampleConfig(
    steps=40,
    sigma_min=0.002,
    sigma_max=80.0,
    sigma_data=0.5,
    rho=7.0,
    cfg_scale=3.0,
)

TARGET_DIGIT = 3
FAT_QUANTILE = 0.80
EVID_SHARPEN = 0.60

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
PER_CLASS = 16
GRID_SHOW = 64
REF_SAMPLES = 4096
NUM_FINAL = 2048

# ----------------------------
# Calibration
# ----------------------------
DO_CALIBRATION = True

CALIB_NUM = 512
CALIB_GRIDS = dict(
    guide_strength=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    guide_sigma_min=[0.0, 0.4, 0.8, 1.2],
    guide_sigma_max=[1.5, 3.0, 5.0, 10.0],
    w_gamma=[0.0, 0.5, 1.0, 2.0],
    sigma_weight_mode=["edm", "high", "flat"],
    scale_by_sigma2=[True],
)

CALIB_W_MEAN = 1.0
CALIB_W_STD = 1.0
CALIB_W_W1 = 0.25
CALIB_W_SHARPNESS = 2.0


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


def load_mnist_nchw_01_train():
    tr = tfds.load("mnist", split="train", as_supervised=True, batch_size=-1)
    (x_tr, y_tr) = tfds.as_numpy(tr)
    x_tr = x_tr.astype(np.float32) / 255.0
    x_tr = np.transpose(x_tr, (0, 3, 1, 2))
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


def save_csv(path: Path, rows: list[dict]) -> None:
    _ensure_dir(path.parent)
    if not rows:
        raise ValueError("No rows to save.")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


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


def prep_x_for_vae(x01: jnp.ndarray) -> jnp.ndarray:
    name = VAE_NAME.lower().replace("-", "").replace("_", "").strip()
    if name == "mlp":
        return x01.reshape((x01.shape[0], -1))
    return x01


def decode_to_x01(vae_eval, z: jnp.ndarray) -> jnp.ndarray:
    name = VAE_NAME.lower().replace("-", "").replace("_", "").strip()
    logits = vae_eval.decoder(z, rng=None, train=False)
    x01 = jax.nn.sigmoid(logits)
    if name == "mlp":
        x01 = x01.reshape((x01.shape[0], 1, 28, 28))
    return x01


def decode_logits_fn_factory(vae_eval):
    name = VAE_NAME.lower().replace("-", "").replace("_", "").strip()

    def decode_logits(z):
        logits = vae_eval.decoder(z, rng=None, train=False)
        if name == "mlp":
            logits = logits.reshape((logits.shape[0], 1, 28, 28))
        return logits

    return decode_logits


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
    calib_csv: Path
    calib_best_json: Path
    summary_json: Path


def make_paths() -> Paths:
    _ensure_dir(ARTIFACT_ROOT)
    tag = f"vae_{VAE_NAME}_z{LATENT_DIM}__cond_{COND_PRIOR_NAME}__cfg{PRIOR_SAMPLE.cfg_scale}__label{TARGET_DIGIT}__sh{int(round(100*EVID_SHARPEN)):03d}"
    run_dir = ARTIFACT_ROOT / tag
    _ensure_dir(run_dir)
    prior_ckpt = (
        COND_PRIOR_CKPT_DIR
        if COND_PRIOR_CKPT_DIR is not None
        else (run_dir / "latent_edm_cond_ckpt")
    )
    _ensure_dir(prior_ckpt)
    return Paths(
        run_dir=run_dir,
        vae_path=run_dir / "vae.eqx",
        z_cache=run_dir / "z_agg.npy",
        y_cache=run_dir / "y_agg.npy",
        prior_ckpt=prior_ckpt,
        ref_score_path=run_dir / f"score_pi_z_decoded_ink_label{TARGET_DIGIT}.eqx",
        calib_csv=run_dir / "calibration_sweep.csv",
        calib_best_json=run_dir / "calibration_best.json",
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
    _, ema_model, _, step = load_checkpoint(
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
# Calibration
# ----------------------------
def calibration_loss(mean_z, std_z, w1, mu_z, tau_z):
    denom = float(tau_z + 1e-12)
    mean_err = (mean_z - mu_z) / denom
    std_err = (std_z - tau_z) / denom
    sharp = max(0.0, (tau_z - std_z) / denom)
    return (
        CALIB_W_MEAN * (mean_err**2)
        + CALIB_W_STD * (std_err**2)
        + CALIB_W_W1 * (w1 / denom)
        + CALIB_W_SHARPNESS * (sharp**2)
    )


def calibrate_guidance_conditional(
    *,
    ema_model,
    null_label,
    vae_eval,
    ref_score_net,
    evidence: InkEvidence,
    mean_d: float,
    std_d: float,
    mu_z: float,
    tau_z: float,
    paths: Paths,
):
    rows = []
    best = None
    idx = 0
    base_key = jr.PRNGKey(SEED + 9000)

    for lam in CALIB_GRIDS["guide_strength"]:
        for smin in CALIB_GRIDS["guide_sigma_min"]:
            for smax in CALIB_GRIDS["guide_sigma_max"]:
                if smin > smax:
                    continue
                for w_gamma in CALIB_GRIDS["w_gamma"]:
                    for mode in CALIB_GRIDS["sigma_weight_mode"]:
                        for scale_by_sigma2 in CALIB_GRIDS["scale_by_sigma2"]:
                            idx += 1
                            k = jr.fold_in(base_key, idx)

                            cfg = DecodedInkPKConfig(
                                ink_thr=0.35,
                                ink_temp=0.08,
                                guide_strength=float(lam),
                                guide_sigma_min=float(smin),
                                guide_sigma_max=float(smax),
                                max_guide_norm=50.0,
                                sigma_data=0.5,
                                w_gamma=float(w_gamma),
                                sigma_weight_mode=str(mode),
                                scale_by_sigma2=bool(scale_by_sigma2),
                            )

                            # evidence-only
                            g_evd = DecodedInkPKGuidance(
                                vae_decoder=vae_eval.decoder,
                                evidence=evidence,
                                ref_score_net=ref_score_net,
                                cfg=cfg,
                                mode="evidence",
                            )
                            z_evd = sample_latent_edm_conditional_cfg(
                                ema_model=ema_model,
                                key=k,
                                label=TARGET_DIGIT,
                                num_samples=CALIB_NUM,
                                latent_dim=LATENT_DIM,
                                cfg=PRIOR_SAMPLE,
                                null_label=null_label,
                                pk_guidance=g_evd,
                            )
                            x_evd = decode_to_x01(vae_eval, z_evd)
                            d_evd = np.asarray(
                                ink_fraction_01(
                                    jnp.asarray(x_evd),
                                    thr=cfg.ink_thr,
                                    temp=cfg.ink_temp,
                                )
                            )
                            z_evd_d = (d_evd - mean_d) / std_d
                            mean_e = float(np.mean(z_evd_d))
                            std_e = float(np.std(z_evd_d) + 1e-12)
                            w1_e = float(
                                w1_to_gaussian(
                                    z_evd_d,
                                    mu=mu_z,
                                    tau=tau_z,
                                    seed=SEED + 100000 + idx,
                                )
                            )
                            loss_e = calibration_loss(mean_e, std_e, w1_e, mu_z, tau_z)

                            # pk
                            g_pk = DecodedInkPKGuidance(
                                vae_decoder=vae_eval.decoder,
                                evidence=evidence,
                                ref_score_net=ref_score_net,
                                cfg=cfg,
                                mode="pk",
                            )
                            z_pk = sample_latent_edm_conditional_cfg(
                                ema_model=ema_model,
                                key=jr.fold_in(k, 999),
                                label=TARGET_DIGIT,
                                num_samples=CALIB_NUM,
                                latent_dim=LATENT_DIM,
                                cfg=PRIOR_SAMPLE,
                                null_label=null_label,
                                pk_guidance=g_pk,
                            )
                            x_pk = decode_to_x01(vae_eval, z_pk)
                            d_pk = np.asarray(
                                ink_fraction_01(
                                    jnp.asarray(x_pk),
                                    thr=cfg.ink_thr,
                                    temp=cfg.ink_temp,
                                )
                            )
                            z_pk_d = (d_pk - mean_d) / std_d
                            mean_p = float(np.mean(z_pk_d))
                            std_p = float(np.std(z_pk_d) + 1e-12)
                            w1_p = float(
                                w1_to_gaussian(
                                    z_pk_d, mu=mu_z, tau=tau_z, seed=SEED + 200000 + idx
                                )
                            )
                            loss_p = calibration_loss(mean_p, std_p, w1_p, mu_z, tau_z)

                            rec = dict(
                                lam=float(lam),
                                sigma_min=float(smin),
                                sigma_max=float(smax),
                                w_gamma=float(w_gamma),
                                sigma_weight_mode=str(mode),
                                scale_by_sigma2=bool(scale_by_sigma2),
                                evd_mean=mean_e,
                                evd_std=std_e,
                                evd_w1=w1_e,
                                evd_loss=loss_e,
                                pk_mean=mean_p,
                                pk_std=std_p,
                                pk_w1=w1_p,
                                pk_loss=loss_p,
                            )
                            rows.append(rec)
                            if (best is None) or (loss_p < best["pk_loss"]):
                                best = rec

    assert best is not None
    save_csv(paths.calib_csv, rows)
    _save_json(paths.calib_best_json, best)
    print("[calib] saved:", paths.calib_csv)
    print("[calib] best:", paths.calib_best_json)
    print(
        f"[best PK] lam={best['lam']} | [{best['sigma_min']},{best['sigma_max']}]"
        f" | mode={best['sigma_weight_mode']} | wγ={best['w_gamma']} | loss={best['pk_loss']:.4f}"
        f" | mean={best['pk_mean']:.3f} std={best['pk_std']:.3f} w1={best['pk_w1']:.4f}"
    )
    return best


# ----------------------------
# Main
# ----------------------------
def main():
    paths = make_paths()
    print("[run_dir]", paths.run_dir)

    x_train, y_train = load_mnist_nchw_01_train()
    print("[MNIST]", x_train.shape, y_train.shape)

    # VAE
    vae = get_or_train_vae(x_train, paths)
    vae_eval = inference_copy(vae)

    # z,y cache
    z_data, y_data = get_or_compute_z_y(vae_eval, x_train, y_train, paths)

    # cond prior
    ema_model, null_label = get_or_train_cond_prior(z_data, y_data, paths)

    # Visual: CFG mosaic
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
        imgs_by_label.append(np.asarray(decode_to_x01(vae_eval, z)))

    mosaic = np.concatenate(
        [
            make_grid(imgs_by_label[lbl], nrow=1, ncol=PER_CLASS)
            for lbl in range(NUM_CLASSES)
        ],
        axis=0,
    )
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

    # reference π(d|y)
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
    decode_logits_fn = decode_logits_fn_factory(vae_eval)
    ref_score_net, mean_d, std_d = train_reference_score_for_decoded_ink(
        vae=vae_eval,
        decode_logits_fn=decode_logits_fn,
        z_samples=z_ref,
        score_path=paths.ref_score_path,
        cfg=REF_SCORE_CFG,
        ink_thr=0.35,
        ink_temp=0.08,
    )
    print(f"[ref π(d|y)] mean_d={mean_d:.4f}, std_d={std_d:.4f} (label={TARGET_DIGIT})")

    # evidence
    mu_d, tau_d = compute_ink_evidence_from_real_data(
        x_train,
        digit_labels=y_train,
        target_digit=TARGET_DIGIT,
        fat_quantile=FAT_QUANTILE,
        sharpen=EVID_SHARPEN,
        ink_thr=0.35,
        ink_temp=0.08,
    )
    mu_z = float((mu_d - mean_d) / std_d)
    tau_z = float(tau_d / std_d)
    evidence = InkEvidence(mean_d=mean_d, std_d=std_d, mu_z=mu_z, tau_z=tau_z)
    print(
        f"[evidence] mu_d={mu_d:.4f}, tau_d={tau_d:.4f} | mu_z={mu_z:.3f}, tau_z={tau_z:.3f}"
    )

    # calibration -> best cfg
    if DO_CALIBRATION:
        best = calibrate_guidance_conditional(
            ema_model=ema_model,
            null_label=null_label,
            vae_eval=vae_eval,
            ref_score_net=ref_score_net,
            evidence=evidence,
            mean_d=mean_d,
            std_d=std_d,
            mu_z=mu_z,
            tau_z=tau_z,
            paths=paths,
        )
        cfg_best = DecodedInkPKConfig(
            ink_thr=0.35,
            ink_temp=0.08,
            guide_strength=float(best["lam"]),
            guide_sigma_min=float(best["sigma_min"]),
            guide_sigma_max=float(best["sigma_max"]),
            max_guide_norm=50.0,
            sigma_data=0.5,
            w_gamma=float(best["w_gamma"]),
            sigma_weight_mode=str(best["sigma_weight_mode"]),
            scale_by_sigma2=bool(best["scale_by_sigma2"]),
        )
    else:
        cfg_best = DecodedInkPKConfig()

    # baseline CFG-only
    z_base = sample_latent_edm_conditional_cfg(
        ema_model=ema_model,
        key=jr.PRNGKey(SEED + 7777),
        label=TARGET_DIGIT,
        num_samples=NUM_FINAL,
        latent_dim=LATENT_DIM,
        cfg=PRIOR_SAMPLE,
        null_label=null_label,
        pk_guidance=None,
    )
    x_base = decode_to_x01(vae_eval, z_base)

    # evidence-only
    g_evd = DecodedInkPKGuidance(
        vae_decoder=vae_eval.decoder,
        evidence=evidence,
        ref_score_net=ref_score_net,
        cfg=cfg_best,
        mode="evidence",
    )
    z_evd = sample_latent_edm_conditional_cfg(
        ema_model=ema_model,
        key=jr.PRNGKey(SEED + 8888),
        label=TARGET_DIGIT,
        num_samples=NUM_FINAL,
        latent_dim=LATENT_DIM,
        cfg=PRIOR_SAMPLE,
        null_label=null_label,
        pk_guidance=g_evd,
    )
    x_evd = decode_to_x01(vae_eval, z_evd)

    # pk
    g_pk = DecodedInkPKGuidance(
        vae_decoder=vae_eval.decoder,
        evidence=evidence,
        ref_score_net=ref_score_net,
        cfg=cfg_best,
        mode="pk",
    )
    z_pk = sample_latent_edm_conditional_cfg(
        ema_model=ema_model,
        key=jr.PRNGKey(SEED + 9999),
        label=TARGET_DIGIT,
        num_samples=NUM_FINAL,
        latent_dim=LATENT_DIM,
        cfg=PRIOR_SAMPLE,
        null_label=null_label,
        pk_guidance=g_pk,
    )
    x_pk = decode_to_x01(vae_eval, z_pk)

    # Visual compare (3-way)
    nrow = int(round(math.sqrt(GRID_SHOW)))
    assert nrow * nrow == GRID_SHOW
    g0 = make_grid(np.asarray(x_base[:GRID_SHOW]), nrow=nrow, ncol=nrow)
    g1 = make_grid(np.asarray(x_evd[:GRID_SHOW]), nrow=nrow, ncol=nrow)
    g2 = make_grid(np.asarray(x_pk[:GRID_SHOW]), nrow=nrow, ncol=nrow)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(g0, cmap="gray")
    plt.title(f"CFG only (y={TARGET_DIGIT})")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(g1, cmap="gray")
    plt.title("CFG + evidence-only(ink)")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(g2, cmap="gray")
    plt.title("CFG + PK(ink)")
    plt.axis("off")
    plt.tight_layout()
    out_cmp = paths.run_dir / "pk_within_class_compare_3way.png"
    plt.savefig(out_cmp, dpi=180)
    plt.show()
    print("[saved]", out_cmp)

    # 1D z-space distributions
    def ink_z(x01_np: np.ndarray) -> np.ndarray:
        d = np.asarray(ink_fraction_01(jnp.asarray(x01_np), thr=0.35, temp=0.08))
        return (d - mean_d) / std_d

    x_ref = decode_to_x01(vae_eval, z_ref)
    z_ref_d = ink_z(np.asarray(x_ref))
    z_base_d = ink_z(np.asarray(x_base))
    z_evd_d = ink_z(np.asarray(x_evd))
    z_pk_d = ink_z(np.asarray(x_pk))

    lo = float(min(z_ref_d.min(), z_pk_d.min(), mu_z - 4 * tau_z)) - 0.5
    hi = float(max(z_ref_d.max(), z_pk_d.max(), mu_z + 4 * tau_z)) + 0.5
    grid = np.linspace(lo, hi, 600)

    plt.figure(figsize=(10, 4))
    plt.hist(
        z_ref_d, bins=60, density=True, alpha=0.25, label=f"π(z|y={TARGET_DIGIT}) ref"
    )
    plt.hist(
        z_base_d, bins=60, density=True, histtype="step", linewidth=2, label="CFG only"
    )
    plt.hist(
        z_evd_d,
        bins=60,
        density=True,
        histtype="step",
        linewidth=2,
        label="evidence-only",
    )
    plt.hist(z_pk_d, bins=60, density=True, histtype="step", linewidth=2, label="PK")
    plt.plot(grid, gaussian_pdf(grid, mu_z, tau_z), linewidth=2, label="target p(z)")
    plt.title(f"Decoded ink distribution in z-space (label={TARGET_DIGIT})")
    plt.xlabel("z")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    out_hist = paths.run_dir / f"ink_z_hist_label{TARGET_DIGIT}.png"
    plt.savefig(out_hist, dpi=180)
    plt.show()
    print("[saved]", out_hist)

    # score sanity check
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

    # summary
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
        evid=dict(
            fat_quantile=FAT_QUANTILE,
            sharpen=EVID_SHARPEN,
            mu_d=float(mu_d),
            tau_d=float(tau_d),
            mu_z=float(mu_z),
            tau_z=float(tau_z),
        ),
        ref=dict(mean_d=float(mean_d), std_d=float(std_d)),
        pk_cfg=dict(**cfg_best.__dict__),
        metrics=dict(
            ref=summarize(z_ref_d),
            base=summarize(z_base_d),
            evidence=summarize(z_evd_d),
            pk=summarize(z_pk_d),
        ),
        artifacts=dict(
            cfg_mosaic=str(out_mosaic),
            compare=str(out_cmp),
            ink_hist=str(out_hist),
            score_check=str(out_score),
            calib_csv=str(paths.calib_csv) if DO_CALIBRATION else None,
            calib_best=str(paths.calib_best_json) if DO_CALIBRATION else None,
        ),
    )
    _save_json(paths.summary_json, summary)
    print("[saved]", paths.summary_json)
    print("[done] artifacts:", paths.run_dir)


if __name__ == "__main__":
    main()
