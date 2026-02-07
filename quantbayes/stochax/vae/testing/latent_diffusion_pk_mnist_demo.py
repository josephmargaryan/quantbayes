# ============================================================
# VAE + Latent EDM Prior + PK(Decoded Ink) (Unconditional) — Publication Demo
#
# Adds:
#  - Calibration sweep over guidance schedule to avoid "too sharp" PK
#  - Evidence-only baseline vs PK
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
    LatentEDMMLP,
    LatentEDMConfig,
    EDMNet,
    LatentEDMTrainConfig,
    train_or_load_latent_edm_prior,
    LatentEDMSampleConfig,
    make_latent_denoise_fn,
    sample_latent_edm,
    collect_latents_from_vae,
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
    wrap_denoise_fn_with_x0_guidance,
)
from quantbayes.stochax.vae.testing.pk_prior_mnist_demo import PKPriorConfig

# ----------------------------
# USER CONTROLS
# ----------------------------
SEED = 0
ARTIFACT_ROOT = Path("artifacts/vae_latent_edm_pk_mnist_pub")

VAE_NAME = "conv"  # "conv" | "mlp" | "vit"
PRIOR_NAME = "latent_mlp"
LATENT_DIM = 16

VAE_LOAD_PATH: Optional[Path] = None
LATENT_PRIOR_CKPT_DIR: Optional[Path] = None

TRAIN_IF_MISSING_VAE = True
TRAIN_IF_MISSING_PRIOR = True
RESUME_TRAINING_PRIOR = True

VAE_EPOCHS = 25
VAE_BATCH = 128

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

# Evidence definition
TARGET_DIGIT = 3
FAT_QUANTILE = 0.80
EVID_SHARPEN = 0.60

# Ref score net
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

# Sample sizes for visuals
NUM_FINAL = 2048
NUM_SHOW = 64
REF_SAMPLES = 4096

# ----------------------------
# Calibration (key improvement)
# ----------------------------
DO_CALIBRATION = True

CALIB_NUM = 512  # small but stable
CALIB_STEPS = PRIOR_SAMPLE.steps  # keep aligned
CALIB_GRIDS = dict(
    guide_strength=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    guide_sigma_min=[0.0, 0.4, 0.8, 1.2],
    guide_sigma_max=[1.5, 3.0, 5.0, 10.0],
    w_gamma=[0.0, 0.5, 1.0, 2.0],
    sigma_weight_mode=["edm", "high", "flat"],
    scale_by_sigma2=[True],  # keep True by default; add False if you want to explore
)

# objective weights
CALIB_W_MEAN = 1.0
CALIB_W_STD = 1.0
CALIB_W_W1 = 0.25
CALIB_W_SHARPNESS = 2.0  # penalize std < tau_z strongly


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


def load_mnist_nchw_01():
    tr = tfds.load("mnist", split="train", as_supervised=True, batch_size=-1)
    te = tfds.load("mnist", split="test", as_supervised=True, batch_size=-1)
    (x_tr, y_tr) = tfds.as_numpy(tr)
    (x_te, y_te) = tfds.as_numpy(te)
    x_tr = x_tr.astype(np.float32) / 255.0
    x_te = x_te.astype(np.float32) / 255.0
    x_tr = np.transpose(x_tr, (0, 3, 1, 2))  # NCHW
    x_te = np.transpose(x_te, (0, 3, 1, 2))
    return (
        jnp.asarray(x_tr),
        jnp.asarray(y_tr.astype(np.int32)),
        jnp.asarray(x_te),
        jnp.asarray(y_te.astype(np.int32)),
    )


def make_grid(x01: np.ndarray, nrow: int) -> np.ndarray:
    x = np.asarray(x01)
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    x = x[: nrow * nrow]
    x = x.reshape(nrow, nrow, 28, 28)
    rows = [
        np.concatenate([x[i, j] for j in range(nrow)], axis=-1) for i in range(nrow)
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


def build_latent_prior_model(key: jr.PRNGKey):
    name = PRIOR_NAME.lower().replace("-", "").replace("_", "").strip()
    if name in ("latentmlp", "latent_mlp"):
        edm_cfg = LatentEDMConfig(
            latent_dim=LATENT_DIM, hidden=256, depth=3, time_emb_dim=64
        )
        return EDMNet(LatentEDMMLP(edm_cfg, key=key))
    raise ValueError(f"Unknown PRIOR_NAME={PRIOR_NAME!r}")


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
    prior_ckpt: Path
    ref_score_path: Path
    calib_csv: Path
    calib_best_json: Path
    summary_json: Path


def make_paths() -> Paths:
    _ensure_dir(ARTIFACT_ROOT)
    tag = f"vae_{VAE_NAME}_z{LATENT_DIM}__prior_{PRIOR_NAME}__digit{TARGET_DIGIT}__sh{int(round(100*EVID_SHARPEN)):03d}"
    run_dir = ARTIFACT_ROOT / tag
    _ensure_dir(run_dir)
    prior_ckpt = (
        LATENT_PRIOR_CKPT_DIR
        if LATENT_PRIOR_CKPT_DIR is not None
        else (run_dir / "latent_edm_ckpt")
    )
    _ensure_dir(prior_ckpt)

    return Paths(
        run_dir=run_dir,
        vae_path=run_dir / "vae.eqx",
        z_cache=run_dir / "z_agg.npy",
        prior_ckpt=prior_ckpt,
        ref_score_path=run_dir / "score_pi_z_decoded_ink.eqx",
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


def get_or_compute_z_agg(vae_eval, x_train01: jnp.ndarray, paths: Paths) -> jnp.ndarray:
    if paths.z_cache.exists():
        z = jnp.asarray(np.load(paths.z_cache))
        print("[z_agg] loaded:", paths.z_cache, z.shape)
        return z

    x_in = prep_x_for_vae(x_train01)
    z = collect_latents_from_vae(
        vae_eval,
        x_in,
        key=jr.PRNGKey(SEED + 111),
        batch_size=512,
        num_samples=None,
        use_mu=False,
    )
    np.save(paths.z_cache, np.asarray(z))
    print("[z_agg] cached:", paths.z_cache, z.shape)
    return z


def load_latent_prior_ema_only(
    ckpt_dir: Path, model_template, cfg: LatentEDMTrainConfig
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


def get_or_train_latent_prior(z_data: jnp.ndarray, paths: Paths):
    ckpt_dir = paths.prior_ckpt
    has_ckpt = (ckpt_dir / "latest.txt").exists()

    model_template = build_latent_prior_model(jr.PRNGKey(SEED + 222))

    if has_ckpt and not RESUME_TRAINING_PRIOR:
        ema_model, step = load_latent_prior_ema_only(
            ckpt_dir, model_template, PRIOR_TRAIN
        )
        print(f"[latent prior] loaded (no-resume) step={step} from {ckpt_dir}")
        return ema_model

    if not has_ckpt and not TRAIN_IF_MISSING_PRIOR:
        raise FileNotFoundError(f"Missing latent prior checkpoint at {ckpt_dir}")

    _, ema_model = train_or_load_latent_edm_prior(
        ckpt_dir=ckpt_dir,
        model_template=model_template,
        z_dataset=z_data,
        cfg=PRIOR_TRAIN,
    )
    print("[latent prior] ready:", ckpt_dir)
    return ema_model


# ----------------------------
# Calibration
# ----------------------------
def calibration_loss(mean_z, std_z, w1, mu_z, tau_z):
    denom = float(tau_z + 1e-12)
    mean_err = (mean_z - mu_z) / denom
    std_err = (std_z - tau_z) / denom

    # "sharpness penalty": if std is smaller than target, penalize harder
    sharp = max(0.0, (tau_z - std_z) / denom)
    return (
        CALIB_W_MEAN * (mean_err**2)
        + CALIB_W_STD * (std_err**2)
        + CALIB_W_W1 * (w1 / denom)
        + CALIB_W_SHARPNESS * (sharp**2)
    )


def calibrate_guidance(
    *,
    ema_prior,
    vae_eval,
    denoise_base,
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

    # Use same sampler config for calibration (or reduce steps if you want speed)
    calib_sample_cfg = LatentEDMSampleConfig(
        steps=CALIB_STEPS,
        sigma_min=PRIOR_SAMPLE.sigma_min,
        sigma_max=PRIOR_SAMPLE.sigma_max,
        sigma_data=PRIOR_SAMPLE.sigma_data,
        rho=PRIOR_SAMPLE.rho,
        sampler=PRIOR_SAMPLE.sampler,
    )

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
                                ink_thr=PK_CFG.ink_thr,
                                ink_temp=PK_CFG.ink_temp,
                                guide_strength=float(lam),
                                guide_sigma_min=float(smin),
                                guide_sigma_max=float(smax),
                                max_guide_norm=PK_CFG.max_guide_norm,
                                sigma_data=PK_CFG.sigma_data,
                                w_gamma=float(w_gamma),
                                sigma_weight_mode=str(mode),
                                scale_by_sigma2=bool(scale_by_sigma2),
                            )

                            # Evidence-only
                            guidance_evd = DecodedInkPKGuidance(
                                vae_decoder=vae_eval.decoder,
                                evidence=evidence,
                                ref_score_net=ref_score_net,
                                cfg=cfg,
                                mode="evidence",
                            )
                            denoise_evd = wrap_denoise_fn_with_x0_guidance(
                                denoise_base,
                                sigma_data=PRIOR_SAMPLE.sigma_data,
                                guidance=guidance_evd,
                            )
                            z_evd = sample_latent_edm(
                                denoise_evd,
                                key=k,
                                num_samples=CALIB_NUM,
                                latent_dim=LATENT_DIM,
                                cfg=calib_sample_cfg,
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

                            # PK
                            guidance_pk = DecodedInkPKGuidance(
                                vae_decoder=vae_eval.decoder,
                                evidence=evidence,
                                ref_score_net=ref_score_net,
                                cfg=cfg,
                                mode="pk",
                            )
                            denoise_pk = wrap_denoise_fn_with_x0_guidance(
                                denoise_base,
                                sigma_data=PRIOR_SAMPLE.sigma_data,
                                guidance=guidance_pk,
                            )
                            z_pk = sample_latent_edm(
                                denoise_pk,
                                key=jr.fold_in(k, 999),
                                num_samples=CALIB_NUM,
                                latent_dim=LATENT_DIM,
                                cfg=calib_sample_cfg,
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
                                # evidence metrics
                                evd_mean=mean_e,
                                evd_std=std_e,
                                evd_w1=w1_e,
                                evd_loss=loss_e,
                                # pk metrics
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
    _ensure_dir(paths.run_dir)
    print("[run_dir]", paths.run_dir)

    x_train, y_train, _, _ = load_mnist_nchw_01()
    print("[MNIST]", x_train.shape, y_train.shape)

    # VAE
    vae = get_or_train_vae(x_train, paths)
    vae_eval = inference_copy(vae)

    # z_agg
    z_data = get_or_compute_z_agg(vae_eval, x_train, paths)

    # latent prior
    ema_prior = get_or_train_latent_prior(z_data, paths)
    denoise_base = make_latent_denoise_fn(ema_prior, sigma_data=PRIOR_SAMPLE.sigma_data)

    # sample base (LatentEDM) and std normal
    key = jr.PRNGKey(SEED + 333)
    k_std, k_base, k_ref = jr.split(key, 3)

    z_std = jr.normal(k_std, (NUM_FINAL, LATENT_DIM))
    x_std = decode_to_x01(vae_eval, z_std)

    z_base = sample_latent_edm(
        denoise_base,
        key=k_base,
        num_samples=NUM_FINAL,
        latent_dim=LATENT_DIM,
        cfg=PRIOR_SAMPLE,
    )
    x_base = decode_to_x01(vae_eval, z_base)

    # reference π(d)
    z_ref = sample_latent_edm(
        denoise_base,
        key=k_ref,
        num_samples=REF_SAMPLES,
        latent_dim=LATENT_DIM,
        cfg=PRIOR_SAMPLE,
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
    print(f"[ref] π(d): mean_d={mean_d:.4f}, std_d={std_d:.4f}")

    # evidence from real data
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
    print(
        f"[evidence] mu_d={mu_d:.4f}, tau_d={tau_d:.4f} | mu_z={mu_z:.3f}, tau_z={tau_z:.3f}"
    )

    evidence = InkEvidence(mean_d=mean_d, std_d=std_d, mu_z=mu_z, tau_z=tau_z)

    # calibration -> best cfg
    if DO_CALIBRATION:
        best = calibrate_guidance(
            ema_prior=ema_prior,
            vae_eval=vae_eval,
            denoise_base=denoise_base,
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
        cfg_best = DecodedInkPKConfig()  # default

    # Evidence-only (for contrast)
    guidance_evd = DecodedInkPKGuidance(
        vae_decoder=vae_eval.decoder,
        evidence=evidence,
        ref_score_net=ref_score_net,
        cfg=cfg_best,
        mode="evidence",
    )
    denoise_evd = wrap_denoise_fn_with_x0_guidance(
        denoise_base, sigma_data=PRIOR_SAMPLE.sigma_data, guidance=guidance_evd
    )
    z_evd = sample_latent_edm(
        denoise_evd,
        key=jr.PRNGKey(SEED + 444),
        num_samples=NUM_FINAL,
        latent_dim=LATENT_DIM,
        cfg=PRIOR_SAMPLE,
    )
    x_evd = decode_to_x01(vae_eval, z_evd)

    # PK
    guidance_pk = DecodedInkPKGuidance(
        vae_decoder=vae_eval.decoder,
        evidence=evidence,
        ref_score_net=ref_score_net,
        cfg=cfg_best,
        mode="pk",
    )
    denoise_pk = wrap_denoise_fn_with_x0_guidance(
        denoise_base, sigma_data=PRIOR_SAMPLE.sigma_data, guidance=guidance_pk
    )
    z_pk = sample_latent_edm(
        denoise_pk,
        key=jr.PRNGKey(SEED + 555),
        num_samples=NUM_FINAL,
        latent_dim=LATENT_DIM,
        cfg=PRIOR_SAMPLE,
    )
    x_pk = decode_to_x01(vae_eval, z_pk)

    # ----------------------------
    # Visual 1: image grid (4-way)
    # ----------------------------
    nrow = int(round(math.sqrt(NUM_SHOW)))
    assert nrow * nrow == NUM_SHOW

    g_std = make_grid(np.asarray(x_std[:NUM_SHOW]), nrow)
    g_base = make_grid(np.asarray(x_base[:NUM_SHOW]), nrow)
    g_evd = make_grid(np.asarray(x_evd[:NUM_SHOW]), nrow)
    g_pk = make_grid(np.asarray(x_pk[:NUM_SHOW]), nrow)

    plt.figure(figsize=(22, 5))
    for i, (img, title) in enumerate(
        [
            (g_std, "VAE: z ~ N(0,1)"),
            (g_base, "VAE: z ~ Latent EDM"),
            (g_evd, "Latent EDM + Evidence-only(ink)"),
            (g_pk, "Latent EDM + PK(ink)"),
        ],
        start=1,
    ):
        plt.subplot(1, 4, i)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    out_grid = paths.run_dir / "compare_grid_4way.png"
    plt.savefig(out_grid, dpi=180)
    plt.show()
    print("[saved]", out_grid)

    # ----------------------------
    # Visual 2: 1D z-space distributions
    # ----------------------------
    def ink_z(x01_np: np.ndarray) -> np.ndarray:
        d = np.asarray(ink_fraction_01(jnp.asarray(x01_np), thr=0.35, temp=0.08))
        return (d - mean_d) / std_d

    x_ref = decode_to_x01(vae_eval, z_ref)
    z_ref_d = ink_z(np.asarray(x_ref))
    z_std_d = ink_z(np.asarray(x_std))
    z_base_d = ink_z(np.asarray(x_base))
    z_evd_d = ink_z(np.asarray(x_evd))
    z_pk_d = ink_z(np.asarray(x_pk))

    lo = float(min(z_ref_d.min(), z_pk_d.min(), mu_z - 4 * tau_z)) - 0.5
    hi = float(max(z_ref_d.max(), z_pk_d.max(), mu_z + 4 * tau_z)) + 0.5
    grid = np.linspace(lo, hi, 600)

    plt.figure(figsize=(10, 4))
    plt.hist(z_ref_d, bins=60, density=True, alpha=0.25, label="π(z) ref")
    plt.hist(
        z_std_d,
        bins=60,
        density=True,
        histtype="step",
        linewidth=2,
        alpha=0.7,
        label="N(0,1)",
    )
    plt.hist(
        z_base_d, bins=60, density=True, histtype="step", linewidth=2, label="LatentEDM"
    )
    plt.hist(
        z_evd_d,
        bins=60,
        density=True,
        histtype="step",
        linewidth=2,
        label="Evidence-only",
    )
    plt.hist(z_pk_d, bins=60, density=True, histtype="step", linewidth=2, label="PK")
    plt.plot(grid, gaussian_pdf(grid, mu_z, tau_z), linewidth=2, label="target p(z)")
    plt.title("Decoded ink distribution in z-space")
    plt.xlabel("z")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    out_hist = paths.run_dir / "ink_z_hist.png"
    plt.savefig(out_hist, dpi=180)
    plt.show()
    print("[saved]", out_hist)

    # ----------------------------
    # Visual 3: score sanity check
    # ----------------------------
    centers, fd_score = finite_diff_score_from_hist(
        z_ref_d, lo, hi, bins=400, smooth_sigma_bins=2.0
    )
    s_learn = np.asarray(ref_score_net(jnp.asarray(centers, dtype=jnp.float32)))

    plt.figure(figsize=(9, 4))
    plt.plot(centers, fd_score, linewidth=2, label="finite-diff (smoothed hist)")
    plt.plot(centers, s_learn, linewidth=2, label="learned s_pi(z)")
    plt.title("Reference score sanity check")
    plt.xlabel("z")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    out_score = paths.run_dir / "ref_score_check.png"
    plt.savefig(out_score, dpi=180)
    plt.show()
    print("[saved]", out_score)

    # ----------------------------
    # Summary
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
        prior_name=PRIOR_NAME,
        latent_dim=LATENT_DIM,
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
            std_normal=summarize(z_std_d),
            latent_edm=summarize(z_base_d),
            evidence=summarize(z_evd_d),
            pk=summarize(z_pk_d),
        ),
        artifacts=dict(
            compare_grid=str(out_grid),
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
