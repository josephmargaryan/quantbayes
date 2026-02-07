# ============================================================
# VAE + Latent EDM Prior + PK(Decoded Ink) Demo (Unconditional)
#
# Production-ready:
#  - Train-or-load VAE (eqx file)
#  - Cache aggregated posterior latents z_agg.npy
#  - Train-or-load latent EDM prior (checkpoint dir)
#  - Train-or-load reference score net for decoded ink π(d)
#  - PK guidance during latent sampling
#
# Visuals:
#  1) compare_grid.png        : z~N(0,1) vs z~LatentEDM vs z~LatentEDM+PK
#  2) ink_z_hist.png          : π(z_d) vs base vs PK (+ optional N(0,1)) + target Gaussian
#  3) ref_score_check.png     : learned s_pi(z_d) vs smoothed finite-diff estimate
#
# Notes:
#  - Assumes you patched SinusoidalTimeEmb shape bug already.
#  - Designed for MNIST in [0,1], NCHW.
# ============================================================

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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
    DecodedInkPKConfig,
    train_reference_score_for_decoded_ink,
    compute_ink_evidence_from_real_data,
    DecodedInkPKGuidance,
    wrap_denoise_fn_with_x0_guidance,
)
from quantbayes.stochax.vae.latent_diffusion.coarse import ink_fraction_01
from quantbayes.stochax.diffusion.checkpoint import load_checkpoint


# ----------------------------
# USER CONTROLS
# ----------------------------
SEED = 0
ARTIFACT_ROOT = Path("artifacts/vae_latent_edm_pk_mnist")

# Model switchboards
VAE_NAME = "conv"  # "conv" | "mlp" | "vit"
PRIOR_NAME = "latent_mlp"  # extend if you add more prior nets
LATENT_DIM = 16

# Optional pretrained paths (overrides local run dir)
VAE_LOAD_PATH: Optional[Path] = None  # e.g. Path(".../vae.eqx")
LATENT_PRIOR_CKPT_DIR: Optional[Path] = None  # e.g. Path(".../latent_edm_ckpt")

# Train/load behavior
TRAIN_IF_MISSING_VAE = True
TRAIN_IF_MISSING_PRIOR = True
RESUME_TRAINING_PRIOR = True  # if False: load-only when ckpt exists

# VAE training (only if missing)
VAE_EPOCHS = 25
VAE_BATCH = 128

# Latent EDM prior training config
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

# Latent EDM sampling config
PRIOR_SAMPLE = LatentEDMSampleConfig(
    steps=40,
    sigma_min=0.002,
    sigma_max=80.0,
    sigma_data=0.5,
    rho=7.0,
    sampler="dpmpp_3m",
)

# PK steering target (decoded ink)
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

# Ref score net path/config
REF_SCORE_STEPS = 4000
REF_SAMPLES = 4096  # how many prior latents to estimate π(d)

from quantbayes.stochax.diffusion.pk.reference_score import ScoreNet1DConfig

REF_SCORE_CFG = ScoreNet1DConfig(
    hidden=128,
    lr=2e-3,
    weight_decay=1e-4,
    batch_size=512,
    steps=REF_SCORE_STEPS,
    noise_std=0.08,
    seed=SEED,
    print_every=200,
)

# Visuals / sample sizes
NUM_SAMPLES = 1024
NUM_SHOW = 64  # perfect square


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
    # Gaussian smoothing in bin units
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


def build_latent_prior_model(key: jr.PRNGKey):
    name = PRIOR_NAME.lower().replace("-", "").replace("_", "").strip()
    if name in ("latentmlp", "latent_mlp"):
        edm_cfg = LatentEDMConfig(
            latent_dim=LATENT_DIM, hidden=256, depth=3, time_emb_dim=64
        )
        return EDMNet(LatentEDMMLP(edm_cfg, key=key))
    raise ValueError(f"Unknown PRIOR_NAME={PRIOR_NAME!r}")


def prep_x_for_vae(x_nchw01: jnp.ndarray) -> jnp.ndarray:
    """Adapt MNIST inputs to the chosen VAE."""
    name = VAE_NAME.lower().replace("-", "").replace("_", "").strip()
    if name == "mlp":
        return x_nchw01.reshape((x_nchw01.shape[0], -1))
    return x_nchw01


def decode_to_x01(vae_eval, z: jnp.ndarray) -> jnp.ndarray:
    """Return decoded samples as (B,1,28,28) in [0,1] for any VAE type."""
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
    vae_meta: Path
    z_cache: Path
    prior_ckpt: Path
    ref_score_path: Path
    summary_json: Path


def make_paths() -> Paths:
    tag = (
        f"vae_{VAE_NAME}_z{LATENT_DIM}"
        f"__prior_{PRIOR_NAME}"
        f"__digit{TARGET_DIGIT}"
        f"__sh{int(round(100 * EVID_SHARPEN)):03d}"
        f"__lam{_fmt_float(PK_CFG.guide_strength)}"
        f"__smax{_fmt_float(PK_CFG.guide_sigma_max)}"
        f"__wg{_fmt_float(PK_CFG.w_gamma)}"
    )
    run_dir = ARTIFACT_ROOT / tag
    _ensure_dir(run_dir)

    prior_dir = (
        LATENT_PRIOR_CKPT_DIR
        if LATENT_PRIOR_CKPT_DIR is not None
        else (run_dir / "latent_edm_ckpt")
    )
    _ensure_dir(prior_dir)

    return Paths(
        run_dir=run_dir,
        vae_path=run_dir / "vae.eqx",
        vae_meta=run_dir / "vae_meta.json",
        z_cache=run_dir / "z_agg.npy",
        prior_ckpt=prior_dir,
        ref_score_path=run_dir / "score_pi_z_decoded_ink.eqx",
        summary_json=run_dir / "summary.json",
    )


# ----------------------------
# Train-or-load functions
# ----------------------------
def get_or_train_vae(x_train01: jnp.ndarray, paths: Paths):
    # external override
    if VAE_LOAD_PATH is not None and Path(VAE_LOAD_PATH).exists():
        template = build_vae(jr.PRNGKey(SEED))
        vae = _load_eqx(Path(VAE_LOAD_PATH), template)
        print("[VAE] loaded override:", VAE_LOAD_PATH)
        return vae

    # local cache
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
    print("[VAE] saved:", paths.vae_path)
    return vae


def get_or_compute_z_agg(vae_eval, x_train01: jnp.ndarray, paths: Paths) -> jnp.ndarray:
    if paths.z_cache.exists():
        z_np = np.load(paths.z_cache)
        z = jnp.asarray(z_np)
        print("[z_agg] loaded cache:", paths.z_cache, z.shape)
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
    model, ema_model, opt_state, step = load_checkpoint(
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
# Main
# ----------------------------
def main():
    paths = make_paths()
    _ensure_dir(paths.run_dir)
    print("[run_dir]", paths.run_dir)

    x_train, y_train, _, _ = load_mnist_nchw_01()
    print("[MNIST]", x_train.shape, y_train.shape)

    # 1) VAE train-or-load
    vae = get_or_train_vae(x_train, paths)
    vae_eval = inference_copy(vae)

    # 2) z_agg cache
    z_data = get_or_compute_z_agg(vae_eval, x_train, paths)

    # 3) latent EDM prior train-or-load
    ema_prior = get_or_train_latent_prior(z_data, paths)
    denoise_base = make_latent_denoise_fn(ema_prior, sigma_data=PRIOR_SAMPLE.sigma_data)

    # 4) Baselines: z~N(0,1) vs z~LatentEDM
    key = jr.PRNGKey(SEED + 333)
    k0, k1, kref = jr.split(key, 3)

    z_std = jr.normal(k0, (NUM_SAMPLES, LATENT_DIM))
    x_std = decode_to_x01(vae_eval, z_std)

    z_prior = sample_latent_edm(
        denoise_base,
        key=k1,
        num_samples=NUM_SAMPLES,
        latent_dim=LATENT_DIM,
        cfg=PRIOR_SAMPLE,
    )
    x_prior = decode_to_x01(vae_eval, z_prior)

    # 5) Reference score π(d) under latent prior
    z_ref = sample_latent_edm(
        denoise_base,
        key=kref,
        num_samples=REF_SAMPLES,
        latent_dim=LATENT_DIM,
        cfg=PRIOR_SAMPLE,
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
    print(f"[ref] π(d): mean_d={mean_d:.4f}, std_d={std_d:.4f}")

    # 6) Evidence from real data
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
    print(
        f"[evidence] mu_d={mu_d:.4f}, tau_d={tau_d:.4f} | mu_z={mu_z:.3f}, tau_z={tau_z:.3f}"
    )

    # 7) PK guidance + sampling
    from quantbayes.stochax.vae.latent_diffusion.pk_guidance import InkEvidence

    evidence = InkEvidence(mean_d=mean_d, std_d=std_d, mu_z=mu_z, tau_z=tau_z)

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
    x_pk = decode_to_x01(vae_eval, z_pk)

    # ----------------------------
    # Visual 1: image grid comparison
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
    plt.title(
        f"Latent EDM + PK(ink) | λ={PK_CFG.guide_strength}, smax={PK_CFG.guide_sigma_max}"
    )
    plt.axis("off")
    plt.tight_layout()
    out_grid = paths.run_dir / "compare_grid.png"
    plt.savefig(out_grid, dpi=180)
    plt.show()
    print("[saved]", out_grid)

    # ----------------------------
    # Visual 2: 1D distributions in z-space
    # ----------------------------
    # Compute decoded ink d and standardized z_d
    def ink_z(x01_np: np.ndarray) -> np.ndarray:
        d = np.asarray(
            ink_fraction_01(
                jnp.asarray(x01_np), thr=PK_CFG.ink_thr, temp=PK_CFG.ink_temp
            )
        )
        z = (d - mean_d) / std_d
        return z

    # reference z_d from the same z_ref used to define π(d)
    x_ref = decode_to_x01(vae_eval, z_ref)
    z_ref_d = ink_z(np.asarray(x_ref))
    z_std_d = ink_z(np.asarray(x_std))
    z_prior_d = ink_z(np.asarray(x_prior))
    z_pk_d = ink_z(np.asarray(x_pk))

    # histogram
    lo = (
        float(min(z_ref_d.min(), z_prior_d.min(), z_pk_d.min(), mu_z - 4 * tau_z)) - 0.5
    )
    hi = (
        float(max(z_ref_d.max(), z_prior_d.max(), z_pk_d.max(), mu_z + 4 * tau_z)) + 0.5
    )
    grid = np.linspace(lo, hi, 600)

    plt.figure(figsize=(10, 4))
    plt.hist(z_ref_d, bins=60, density=True, alpha=0.25, label="π(z) (ref)")
    plt.hist(
        z_prior_d,
        bins=60,
        density=True,
        histtype="step",
        linewidth=2,
        label="samples (LatentEDM)",
    )
    plt.hist(
        z_pk_d,
        bins=60,
        density=True,
        histtype="step",
        linewidth=2,
        label="samples (PK)",
    )
    plt.hist(
        z_std_d,
        bins=60,
        density=True,
        histtype="step",
        linewidth=2,
        alpha=0.7,
        label="samples (N(0,1))",
    )
    plt.plot(
        grid,
        gaussian_pdf(grid, mu_z, tau_z),
        linewidth=2,
        label="target p(z) (Gaussian)",
    )
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
    # Visual 3: reference score check (learned vs finite-diff)
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
        prior_name=PRIOR_NAME,
        latent_dim=LATENT_DIM,
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
            std_normal=summarize(z_std_d),
            latent_edm=summarize(z_prior_d),
            pk=summarize(z_pk_d),
        ),
        artifacts=dict(
            compare_grid=str(out_grid),
            ink_z_hist=str(out_hist),
            ref_score_check=str(out_score),
            vae=str(paths.vae_path),
            z_cache=str(paths.z_cache),
            prior_ckpt=str(paths.prior_ckpt),
            ref_score=str(paths.ref_score_path),
        ),
    )
    _save_json(paths.summary_json, summary)
    print("[saved]", paths.summary_json)
    print("[done] artifacts:", paths.run_dir)


if __name__ == "__main__":
    main()
