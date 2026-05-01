# ============================================================
# PK MNIST INK: "PERFECT" EXPERIMENT (FIRST CHAPTER ONLY)
#
# Self-contained test:
#   - Loads MNIST
#   - Trains/loads an EDM prior (checkpointed) automatically
#   - Estimates prior pushforward π(d) for the ink observable
#   - Trains/loads 1D reference score s_pi(z) via DSM
#   - Optionally calibrates PK guidance OR uses fixed (λ, sigma_max)
#   - Compares none vs evidence-only vs PK
#
# Switch models:
#   Edit MODEL_NAME and the build_diffusion_model() switchboard.
#   (By default, each MODEL_NAME gets its own artifact folder + checkpoints.)
# ============================================================

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
import optax

# ---- model(s)
from quantbayes.stochax.diffusion.models.mixer_2d import Mixer2d

# ---- EDM / PK utilities
from quantbayes.stochax.diffusion.edm import edm_precond_scalars
from quantbayes.stochax.diffusion.pk import (
    InkFractionObservable,
    ScoreNet1DConfig,
    train_or_load_score_net_dsm,
    EDMHeunConfig,
    make_edm_heun_gaussian_pk_sampler,
    make_image_grid,
    PKCalibrationGrids,
    PKCalibrationWeights,
    calibrate_pk_gaussian_1d,
    save_csv,
    gaussian_pdf,
    w1_to_gaussian,
)

# ---- train-or-load EDM prior (THIS is the key fix)
from quantbayes.stochax.diffusion.pk.training_edm import (
    EDMTrainConfig,
    train_or_load_edm_unconditional,
)

# ----------------------------
# USER CONTROLS
# ----------------------------
SEED = 0

# ----------------------------
# MODEL SWITCHBOARD
# ----------------------------
MODEL_NAME = "mixer2d"  # <- change this and update build_diffusion_model() below

# Put each model in its own artifact tree automatically
ARTIFACT_ROOT = Path("artifacts/pk_mnist_ink_only")
OUTDIR = ARTIFACT_ROOT / MODEL_NAME
RUN_DIR = OUTDIR / "exp_runs_v3"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# checkpoints / score net
DIFF_CKPT_DIR = OUTDIR / "diffusion_ckpt"
SCORE_PATH = OUTDIR / "score_pi_z_ink.eqx"

# Evidence (real MNIST)
TARGET_DIGIT_FOR_EVIDENCE = 3
FAT_QUANTILE = 0.80
EVIDENCE_SHARPEN = 0.60  # "first chapter" tag sh060

# Observable
INK_THRESH = 0.35
INK_TEMP = 0.08

# EDM training (prior)
PRIOR_NUM_STEPS = 40000  # increase for better samples
PRIOR_BATCH_SIZE = 128
PRIOR_PRINT_EVERY = 500
PRIOR_CKPT_EVERY = 5000
PRIOR_KEEP_LAST = 3

# (Optional) quick debug: train on a subset
PRIOR_TRAIN_MAX_N = None  # e.g. 20000 for quick debug, or None for full train set

# Diffusion sampler / EDM params
SIGMA_DATA = 0.5
SAMPLE_STEPS = 40
SIGMA_MIN_SAMPLE = 0.002
SIGMA_MAX_SAMPLE = 80.0

# Guidance experiment controls
DO_CALIBRATION = True  # set False to use FIXED_* below

# Fixed guidance (used if DO_CALIBRATION=False)
FIXED_W_GAMMA = 0.5
FIXED_GUIDE_STRENGTH = 8.0  # λ
FIXED_GUIDE_SIGMA_MAX = 1.5  # indicator 𝟙[σ <= 1.5]

# Calibration grids (used if DO_CALIBRATION=True)
W_GAMMA_GRID = [0.0, 0.5, 1.0]
GUIDE_STRENGTH_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
GUIDE_SIGMA_MAX_GRID = [1.0, 1.5, 2.0, 3.0, 5.0]

ALPHA_CLIP = 0.25
BETA_W1 = 0.10
MAX_GUIDE_NORM = 10.0

# Sample sizes
N_REF = 4096
N_CALIB = 1024
N_FINAL = 2048
NUM_SHOW = 64  # must be perfect square


# ----------------------------
# Helpers
# ----------------------------
def load_mnist_nchw_scaled():
    train = tfds.load("mnist", split="train", as_supervised=True, batch_size=-1)
    test = tfds.load("mnist", split="test", as_supervised=True, batch_size=-1)
    (x_tr, y_tr) = tfds.as_numpy(train)
    (x_te, y_te) = tfds.as_numpy(test)

    x_tr = x_tr.astype(np.float32) / 255.0
    x_te = x_te.astype(np.float32) / 255.0

    # (N,28,28,1) -> (N,1,28,28)
    if x_tr.ndim == 4 and x_tr.shape[-1] == 1:
        x_tr = np.transpose(x_tr, (0, 3, 1, 2))
        x_te = np.transpose(x_te, (0, 3, 1, 2))

    # scale to [-1,1]
    x_tr = x_tr * 2.0 - 1.0
    x_te = x_te * 2.0 - 1.0

    y_tr = y_tr.astype(np.int32)
    y_te = y_te.astype(np.int32)

    return jnp.asarray(x_tr), jnp.asarray(y_tr), jnp.asarray(x_te), jnp.asarray(y_te)


class EDMNet(eqx.Module):
    """
    Generic wrapper that adapts your model to the EDM trainer.

    The EDM trainer calls:
        model(log_sigma, x_in, key=..., train=True)

    Many of your models accept (t, y, *, key=None) and ignore `train`.
    This wrapper tries to forward `train` if the base model supports it.
    """

    net: eqx.Module

    def __call__(self, log_sigma, x, *, key=None, train=False, **kwargs):
        # Attempt to forward train if supported; otherwise fall back.
        try:
            return self.net(log_sigma, x, key=key, train=train, **kwargs)
        except TypeError:
            return self.net(log_sigma, x, key=key, **kwargs)


def build_diffusion_model():
    """
    >>> EASY MODEL SWAP <<<
    Add cases here for other architectures. Keep the returned module shape/signature
    consistent across runs if you want to reuse the same checkpoint directory.

    Tip: since OUTDIR is already model-specific (OUTDIR/MODEL_NAME), you can freely
    switch MODEL_NAME without colliding checkpoints.
    """
    key = jr.PRNGKey(SEED + 2)

    name = MODEL_NAME.lower().replace("-", "").replace("_", "").strip()

    if name in ("mixer2d", "mixer"):
        base = Mixer2d(
            img_size=(1, 28, 28),
            patch_size=4,
            hidden_size=96,
            mix_patch_size=512,
            mix_hidden_size=512,
            num_blocks=4,
            t1=1.0,  # unused in Mixer2d but kept for API compat
            key=key,
        )
        return EDMNet(base)

    # مثال: add another model
    # elif name in ("unet2d", "unet"):
    #     from quantbayes.stochax.diffusion.models.unet_2d import UNet2d
    #     base = UNet2d(..., key=key)
    #     return EDMNet(base)

    raise ValueError(
        f"Unknown MODEL_NAME={MODEL_NAME!r}. "
        "Edit build_diffusion_model() to add your architecture."
    )


def get_or_train_ema_eval_model(x_train: jnp.ndarray):
    """
    Train EDM prior if no checkpoint exists; otherwise load/resume.
    Returns EMA model in inference mode.
    """
    DIFF_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    train_cfg = EDMTrainConfig(
        lr=2e-4,
        weight_decay=1e-4,
        batch_size=PRIOR_BATCH_SIZE,
        num_steps=PRIOR_NUM_STEPS,
        ema_decay=0.999,
        grad_clip_norm=1.0,
        print_every=PRIOR_PRINT_EVERY,
        checkpoint_every=PRIOR_CKPT_EVERY,
        keep_last=PRIOR_KEEP_LAST,
        sigma_data=SIGMA_DATA,
        sigma_min_train=SIGMA_MIN_SAMPLE,
        sigma_max_train=SIGMA_MAX_SAMPLE,
        seed=SEED,
    )

    # Optionally limit training set for quick debugging
    if PRIOR_TRAIN_MAX_N is not None:
        x_train_use = x_train[: int(PRIOR_TRAIN_MAX_N)]
    else:
        x_train_use = x_train

    _, ema_model = train_or_load_edm_unconditional(
        ckpt_dir=DIFF_CKPT_DIR,
        build_model_fn=build_diffusion_model,
        dataset=x_train_use,
        cfg=train_cfg,
    )

    ema_eval = eqx.tree_inference(ema_model, value=True)
    print(f"[diffusion] EMA ready from {DIFF_CKPT_DIR}")
    return ema_eval


def make_denoise_fn(ema_eval):
    def denoise_fn(log_sigma, x):
        sigma = jnp.exp(log_sigma)
        c_in, _, _ = edm_precond_scalars(sigma, SIGMA_DATA)
        return ema_eval(log_sigma, x * c_in, key=None, train=False)

    return denoise_fn


def main():
    tag = f"sh{int(round(100 * EVIDENCE_SHARPEN)):03d}"
    subdir = RUN_DIR / tag
    subdir.mkdir(parents=True, exist_ok=True)

    # Load MNIST
    x_train, y_train, x_test, y_test = load_mnist_nchw_scaled()
    print("[data] MNIST:", x_train.shape, y_train.shape)

    obs = InkFractionObservable(thr=INK_THRESH, temp=INK_TEMP)

    # ----------------------------
    # Step 1: evidence stats from real MNIST (class-k ink)
    # ----------------------------
    mask_k = y_train == TARGET_DIGIT_FOR_EVIDENCE
    x_k = x_train[mask_k]
    x_k_sub = x_k[:5000] if x_k.shape[0] > 5000 else x_k

    ink_k = obs.value(x_k_sub)
    mu_d_data = float(jnp.mean(ink_k))
    tau_d_data = float(jnp.std(ink_k) + 1e-6)
    mu_d = float(jnp.quantile(ink_k, FAT_QUANTILE))
    tau_d = float(max(1e-4, tau_d_data * float(EVIDENCE_SHARPEN)))

    print(
        f"[data] class-{TARGET_DIGIT_FOR_EVIDENCE} ink: mean={mu_d_data:.4f}, std={tau_d_data:.4f} | "
        f"fat mu_d(q={FAT_QUANTILE})={mu_d:.4f} | tau_d={tau_d:.4f} (sharpen={EVIDENCE_SHARPEN})"
    )

    # ----------------------------
    # Step 2: train-or-load diffusion EMA and define denoise_fn
    # ----------------------------
    ema_eval = get_or_train_ema_eval_model(x_train)
    denoise_fn = make_denoise_fn(ema_eval)

    sample_cfg = EDMHeunConfig(
        steps=SAMPLE_STEPS,
        sigma_min=SIGMA_MIN_SAMPLE,
        sigma_max=SIGMA_MAX_SAMPLE,
        rho=7.0,
        sigma_data=SIGMA_DATA,
    )

    # ----------------------------
    # Step 3: estimate prior pushforward π(d)
    # ----------------------------
    sampler_none_ref = make_edm_heun_gaussian_pk_sampler(
        denoise_fn,
        observable=obs,
        score_net_z=lambda z: jnp.zeros_like(z),
        sample_shape=(1, 28, 28),
        cfg=sample_cfg,
        num_samples=N_REF,
        mode="none",
    )

    print(f"\n[prior] Sampling {N_REF} prior images to estimate π(d)...")
    x_ref, _ = sampler_none_ref(
        jr.PRNGKey(SEED + 4242),
        jnp.array(0.0, jnp.float32),  # mean_d (dummy)
        jnp.array(1.0, jnp.float32),  # std_d  (dummy)
        jnp.array(0.0, jnp.float32),  # mu_z   (dummy)
        jnp.array(1.0, jnp.float32),  # tau_z  (dummy)
        jnp.array(0.0, jnp.float32),  # guide_strength
        jnp.array(0.0, jnp.float32),  # guide_sigma_max
        jnp.array(1.0, jnp.float32),  # max_guide_norm
        jnp.array(0.0, jnp.float32),  # w_gamma
    )

    d_ref = obs.value(jnp.clip(x_ref, -1.0, 1.0))
    mean_d = float(jnp.mean(d_ref))
    std_d = float(jnp.std(d_ref) + 1e-6)
    z_ref = (np.asarray(d_ref) - mean_d) / std_d

    print(f"[prior] π(d): mean_d={mean_d:.4f}, std_d={std_d:.4f}")

    # Evidence in z-space
    mu_z = float((mu_d - mean_d) / std_d)
    tau_z = float(tau_d / std_d)
    print(f"[target] z-space: mu_z={mu_z:.3f}, tau_z={tau_z:.3f}")

    # ----------------------------
    # Step 4: load/train score net s_pi(z) via DSM (loads if file exists)
    # ----------------------------
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

    z_ref_jax = jnp.asarray(z_ref, dtype=jnp.float32).reshape(-1, 1)
    score_net = train_or_load_score_net_dsm(z_ref_jax, SCORE_PATH, cfg=score_cfg)
    print(f"[score] s_pi(z) ready at: {SCORE_PATH}")

    # ----------------------------
    # Step 5: build samplers
    # ----------------------------
    def coarse_value_fn(x):
        return obs.value(jnp.clip(x, -1.0, 1.0))

    sampler_pk_calib = make_edm_heun_gaussian_pk_sampler(
        denoise_fn,
        observable=obs,
        score_net_z=score_net,
        sample_shape=(1, 28, 28),
        cfg=sample_cfg,
        num_samples=N_CALIB,
        mode="pk",
    )

    sampler_none_final = make_edm_heun_gaussian_pk_sampler(
        denoise_fn,
        observable=obs,
        score_net_z=score_net,
        sample_shape=(1, 28, 28),
        cfg=sample_cfg,
        num_samples=N_FINAL,
        mode="none",
    )
    sampler_evd_final = make_edm_heun_gaussian_pk_sampler(
        denoise_fn,
        observable=obs,
        score_net_z=score_net,
        sample_shape=(1, 28, 28),
        cfg=sample_cfg,
        num_samples=N_FINAL,
        mode="evidence",
    )
    sampler_pk_final = make_edm_heun_gaussian_pk_sampler(
        denoise_fn,
        observable=obs,
        score_net_z=score_net,
        sample_shape=(1, 28, 28),
        cfg=sample_cfg,
        num_samples=N_FINAL,
        mode="pk",
    )

    # ----------------------------
    # Step 6: choose guidance params (calibrate OR fixed)
    # ----------------------------
    if DO_CALIBRATION:
        grids = PKCalibrationGrids(
            w_gamma_grid=W_GAMMA_GRID,
            guide_strength_grid=GUIDE_STRENGTH_GRID,
            guide_sigma_max_grid=GUIDE_SIGMA_MAX_GRID,
        )
        weights = PKCalibrationWeights(alpha_clip=ALPHA_CLIP, beta_w1=BETA_W1)

        print("\n[calib] Sweeping PK hyperparams...")
        best, rows = calibrate_pk_gaussian_1d(
            sampler_pk_calib,
            coarse_value_fn,
            mean_d=mean_d,
            std_d=std_d,
            mu_z=mu_z,
            tau_z=tau_z,
            max_guide_norm=MAX_GUIDE_NORM,
            grids=grids,
            weights=weights,
            seed=SEED,
        )

        csv_path = subdir / "calibration_sweep_pk.csv"
        save_csv(csv_path, rows)
        print(f"[saved] {csv_path}")

        print(
            "[best PK calib] "
            f"γ={best['w_gamma']} | gs={best['guide_strength']} | smax={best['guide_sigma_max']} || "
            f"mean_z={best['mean_z']:.3f}, std_z={best['std_z']:.3f}, "
            f"W1={best['w1']:.4f}, clip={best['clip_rate']:.3f}, loss={best['calib_loss']:.4f}"
        )

        w_best = float(best["w_gamma"])
        gs_best = float(best["guide_strength"])
        smax_best = float(best["guide_sigma_max"])
    else:
        w_best = float(FIXED_W_GAMMA)
        gs_best = float(FIXED_GUIDE_STRENGTH)
        smax_best = float(FIXED_GUIDE_SIGMA_MAX)
        print(
            "\n[fixed] Using fixed guidance: "
            f"γ={w_best} | λ={gs_best} | sigma_max={smax_best}"
        )

    # ----------------------------
    # Step 7: final sampling (none / evidence / pk)
    # ----------------------------
    print("\n[final] Sampling none / evidence / PK ...")
    k_base = jr.PRNGKey(SEED + 123456)
    k_none = jr.fold_in(k_base, 1)
    k_evd = jr.fold_in(k_base, 2)
    k_pk = jr.fold_in(k_base, 3)

    mean_d_a = jnp.array(mean_d, jnp.float32)
    std_d_a = jnp.array(std_d, jnp.float32)
    mu_z_a = jnp.array(mu_z, jnp.float32)
    tau_z_a = jnp.array(tau_z, jnp.float32)
    max_norm_a = jnp.array(MAX_GUIDE_NORM, jnp.float32)
    w_best_a = jnp.array(w_best, jnp.float32)

    x_none, clip_none = sampler_none_final(
        k_none,
        mean_d_a,
        std_d_a,
        mu_z_a,
        tau_z_a,
        jnp.array(0.0, jnp.float32),
        jnp.array(0.0, jnp.float32),
        max_norm_a,
        w_best_a,
    )
    x_evd, clip_evd = sampler_evd_final(
        k_evd,
        mean_d_a,
        std_d_a,
        mu_z_a,
        tau_z_a,
        jnp.array(gs_best, jnp.float32),
        jnp.array(smax_best, jnp.float32),
        max_norm_a,
        w_best_a,
    )
    x_pk, clip_pk = sampler_pk_final(
        k_pk,
        mean_d_a,
        std_d_a,
        mu_z_a,
        tau_z_a,
        jnp.array(gs_best, jnp.float32),
        jnp.array(smax_best, jnp.float32),
        max_norm_a,
        w_best_a,
    )

    d_none = coarse_value_fn(x_none)
    d_evd = coarse_value_fn(x_evd)
    d_pk = coarse_value_fn(x_pk)

    z_none = (np.asarray(d_none) - mean_d) / std_d
    z_evd = (np.asarray(d_evd) - mean_d) / std_d
    z_pk = (np.asarray(d_pk) - mean_d) / std_d

    def summarize(z):
        m = float(np.mean(z))
        s = float(np.std(z) + 1e-12)
        w1 = float(w1_to_gaussian(z, mu=mu_z, tau=tau_z, seed=SEED + 7777))
        return m, s, w1

    m0, s0, w10 = summarize(z_none)
    m1, s1, w11 = summarize(z_evd)
    m2, s2, w12 = summarize(z_pk)

    print("[z summaries]")
    print(f" target: mu_z={mu_z:.3f}, tau_z={tau_z:.3f}")
    print(
        f"  none : mean={m0:.3f}, std={s0:.3f}, W1={w10:.4f}, clip={float(np.asarray(clip_none)):.3f}"
    )
    print(
        f"  evd  : mean={m1:.3f}, std={s1:.3f}, W1={w11:.4f}, clip={float(np.asarray(clip_evd)):.3f}"
    )
    print(
        f"  pk   : mean={m2:.3f}, std={s2:.3f}, W1={w12:.4f}, clip={float(np.asarray(clip_pk)):.3f}"
    )

    # ----------------------------
    # Save summary
    # ----------------------------
    summary_path = subdir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"model_name={MODEL_NAME}\n")
        f.write(f"chapter={tag}\n")
        f.write(f"digit={TARGET_DIGIT_FOR_EVIDENCE}\n")
        f.write(f"fat_quantile={FAT_QUANTILE}\n")
        f.write(f"evidence_sharpen={EVIDENCE_SHARPEN}\n")
        f.write(f"mu_d={mu_d}\n")
        f.write(f"tau_d={tau_d}\n")
        f.write(f"mu_z={mu_z}\n")
        f.write(f"tau_z={tau_z}\n")
        f.write(f"prior_mean_d={mean_d}\n")
        f.write(f"prior_std_d={std_d}\n")
        f.write(f"best_w_gamma={w_best}\n")
        f.write(f"best_guide_strength={gs_best}\n")
        f.write(f"best_guide_sigma_max={smax_best}\n")
        f.write(
            f"none_mean_z={m0}\nnone_std_z={s0}\nnone_w1={w10}\nclip_none={float(np.asarray(clip_none))}\n"
        )
        f.write(
            f"evd_mean_z={m1}\nev_d_std_z={s1}\nev_d_w1={w11}\nclip_evd={float(np.asarray(clip_evd))}\n"
        )
        f.write(
            f"pk_mean_z={m2}\npk_std_z={s2}\npk_w1={w12}\nclip_pk={float(np.asarray(clip_pk))}\n"
        )
    print(f"[saved] {summary_path}")

    # ----------------------------
    # Plots: grids + histogram
    # ----------------------------
    nrow = int(round(math.sqrt(NUM_SHOW)))
    assert nrow * nrow == NUM_SHOW

    g_none = make_image_grid(np.asarray(x_none[:NUM_SHOW]), nrow=nrow)
    g_evd = make_image_grid(np.asarray(x_evd[:NUM_SHOW]), nrow=nrow)
    g_pk = make_image_grid(np.asarray(x_pk[:NUM_SHOW]), nrow=nrow)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(g_none, cmap="gray")
    plt.title("Unconditional prior π(x)")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(g_evd, cmap="gray")
    plt.title(f"Evidence-only | γ={w_best}, λ={gs_best}, smax={smax_best}")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(g_pk, cmap="gray")
    plt.title(f"PK | γ={w_best}, λ={gs_best}, smax={smax_best}")
    plt.axis("off")
    plt.tight_layout()
    grid_path = subdir / "compare_grids.png"
    plt.savefig(grid_path, dpi=180)
    plt.show()
    print(f"[saved] {grid_path}")

    plt.figure(figsize=(10, 4))
    bins = 60
    plt.hist(
        z_ref, bins=bins, density=True, alpha=0.25, label="π(z) (prior pushforward)"
    )
    plt.hist(
        z_none,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="samples (none)",
    )
    plt.hist(
        z_evd,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="samples (evidence)",
    )
    plt.hist(
        z_pk,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="samples (PK)",
    )

    grid = np.linspace(
        min(z_ref.min(), z_pk.min()) - 1, max(z_ref.max(), z_pk.max()) + 1, 500
    )
    plt.plot(
        grid,
        gaussian_pdf(grid, mu_z, tau_z),
        linewidth=2,
        label="target p(z) (Gaussian)",
    )
    plt.title(f"{MODEL_NAME} | {tag}: ink fraction distribution in z-space")
    plt.xlabel("z")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    hist_path = subdir / "compare_hist_z.png"
    plt.savefig(hist_path, dpi=180)
    plt.show()
    print(f"[saved] {hist_path}")

    print("\nAll artifacts saved under:", subdir)


if __name__ == "__main__":
    main()
