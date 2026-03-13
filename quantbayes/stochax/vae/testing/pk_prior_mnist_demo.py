# ============================================================
# VAE + PK Prior Demo (MNIST) — Production / Publication Baseline
#
# Shows:
#   - VAE sampling holes: z ~ N(0,I)
#   - PK-updated prior sampling: z ~ p*(z) using evidence from aggregate posterior q(z|x)
#
# Evidence:
#   - u = F(z); default F(z)=z (IdentityFeatureMap)
#   - evidence_score trained with DSM on u samples generated from q(z|x)
# Reference:
#   - analytic for IdentityFeatureMap: s_pi(u) = -u
#
# Visuals:
#   1) grid_compare.png             : decoded images N(0,1) vs PK prior
#   2) latent_scatter.png           : (z0,z1) for q(z|x), N(0,1), PK prior
#   3) latent_radius_hist.png       : ||z|| distribution (q vs N(0,1) vs PK)
#
# Optional:
#   - train-or-load a tiny MNIST classifier and report confidence/diversity on generated samples
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

from quantbayes.stochax.vae.components import ConvVAE
from quantbayes.stochax.vae.train_vae import TrainConfig, train_vae

from quantbayes.stochax.vae.pk import (
    IdentityFeatureMap,
    LatentScoreNet,
    ScoreNetConfig,
    ScoreDSMTrainConfig,
    train_score_from_vae_aggregate,
    PKLatentPrior,
    PKPriorConfig,
    AnnealedLangevinConfig,
    make_annealed_langevin_sampler,
)
from quantbayes.stochax.vae.pk.aggregate import collect_aggregate_latents


# ----------------------------
# Settings
# ----------------------------
SEED = 0
OUTROOT = Path("artifacts/vae_pk_mnist_pub")
OUTROOT.mkdir(parents=True, exist_ok=True)

# VAE
LATENT_DIM = 16
VAE_EPOCHS = 25
VAE_BATCH = 128

# Aggregate score training (DSM on u=z)
SCORE_STEPS = 20_000
SCORE_BATCH = 256

# Sampling
N_SAMPLES = 512
SHOW = 64  # perfect square
ALD_CFG = AnnealedLangevinConfig(
    n_sigmas=30,
    sigma_min=0.01,
    sigma_max=1.0,
    rho=7.0,
    steps_per_sigma=6,
    step_scale=0.08,
    final_denoise=True,
    max_norm=30.0,
)

# PK knobs (now supports sigma_min + weight_mode)
PK_CFG = PKPriorConfig(
    lambda_strength=1.0,
    sigma_min=0.2,  # NEW: stop late collapse by not guiding below this (tune)
    sigma_max=1.0,  # guide in [sigma_min, sigma_max]
    sigma_ref=1.0,
    sigma_weight_gamma=1.0,
    sigma_weight_mode="high",  # NEW: guide earlier (preserves variance)
    max_correction_norm=80.0,
)

# Latent plotting
N_LATENT_PLOT = 5000

# Optional classifier
TRAIN_CLASSIFIER = True


# ----------------------------
# Paths
# ----------------------------
@dataclass(frozen=True)
class Paths:
    run_dir: Path
    vae_path: Path
    score_path: Path
    z_agg_cache: Path
    clf_path: Path
    summary_json: Path


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


def make_paths() -> Paths:
    tag = (
        f"z{LATENT_DIM}"
        f"__lam{PK_CFG.lambda_strength}"
        f"__smin{PK_CFG.sigma_min}"
        f"__smax{PK_CFG.sigma_max}"
        f"__mode{PK_CFG.sigma_weight_mode}"
        f"__wg{PK_CFG.sigma_weight_gamma}"
    ).replace(".", "p")
    run_dir = OUTROOT / tag
    _ensure_dir(run_dir)
    return Paths(
        run_dir=run_dir,
        vae_path=run_dir / "vae.eqx",
        score_path=run_dir / "score_agg_z.eqx",
        z_agg_cache=run_dir / "z_agg_cache.npy",
        clf_path=run_dir / "tiny_clf.eqx",
        summary_json=run_dir / "summary.json",
    )


# ----------------------------
# Data
# ----------------------------
def load_mnist_nchw_float01():
    tr = tfds.load("mnist", split="train", as_supervised=True, batch_size=-1)
    te = tfds.load("mnist", split="test", as_supervised=True, batch_size=-1)
    (x_tr, y_tr) = tfds.as_numpy(tr)
    (x_te, y_te) = tfds.as_numpy(te)

    x_tr = x_tr.astype(np.float32) / 255.0
    x_te = x_te.astype(np.float32) / 255.0
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


# ----------------------------
# Optional classifier
# ----------------------------
class TinyMNISTClassifier(eqx.Module):
    c1: eqx.nn.Conv2d
    c2: eqx.nn.Conv2d
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear

    def __init__(self, *, key):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.c1 = eqx.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding="SAME", key=k1)
        self.c2 = eqx.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding="SAME", key=k2)
        self.l1 = eqx.nn.Linear(64 * 14 * 14, 256, key=k3)
        self.l2 = eqx.nn.Linear(256, 10, key=k4)

    def __call__(self, x):
        x = jax.vmap(self.c1)(x)
        x = jax.nn.relu(x)
        x = jax.vmap(self.c2)(x)
        x = jax.nn.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = jax.vmap(self.l1)(x)
        x = jax.nn.relu(x)
        x = jax.vmap(self.l2)(x)
        return x


def train_classifier(x_train, y_train, *, seed=0, epochs=3, batch=256):
    key = jr.PRNGKey(seed + 9999)
    model = TinyMNISTClassifier(key=key)

    tx = optax.adamw(1e-3, weight_decay=1e-4)
    opt_state = tx.init(eqx.filter(model, eqx.is_inexact_array))

    def iter_batches(k):
        n = x_train.shape[0]
        perm = jr.permutation(k, n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            yield x_train[idx], y_train[idx]

    @eqx.filter_jit
    def step(m, opt_state, xb, yb):
        def loss_fn(mm):
            logits = mm(xb)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, yb)
            return jnp.mean(loss)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        updates, opt_state = tx.update(
            grads, opt_state, eqx.filter(m, eqx.is_inexact_array)
        )
        m = eqx.apply_updates(m, updates)
        return m, opt_state, loss

    for e in range(epochs):
        key, sub = jr.split(key)
        run, ct = 0.0, 0
        for xb, yb in iter_batches(sub):
            model, opt_state, loss = step(model, opt_state, xb, yb)
            run += float(loss)
            ct += 1
        print(f"[clf] epoch {e+1}/{epochs} | loss={run/max(ct,1):.4f}")
    return model


def classifier_stats(clf, x):
    logits = clf(x)
    p = jax.nn.softmax(logits, axis=-1)
    conf = jnp.mean(jnp.max(p, axis=-1))
    preds = jnp.argmax(p, axis=-1)
    hist = jnp.bincount(preds, length=10) / preds.shape[0]
    ent = -jnp.sum(hist * jnp.log(hist + 1e-12))
    return float(conf), float(ent), np.asarray(hist)


# ----------------------------
# Main
# ----------------------------
def main():
    paths = make_paths()
    x_train, y_train, _, _ = load_mnist_nchw_float01()
    print("[MNIST]", x_train.shape, y_train.shape)
    print("[run_dir]", paths.run_dir)

    # ---- Train-or-load VAE ----
    vae_template = ConvVAE(
        image_size=28,
        channels=1,
        hidden_channels=64,
        latent_dim=LATENT_DIM,
        key=jr.PRNGKey(SEED),
    )
    if paths.vae_path.exists():
        vae = _load_eqx(paths.vae_path, vae_template)
        print("[VAE] loaded:", paths.vae_path)
    else:
        print("[VAE] training...")
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
        _save_eqx(paths.vae_path, vae)
        print("[VAE] saved:", paths.vae_path)

    vae_eval = eqx.tree_inference(vae, value=True)

    # ---- Cache a sample of aggregated posterior latents (for plots) ----
    if paths.z_agg_cache.exists():
        z_agg = jnp.asarray(np.load(paths.z_agg_cache))
        print("[z_agg] loaded cache:", z_agg.shape)
    else:
        z_agg = collect_aggregate_latents(
            vae_eval,
            x_train,
            key=jr.PRNGKey(SEED + 4242),
            batch_size=512,
            num_samples=N_LATENT_PLOT,
            use_mu=False,
        )
        np.save(paths.z_agg_cache, np.asarray(z_agg))
        print("[z_agg] cached:", z_agg.shape)

    # ---- Train-or-load evidence score on aggregate posterior ----
    F = IdentityFeatureMap(latent_dim=LATENT_DIM)
    score_cfg = ScoreNetConfig(dim=F.out_dim, hidden=256, depth=3, time_emb_dim=64)
    score_template = LatentScoreNet(score_cfg, key=jr.PRNGKey(SEED + 1234))

    score_train_cfg = ScoreDSMTrainConfig(
        steps=SCORE_STEPS,
        batch_size=SCORE_BATCH,
        lr=2e-4,
        weight_decay=1e-5,
        grad_clip_norm=1.0,
        print_every=500,
        seed=SEED + 10,
        sigma_min=0.01,
        sigma_max=1.0,
        sample="log_uniform",
        loss_weight="sigma2",
        save_path=paths.score_path,
        save_every=0,
    )

    score = train_score_from_vae_aggregate(
        score_template,
        vae_eval,
        x_train,
        F,
        score_train_cfg,
        key=jr.PRNGKey(SEED + 2026),
        use_mu=False,
        save_path=paths.score_path,
    )
    print("[score] ready:", paths.score_path)

    # ---- Build PK prior score ----
    pk_prior = PKLatentPrior(
        feature_map=F,
        evidence_score=score,
        reference_score=None,  # IdentityFeatureMap => analytic s_pi(u)=-u
        cfg=PK_CFG,
    )

    # ---- Sample baseline vs PK ----
    k1, k2 = jr.split(jr.PRNGKey(SEED + 3333), 2)

    # Baseline z~N(0,I)
    z_prior = jr.normal(k1, (N_SAMPLES, LATENT_DIM))
    x_prior = jax.nn.sigmoid(vae_eval.decoder(z_prior, rng=None, train=False))

    # PK sample with compiled ALD sampler (no recompiles)
    sampler = make_annealed_langevin_sampler(
        lambda log_sigma, z: pk_prior(log_sigma, z),
        shape=(N_SAMPLES, LATENT_DIM),
        cfg=ALD_CFG,
    )
    z_pk = sampler(k2)
    x_pk = jax.nn.sigmoid(vae_eval.decoder(z_pk, rng=None, train=False))

    # ---- Visual 1: image grids ----
    nrow = int(round(math.sqrt(SHOW)))
    assert nrow * nrow == SHOW
    g0 = make_grid(np.asarray(x_prior[:SHOW]), nrow)
    g1 = make_grid(np.asarray(x_pk[:SHOW]), nrow)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(g0, cmap="gray")
    plt.title("VAE samples: z~N(0,I)")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(g1, cmap="gray")
    plt.title("VAE samples: PK-updated prior")
    plt.axis("off")
    plt.tight_layout()
    out_grid = paths.run_dir / "grid_compare.png"
    plt.savefig(out_grid, dpi=180)
    plt.show()
    print("[saved]", out_grid)

    # ---- Visual 2: latent scatter (z0,z1) ----
    z0 = np.asarray(z_agg[:, :2])
    z1 = np.asarray(z_prior[:, :2])
    z2 = np.asarray(z_pk[:, :2])

    plt.figure(figsize=(12, 4))
    for i, (Z, title) in enumerate(
        [
            (z0, "Aggregate posterior q(z|x)"),
            (z1, "N(0,1) prior samples"),
            (z2, "PK-updated prior samples"),
        ],
        start=1,
    ):
        plt.subplot(1, 3, i)
        plt.scatter(Z[:, 0], Z[:, 1], s=2, alpha=0.3)
        plt.title(title)
        plt.xlabel("z0")
        plt.ylabel("z1")
    plt.tight_layout()
    out_scatter = paths.run_dir / "latent_scatter.png"
    plt.savefig(out_scatter, dpi=180)
    plt.show()
    print("[saved]", out_scatter)

    # ---- Visual 3: radius histogram ----
    def radius(Z):
        Z = np.asarray(Z)
        return np.sqrt(np.sum(Z * Z, axis=-1))

    r_agg = radius(z_agg)
    r_n01 = radius(z_prior)
    r_pk = radius(z_pk)

    plt.figure(figsize=(10, 4))
    bins = 60
    plt.hist(r_agg, bins=bins, density=True, alpha=0.25, label="q(z|x) radius")
    plt.hist(
        r_n01,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="N(0,1) radius",
    )
    plt.hist(
        r_pk,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="PK prior radius",
    )
    plt.title("Latent radius distribution")
    plt.xlabel("||z||")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    out_r = paths.run_dir / "latent_radius_hist.png"
    plt.savefig(out_r, dpi=180)
    plt.show()
    print("[saved]", out_r)

    # ---- Optional classifier metric ----
    clf = None
    if TRAIN_CLASSIFIER:
        if paths.clf_path.exists():
            clf_template = TinyMNISTClassifier(key=jr.PRNGKey(SEED + 9999))
            clf = _load_eqx(paths.clf_path, clf_template)
            print("[clf] loaded:", paths.clf_path)
        else:
            clf = train_classifier(x_train, y_train, seed=SEED, epochs=3, batch=256)
            _save_eqx(paths.clf_path, clf)
            print("[clf] saved:", paths.clf_path)

        conf0, ent0, hist0 = classifier_stats(clf, x_prior)
        conf1, ent1, hist1 = classifier_stats(clf, x_pk)

        print("\n[Classifier sanity on generated samples]")
        print(
            f"  N(0,1)  : mean max-prob={conf0:.3f}, label-entropy={ent0:.3f}, hist={hist0}"
        )
        print(
            f"  PK prior: mean max-prob={conf1:.3f}, label-entropy={ent1:.3f}, hist={hist1}"
        )

    summary = dict(
        seed=SEED,
        latent_dim=LATENT_DIM,
        pk_cfg={k: getattr(PK_CFG, k) for k in PK_CFG.__dataclass_fields__.keys()},
        ald_cfg={k: getattr(ALD_CFG, k) for k in ALD_CFG.__dataclass_fields__.keys()},
        artifacts=dict(
            grid=str(out_grid),
            scatter=str(out_scatter),
            radius_hist=str(out_r),
            vae=str(paths.vae_path),
            score=str(paths.score_path),
        ),
    )
    _save_json(paths.summary_json, summary)
    print("[saved]", paths.summary_json)
    print("[done] artifacts:", paths.run_dir)


if __name__ == "__main__":
    main()
