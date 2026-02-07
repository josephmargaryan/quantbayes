# quantbayes/stochax/vae/testing/pk_prior_mnist_demo.py
# ============================================================
# VAE + PK Prior Demo (MNIST)
#
# Shows: VAE sampling "holes" (z~N(0,I)) vs PK-updated prior sampling
# where evidence p(F(z)) comes from aggregate posterior q(z|x).
#
# Default feature map: F(z)=z (identity) -> strongest hole fix.
#
# Artifacts:
#   artifacts/vae_pk_mnist/{grid_prior.png, grid_pk.png, grid_compare.png}
# Optional:
#   trains a small MNIST classifier and reports confidence/diversity on samples
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
    sample_annealed_langevin,
)

# ----------------------------
# Settings
# ----------------------------
OUTDIR = Path("artifacts/vae_pk_mnist")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 0

# VAE
LATENT_DIM = 16
EPOCHS = 25
BATCH = 128

# Score prior training
SCORE_STEPS = 20_000
SCORE_BATCH = 256

# Sampling
N_SAMPLES = 256
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

# PK knobs (λ + gating + sigma-weight)
PK_CFG = PKPriorConfig(
    lambda_strength=1.0,  # set >1 for “stronger pull”
    sigma_max=0.50,  # apply correction only when sigma <= 0.50 (late refinement)
    sigma_ref=1.0,
    sigma_weight_gamma=0.5,  # “middle ground” dial
    max_correction_norm=80.0,
)

TRAIN_CLASSIFIER = True  # set False if you only want visuals


def load_mnist_nchw_float01():
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


# ------------------------------------------------------------
# Optional: tiny classifier for quantitative “wow”
# ------------------------------------------------------------
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
        # x: (B,1,28,28)
        x = jax.vmap(self.c1)(x)
        x = jax.nn.relu(x)
        x = jax.vmap(self.c2)(x)
        x = jax.nn.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = jax.vmap(self.l1)(x)
        x = jax.nn.relu(x)
        x = jax.vmap(self.l2)(x)
        return x  # logits


def train_classifier(x_train, y_train, *, seed=0, epochs=3, batch=256):
    key = jr.PRNGKey(seed + 9999)
    model = TinyMNISTClassifier(key=key)

    tx = optax.adamw(1e-3, weight_decay=1e-4)
    params0 = eqx.filter(model, eqx.is_inexact_array)
    opt_state = tx.init(params0)

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
        params = eqx.filter(m, eqx.is_inexact_array)
        updates, opt_state = tx.update(grads, opt_state, params)
        m = eqx.apply_updates(m, updates)
        return m, opt_state, loss

    for e in range(epochs):
        key, sub = jr.split(key)
        run = 0.0
        ct = 0
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
    # diversity proxy: entropy of predicted label histogram
    hist = jnp.bincount(preds, length=10) / preds.shape[0]
    ent = -jnp.sum(hist * jnp.log(hist + 1e-12))
    return float(conf), float(ent), np.asarray(hist)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    x_train, y_train, x_test, y_test = load_mnist_nchw_float01()
    print("[MNIST]", x_train.shape, y_train.shape)

    # ---- Train VAE (Bernoulli is usually best for MNIST) ----
    key = jr.PRNGKey(SEED)
    vae = ConvVAE(
        image_size=28, channels=1, hidden_channels=64, latent_dim=LATENT_DIM, key=key
    )

    vae_cfg = TrainConfig(
        epochs=EPOCHS,
        batch_size=BATCH,
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

    # ---- Train evidence score on aggregated posterior ----
    # Feature map F(z) = z
    F = IdentityFeatureMap(latent_dim=LATENT_DIM)

    score_cfg = ScoreNetConfig(dim=F.out_dim, hidden=256, depth=3, time_emb_dim=64)
    score = LatentScoreNet(score_cfg, key=jr.PRNGKey(SEED + 1234))

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
        save_path=OUTDIR / "score_agg_z.eqx",
        save_every=0,
    )

    score = train_score_from_vae_aggregate(
        score,
        vae,
        x_train,
        F,
        score_train_cfg,
        key=jr.PRNGKey(SEED + 2026),
        use_mu=False,
        save_path=OUTDIR / "score_agg_z.eqx",
    )

    # ---- Build PK prior score ----
    pk_prior = PKLatentPrior(
        feature_map=F,
        evidence_score=score,
        reference_score=None,  # IdentityFeatureMap provides analytic s_pi(u)=-u
        cfg=PK_CFG,
    )

    # ---- Sample baseline vs PK ----
    # Baseline: z ~ N(0,I)
    k1, k2 = jr.split(jr.PRNGKey(SEED + 3333), 2)
    z_prior = jr.normal(k1, (N_SAMPLES, LATENT_DIM))
    logits_prior = vae.decoder(z_prior, rng=None, train=False)
    x_prior = jax.nn.sigmoid(logits_prior)  # (B,1,28,28)

    # PK: annealed Langevin with PK score
    def score_fn(log_sigma, z):
        return pk_prior(log_sigma, z)

    z_pk = sample_annealed_langevin(
        score_fn, key=k2, shape=(N_SAMPLES, LATENT_DIM), cfg=ALD_CFG
    )
    logits_pk = vae.decoder(z_pk, rng=None, train=False)
    x_pk = jax.nn.sigmoid(logits_pk)

    # ---- Save grids ----
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
    plt.title("VAE samples: PK prior (score-guided)")
    plt.axis("off")
    plt.tight_layout()
    out = OUTDIR / "grid_compare.png"
    plt.savefig(out, dpi=180)
    plt.show()
    print("[saved]", out)

    # ---- Optional classifier metric ----
    if TRAIN_CLASSIFIER:
        clf = train_classifier(x_train, y_train, seed=SEED, epochs=3, batch=256)
        conf0, ent0, hist0 = classifier_stats(clf, x_prior)
        conf1, ent1, hist1 = classifier_stats(clf, x_pk)
        print("\n[Classifier sanity on generated samples]")
        print(
            f"  baseline prior: mean max-prob={conf0:.3f}, label-entropy={ent0:.3f}, hist={hist0}"
        )
        print(
            f"  PK prior      : mean max-prob={conf1:.3f}, label-entropy={ent1:.3f}, hist={hist1}"
        )

    print("\nDone. Artifacts in:", OUTDIR)


if __name__ == "__main__":
    main()
