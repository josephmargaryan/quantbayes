import os
import math
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx

# --- your library imports ---
from quantbayes.ball_dp.rff import sample_rff_rbf, rff_transform
from quantbayes.ball_dp.heads.logreg_eqx import LogisticRegressorEqx

from quantbayes.stochax.utils.pk import (
    pk_weights_from_y_samples_1d,
    effective_sample_size,
    normal_logpdf,
)

# ----------------------------
# Config (edit freely)
# ----------------------------
SEED = 0
DATA_ROOT = os.path.expanduser("~/data")
OUT_DIR = "reports/pk_rff_mnist"
os.makedirs(OUT_DIR, exist_ok=True)

DIGITS = (0, 1)  # binary MNIST for simplicity
N_TRAIN = 6000
N_TEST = 2000

M_RFF = 2048  # number of random features
GAMMA = 2.0  # RBF gamma; tune if needed
L2_LAM = 1e-2  # strong convexity via L2
CLIP_OMEGA_NORM = (
    10.0  # optional: makes feature map Lipschitz bounded (already in your code)
)

ENSEMBLE_SIZE = 15  # "prior samples" over models (RFF seeds)
LBFGS_ITERS = 60  # optimization iterations per model

# PK evidence on Y = log sigma(w)
EVIDENCE_SHIFT_STD = (
    0.8  # shift mean of q to prefer smaller norms (in units of prior std)
)
EVIDENCE_STD_SCALE = 0.45  # std of q relative to prior std


# ----------------------------
# Data loading (torchvision -> numpy -> jax)
# ----------------------------
def load_mnist_binary(root, digits=(0, 1), n_train=6000, n_test=2000):
    import torch
    from torchvision.datasets import MNIST

    tr = MNIST(root=root, train=True, download=True)
    te = MNIST(root=root, train=False, download=True)

    Xtr = tr.data.numpy().astype(np.float32) / 255.0
    ytr = tr.targets.numpy().astype(np.int32)
    Xte = te.data.numpy().astype(np.float32) / 255.0
    yte = te.targets.numpy().astype(np.int32)

    mask_tr = np.isin(ytr, np.array(digits, dtype=np.int32))
    mask_te = np.isin(yte, np.array(digits, dtype=np.int32))
    Xtr, ytr = Xtr[mask_tr], ytr[mask_tr]
    Xte, yte = Xte[mask_te], yte[mask_te]

    # map digits -> {-1,+1}
    ytr_pm1 = np.where(ytr == digits[1], 1.0, -1.0).astype(np.float32)
    yte_pm1 = np.where(yte == digits[1], 1.0, -1.0).astype(np.float32)

    Xtr = Xtr.reshape(Xtr.shape[0], -1)
    Xte = Xte.reshape(Xte.shape[0], -1)

    # subsample
    Xtr, ytr_pm1 = Xtr[:n_train], ytr_pm1[:n_train]
    Xte, yte_pm1 = Xte[:n_test], yte_pm1[:n_test]
    return Xtr, ytr_pm1, Xte, yte_pm1


def standardize(Xtr, Xte, eps=1e-6):
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + eps
    return (Xtr - mu) / sd, (Xte - mu) / sd


# ----------------------------
# Strongly convex training: full-batch LBFGS on logistic regression
# ----------------------------
def train_logreg_lbfgs(Phi, y_pm1, lam, *, key, max_iters=60):
    Phi = jnp.asarray(Phi, dtype=jnp.float32)
    y = jnp.asarray(y_pm1, dtype=jnp.float32)

    model = LogisticRegressorEqx(d_in=Phi.shape[1], key=key, init_scale=1e-2)
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def f(p):
        m = eqx.combine(p, static)
        scores = jax.vmap(m)(Phi)  # (N,)
        loss = jnp.mean(jnp.logaddexp(0.0, -y * scores))
        reg = 0.5 * lam * (jnp.sum(m.w * m.w) + m.b * m.b)
        return loss + reg

    # L-BFGS + zoom line search (deterministic objective)
    linesearch = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=20,
        slope_rtol=1e-4,  # Armijo
        curv_rtol=0.9,  # Wolfe
        initial_guess_strategy="one",
    )
    solver = optax.lbfgs(memory_size=10, linesearch=linesearch)
    opt_state = solver.init(params)

    value_and_grad = optax.value_and_grad_from_state(f)

    value = None
    for _ in range(int(max_iters)):
        value, grad = value_and_grad(params, state=opt_state)
        updates, opt_state = solver.update(
            grad, opt_state, params, value=value, grad=grad, value_fn=f
        )
        params = optax.apply_updates(params, updates)

    trained = eqx.combine(params, static)
    return trained, float(value)


def accuracy_from_scores(scores, y_pm1):
    y = jnp.asarray(y_pm1, dtype=jnp.float32)
    preds = jnp.where(scores >= 0.0, 1.0, -1.0)
    return float(jnp.mean(preds == y))


def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


# ----------------------------
# Main experiment
# ----------------------------
def main():
    key = jr.PRNGKey(SEED)

    Xtr, ytr, Xte, yte = load_mnist_binary(
        DATA_ROOT, digits=DIGITS, n_train=N_TRAIN, n_test=N_TEST
    )
    Xtr, Xte = standardize(Xtr, Xte)

    Xtr_j = jnp.asarray(Xtr, dtype=jnp.float32)
    Xte_j = jnp.asarray(Xte, dtype=jnp.float32)
    ytr_j = jnp.asarray(ytr, dtype=jnp.float32)
    yte_j = jnp.asarray(yte, dtype=jnp.float32)

    # Train an ensemble via different RFF seeds (this is your empirical "prior" π over weights)
    models = []
    omegas = []
    phases = []
    losses = []
    accs = []
    sigmas = []

    for i in range(ENSEMBLE_SIZE):
        key, k_rff, k_init = jr.split(key, 3)

        omega, phase = sample_rff_rbf(
            k_rff,
            d_in=Xtr_j.shape[1],
            m=M_RFF,
            gamma=GAMMA,
            clip_omega_norm=CLIP_OMEGA_NORM,
            dtype=jnp.float32,
        )
        Phi_tr = rff_transform(Xtr_j, omega, phase)
        Phi_te = rff_transform(Xte_j, omega, phase)

        mdl, final_obj = train_logreg_lbfgs(
            Phi_tr, ytr_j, L2_LAM, key=k_init, max_iters=LBFGS_ITERS
        )

        scores_te = jax.vmap(mdl)(Phi_te)
        acc = accuracy_from_scores(scores_te, yte_j)

        # "spectral norm via SVD" (for a vector, this equals ||w||2)
        Wmat = mdl.w[None, :]  # (1, d)
        sigma = jnp.linalg.svd(Wmat, compute_uv=False)[0]

        models.append(mdl)
        omegas.append(omega)
        phases.append(phase)
        losses.append(final_obj)
        accs.append(acc)
        sigmas.append(float(sigma))

        print(
            f"[{i+1:02d}/{ENSEMBLE_SIZE}] acc={acc:.4f}  sigma={float(sigma):.4f}  obj={final_obj:.4f}"
        )

    accs_np = np.array(accs, dtype=np.float32)
    sigmas_np = np.array(sigmas, dtype=np.float32)

    # Evidence variable Y = log sigma
    y_samples = jnp.log(jnp.asarray(sigmas_np, dtype=jnp.float32) + 1e-12)

    # Define target evidence q(y): Normal centered left of the prior mean
    mu0 = jnp.mean(y_samples)
    s0 = jnp.std(y_samples) + 1e-6
    mu_q = mu0 - EVIDENCE_SHIFT_STD * s0
    s_q = EVIDENCE_STD_SCALE * s0

    def log_q(y):
        return normal_logpdf(y, mu_q, s_q)

    # PK weights: w_i ∝ q(y_i) / p_Y(y_i)
    w_pk, logw, logp_hat = pk_weights_from_y_samples_1d(
        y_samples,
        log_q,
        bandwidth=None,  # Silverman
        leave_one_out=True,
        max_logw=60.0,
    )

    w_pk_np = np.array(w_pk, dtype=np.float64)
    ess = float(effective_sample_size(w_pk))

    # Baseline (uniform) vs PK-weighted summary
    acc_uniform = float(accs_np.mean())
    acc_weighted_avg = float((w_pk_np * accs_np).sum())

    # Weighted ensemble prediction (mixture of Bernoulli probs)
    # p(x)=Σ w_i σ(f_i(x)) using each model's own RFF features.
    probs = 0.0
    for wi, mdl, omega, phase in zip(w_pk_np, models, omegas, phases):
        Phi_te = rff_transform(Xte_j, omega, phase)
        scores_te = jax.vmap(mdl)(Phi_te)
        probs = probs + wi * sigmoid(scores_te)
    preds = jnp.where(probs >= 0.5, 1.0, -1.0)
    acc_pk_ensemble = float(jnp.mean(preds == yte_j))

    # A simple "spectral clipping" baseline (optional): clip each model to target sigma=exp(mu_q)
    sigma_target = float(jnp.exp(mu_q))
    acc_clip = []
    for mdl, omega, phase in zip(models, omegas, phases):
        W = mdl.w
        s = float(jnp.linalg.norm(W) + 1e-12)
        scale = min(1.0, sigma_target / s)
        mdl_clip = eqx.tree_at(lambda m: m.w, mdl, mdl.w * scale)
        Phi_te = rff_transform(Xte_j, omega, phase)
        scores_te = jax.vmap(mdl_clip)(Phi_te)
        acc_clip.append(accuracy_from_scores(scores_te, yte_j))
    acc_clip_mean = float(np.mean(acc_clip))

    print("\n=== Summary ===")
    print(f"Uniform avg acc:        {acc_uniform:.4f}")
    print(f"PK weighted avg acc:    {acc_weighted_avg:.4f}")
    print(f"PK ensemble acc:        {acc_pk_ensemble:.4f}")
    print(f"Clip-to-target mean acc:{acc_clip_mean:.4f}")
    print(f"ESS (PK weights):       {ess:.2f} / {ENSEMBLE_SIZE}")
    print(f"Prior mean sigma:       {float(sigmas_np.mean()):.4f}")
    print(f"PK-weighted mean sigma: {float((w_pk_np * sigmas_np).sum()):.4f}")
    print(f"Target sigma (exp(mu_q)):{sigma_target:.4f}")

    # ----------------------------
    # Plots
    # ----------------------------
    # 1) Histogram of log sigma: prior vs PK-weighted + target q curve
    y_np = np.array(y_samples, dtype=np.float64)
    grid = np.linspace(y_np.min() - 0.5, y_np.max() + 0.5, 400)
    q_pdf = np.exp(
        np.array(
            normal_logpdf(jnp.asarray(grid, jnp.float32), mu_q, s_q), dtype=np.float64
        )
    )

    # KDE curve for pY (just for visualization)
    # (reuse the same KDE estimator indirectly via logp at grid)
    from quantbayes.stochax.utils.pk import (
        gaussian_kde_logpdf_1d,
        silverman_bandwidth_1d,
    )

    bw = float(silverman_bandwidth_1d(y_samples))
    p_pdf = np.exp(
        np.array(
            gaussian_kde_logpdf_1d(
                jnp.asarray(grid, jnp.float32), y_samples, jnp.asarray(bw, jnp.float32)
            ),
            dtype=np.float64,
        )
    )

    plt.figure(figsize=(7.5, 4.2))
    bins = 12
    plt.hist(y_np, bins=bins, density=True, alpha=0.55, label="prior p_Y (empirical)")
    plt.hist(
        y_np,
        bins=bins,
        density=True,
        weights=w_pk_np,
        alpha=0.55,
        label="PK-updated (weighted)",
    )
    plt.plot(grid, p_pdf, linewidth=2.0, label="KDE(p_Y)")
    plt.plot(grid, q_pdf, linewidth=2.0, label="target q")
    plt.title("Evidence space: Y = log spectral norm")
    plt.xlabel("y = log sigma(w)")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "log_sigma_pk_hist.png"), dpi=160)

    # 2) Scatter: sigma vs accuracy, sized by PK weights
    plt.figure(figsize=(7.5, 4.2))
    sizes = 200.0 * (w_pk_np / (w_pk_np.max() + 1e-12) + 0.05)
    plt.scatter(sigmas_np, accs_np, s=sizes, alpha=0.8)
    plt.title("Models in the ensemble (size ∝ PK weight)")
    plt.xlabel("sigma(w) (spectral norm via SVD)")
    plt.ylabel("test accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sigma_vs_acc_scatter.png"), dpi=160)

    # 3) Show the PK weights themselves
    order = np.argsort(-w_pk_np)
    plt.figure(figsize=(7.5, 3.2))
    plt.bar(np.arange(len(w_pk_np)), w_pk_np[order])
    plt.title(f"PK weights (sorted) — ESS={ess:.2f}/{ENSEMBLE_SIZE}")
    plt.xlabel("model rank")
    plt.ylabel("weight")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pk_weights_sorted.png"), dpi=160)

    print(f"\nSaved plots to: {OUT_DIR}")
    print("  - log_sigma_pk_hist.png")
    print("  - sigma_vs_acc_scatter.png")
    print("  - pk_weights_sorted.png")


if __name__ == "__main__":
    main()
