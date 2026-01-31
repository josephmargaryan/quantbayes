# quantbayes/bnn/lipschitz.py

from __future__ import annotations
from typing import Any, Iterable, Sequence, Optional

import jax
import jax.numpy as jnp
from numpyro import distributions as dist

# Optionally import your specific layers for isinstance checks
from quantbayes.bnn.layers.spectral_layers import RFFTCirculant1D, RFFTCirculant2D
from quantbayes.bnn.layers.layers import Linear

_LOG_DTYPE = jnp.float32


def _layer_sigma(obj: Any) -> Optional[jnp.ndarray]:
    """
    Try to read a per-layer operator norm σ(obj).

    We use a duck-typed `__operator_norm_hint__` first, and fall back
    to a few explicit layer types.
    """
    if hasattr(obj, "__operator_norm_hint__"):
        return jnp.asarray(obj.__operator_norm_hint__(), _LOG_DTYPE)

    if isinstance(obj, (RFFTCirculant1D, RFFTCirculant2D, Linear)):
        # should be covered by __operator_norm_hint__, but just in case
        return jnp.asarray(obj.__operator_norm_hint__(), _LOG_DTYPE)

    return None


def lipschitz_product_from_layers(
    layers: Sequence[Any],
    *,
    act_lipschitz: float = 1.0,
    return_log: bool = False,
) -> jnp.ndarray:
    """
    Simple sequential Lipschitz upper bound:
        L ≤ (∏_layers σ_l) · act_lipschitz^(#activations)

    Assumes all non-linearities between layers are ≤ `act_lipschitz`. If you
    use ReLU/tanh this is just 1.0. If you use SiLU or GELU you can pass ~1.1.
    """
    log_L = jnp.array(0.0, dtype=_LOG_DTYPE)

    for layer in layers:
        sig = _layer_sigma(layer)
        if sig is None:
            raise TypeError(
                f"Layer of type {type(layer).__name__} "
                "does not expose __operator_norm_hint__()."
            )
        sig = jnp.clip(sig, 1e-12, 1e12)
        log_L = log_L + jnp.log(sig)

        # 1-Lipschitz activations ⇒ skip; otherwise you could add
        # e.g. log_L += jnp.log(act_lipschitz) after each layer.

    if act_lipschitz != 1.0:
        # if you know exactly how many non-linearities there are, pass that and
        # multiply log_L by it; here we just assume one per layer as a default.
        log_L = log_L + len(layers) * jnp.log(jnp.asarray(act_lipschitz, _LOG_DTYPE))

    if return_log:
        return log_L

    max_log = jnp.log(jnp.asarray(jnp.finfo(_LOG_DTYPE).max, _LOG_DTYPE)) - 1.0
    log_L = jnp.minimum(log_L, max_log)
    return jnp.exp(log_L)


def linear_prior_lipschitz_bound(
    sigma: float,
    in_features: int,
    out_features: int,
    delta: float = 1e-6,
) -> float:
    m, n = in_features, out_features
    t = jnp.sqrt(2.0 * jnp.log(1.0 / delta))
    return float(sigma * (jnp.sqrt(m) + jnp.sqrt(n) + t))


def rfft1d_prior_lipschitz_bound(
    std: jnp.ndarray,  # shape (k_half,) = σ_k for each active freq
    delta: float = 1e-6,
) -> float:
    sigma_max = jnp.max(std)
    # account for Hermitian pair by using N = len(std)*2-2, but in the bound
    # this is logarithmic so we can just use len(std) as a safe underestimate.
    N = std.shape[0]
    L = 2.0 * sigma_max * jnp.sqrt(2.0 * jnp.log(4.0 * N / delta))
    return float(L)


def sebr_bound_linear(
    M: jnp.ndarray,
    A: jnp.ndarray,
    *,
    num_mc_eps: int = 8,
    key: jax.Array,
) -> jnp.ndarray:
    """
    One-layer SEBR bound term (inside the () of Eq. (16) in SEBR).
    """
    # 1) ||M||_2 via SVD
    M2 = M.reshape(M.shape[0], -1)
    sigma_M = jnp.linalg.svd(M2, compute_uv=False)[0]

    # 2) max row / col std norms
    row_norms = jnp.linalg.norm(A, axis=1)  # (m,)
    col_norms = jnp.linalg.norm(A, axis=0)  # (n,)
    rc_term = jnp.max(row_norms) + jnp.max(col_norms)

    # 3) E max_{i,j} |G_ij| ≈ MC
    m, n = A.shape
    keys = jax.random.split(key, num_mc_eps)

    def one_mc(k):
        eps = jax.random.normal(k, shape=(m, n), dtype=A.dtype)
        G = eps * A
        return jnp.max(jnp.abs(G))

    g_term = jnp.mean(jax.vmap(one_mc)(keys))

    # ignore the universal constant c, just absorb into λ
    return sigma_M + rc_term + g_term


def rfft2d_prior_lipschitz_bound(
    std2d: jnp.ndarray,  # (H_pad, W_half) = per-frequency std σ_{uv}
    C_in: int,
    C_out: int,
    *,
    delta: float = 1e-6,
    active_mask: Optional[jnp.ndarray] = None,
) -> float:
    """
    High-probability prior Lipschitz upper bound for a 2D spectral circulant layer.

    Assumes that for each frequency (u,v), the (C_out x C_in) block K(u,v)
    has i.i.d. complex Gaussian entries with per-entry std std2d[u,v].
    Then with prob ≥ 1-δ,

        ||T||_2 <= σ_max [ sqrt(C_out) + sqrt(C_in) + sqrt(2 log(2 N_eff / δ)) ],

    where N_eff is the number of active frequencies and σ_max is the maximum
    per-frequency std over those.
    """
    if active_mask is not None:
        std_eff = std2d[active_mask > 0]
    else:
        std_eff = std2d.reshape(-1)

    if std_eff.size == 0:
        return 0.0

    sigma_max = jnp.max(std_eff)
    N_eff = std_eff.size

    t = jnp.sqrt(2.0 * jnp.log(2.0 * N_eff / delta))
    L = sigma_max * (jnp.sqrt(C_out) + jnp.sqrt(C_in) + t)
    return float(L)


def sebr_loss_for_linears(
    linears: Sequence[Linear],
    *,
    guide_params: dict,
    lambda_sebr: float,
    key: jax.Array,
) -> jnp.ndarray:
    """
    Example: assuming each Linear in your guide has params
    {name}_loc and {name}_rho in `guide_params`.
    """
    total = jnp.array(0.0)

    k_seq = jax.random.split(key, len(linears))
    for lin, k in zip(linears, k_seq):
        # You must define how to map `lin.name` -> (loc, rho) in your guide
        loc = guide_params[f"{lin.name}_loc"]  # (in,out)
        rho = guide_params[f"{lin.name}_rho"]
        A = jax.nn.softplus(rho)
        bound = sebr_bound_linear(loc, A, key=k)
        total = total + 0.5 * bound**2

    return lambda_sebr * total


def simple_lip_penalty_per_sample(layers, tau=1e-4):
    sigmas = [l.__operator_norm_hint__() for l in layers]
    log_L = jnp.sum(jnp.log(jnp.clip(jnp.stack(sigmas), 1e-12, 1e12)))
    return tau * log_L


def spectral_prior(std_act, scale):
    # std_act is your frequency-dependent template, scale is ψ
    return dist.Normal(0.0, scale * std_act)


def sample_bnn_prior_functions(model_fn, X_M, psi, num_fns, key):
    """
    model_fn: function that, given X and psi, samples f(X) under the prior.
    You can implement it e.g. with numpyro.infer.Predictive(model_fn, num_samples=...),
    but I’d recommend a pure-JAX version that samples weights and applies
    your spectral layers directly.
    """
    keys = jax.random.split(key, num_fns)

    def one_sample(k):
        return model_fn(X_M, psi, k)  # (M, out_dim)

    return jax.vmap(one_sample)(keys)  # (num_fns, M, out_dim)


def critic_loss(phi, params_phi, f_gp, f_bnn, lam_gp, key):
    # f_gp, f_bnn: (B, M, C)
    def apply_phi(z):
        return phi.apply(params_phi, z).mean()  # e.g. sum or mean over M,C

    L = jnp.mean(jax.vmap(apply_phi)(f_gp)) - jnp.mean(jax.vmap(apply_phi)(f_bnn))

    # gradient penalty: interpolate in function space
    eps = jax.random.uniform(key, shape=(f_gp.shape[0], 1, 1), minval=0.0, maxval=1.0)
    f_hat = eps * f_bnn + (1.0 - eps) * f_gp

    def grad_norm(z):
        g = jax.grad(apply_phi)(z)
        return jnp.linalg.norm(g.reshape(-1))

    gn = jax.vmap(grad_norm)(f_hat)
    gp = jnp.mean((gn - 1.0) ** 2)
    return L + lam_gp * gp


def wasserstein_objective(phi, params_phi, f_gp, f_bnn):
    def apply_phi(z):
        return phi.apply(params_phi, z).mean()

    return jnp.mean(jax.vmap(apply_phi)(f_gp)) - jnp.mean(jax.vmap(apply_phi)(f_bnn))


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt

    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer.autoguide import AutoNormal

    from quantbayes import bnn
    from quantbayes.bnn.lipschitz import lipschitz_product_from_layers
    from quantbayes.bnn.wrapper.base import NumpyroClassifier

    # ---------------------------------------------------------------------
    # 1) Synthetic binary classification data
    # ---------------------------------------------------------------------
    key = jax.random.PRNGKey(0)
    N_train, N_test, D = 256, 256, 2
    key_x, key_y = jax.random.split(key)

    X_all = jax.random.normal(key_x, (N_train + N_test, D))
    w_true = jnp.array([1.0, -1.0])
    b_true = 0.2
    logits_true = X_all @ w_true + b_true
    probs_true = jax.nn.sigmoid(logits_true)
    y_all = jax.random.bernoulli(key_y, probs_true).astype(jnp.int32)

    X_train, X_test = X_all[:N_train], X_all[N_train:]
    y_train, y_test = y_all[:N_train], y_all[N_train:]

    # ---------------------------------------------------------------------
    # 2) Models with Lipschitz deterministics inside
    # ---------------------------------------------------------------------
    def bnn_model(X, y=None):
        N, D_ = X.shape

        spec1 = bnn.RFFTCirculant1D(in_features=D_, alpha=2.0, name="spec1")
        head = bnn.Linear(in_features=D_, out_features=1, name="head")

        h = spec1(X)
        h = jax.nn.tanh(h)
        out = head(h)
        logits = out.squeeze(-1)

        numpyro.deterministic("out", logits)

        with numpyro.plate("data", N):
            numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

        numpyro.deterministic("Lip_spec1", spec1.__operator_norm_hint__())
        numpyro.deterministic("Lip_head", head.__operator_norm_hint__())
        layers = [spec1, head]
        L_net = lipschitz_product_from_layers(layers, act_lipschitz=1.0)
        numpyro.deterministic("Lip_network", L_net)

    lambda_lip = 1e-2  # slightly stronger to see an effect

    def bnn_model_lip(X, y=None):
        N, D_ = X.shape

        spec1 = bnn.RFFTCirculant1D(in_features=D_, alpha=2.0, name="spec1")
        head = bnn.Linear(in_features=D_, out_features=1, name="head")

        h = spec1(X)
        h = jax.nn.tanh(h)
        out = head(h)
        logits = out.squeeze(-1)

        numpyro.deterministic("out", logits)

        with numpyro.plate("data", N):
            numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

        numpyro.deterministic("Lip_spec1", spec1.__operator_norm_hint__())
        numpyro.deterministic("Lip_head", head.__operator_norm_hint__())
        layers = [spec1, head]
        L_net = lipschitz_product_from_layers(layers, act_lipschitz=1.0)
        numpyro.deterministic("Lip_network", L_net)

        numpyro.factor("lip_reg", -lambda_lip * jnp.log(L_net + 1e-12))

    # ---------------------------------------------------------------------
    # 3) Baseline and Lip-reg classifiers with Lipschitz logging
    # ---------------------------------------------------------------------
    clf_base = NumpyroClassifier(
        model=bnn_model,
        method="svi",
        guide=AutoNormal(bnn_model),
        optimizer="adam",
        learning_rate=1e-2,
        num_steps=2000,
        n_posterior_samples=500,
        random_state=1,
        progress_bar=True,
        logits_site="out",
        log_lipschitz=True,
        lip_sites=("Lip_network", "Lip_spec1", "Lip_head"),
        lip_num_samples=1,
        lip_log_interval=10,
        lip_batch_size=1,
    )

    clf_lip = NumpyroClassifier(
        model=bnn_model_lip,
        method="svi",
        guide=AutoNormal(bnn_model_lip),
        optimizer="adam",
        learning_rate=1e-2,
        num_steps=2000,
        n_posterior_samples=500,
        random_state=2,
        progress_bar=True,
        logits_site="out",
        log_lipschitz=True,
        lip_sites=("Lip_network", "Lip_spec1", "Lip_head"),
        lip_num_samples=1,
        lip_log_interval=10,
        lip_batch_size=1,
    )

    clf_base.fit(X_train, y_train)
    clf_lip.fit(X_train, y_train)

    # ---------------------------------------------------------------------
    # 4) Lipschitz evolution (network)
    # ---------------------------------------------------------------------
    hist_base = clf_base.get_lipschitz_history()
    hist_lip = clf_lip.get_lipschitz_history()

    steps_base = np.array(hist_base["step"])
    Lnet_base = np.array(hist_base["sites"]["Lip_network"]["mean"])
    steps_lip = np.array(hist_lip["step"])
    Lnet_lip = np.array(hist_lip["sites"]["Lip_network"]["mean"])

    plt.figure(figsize=(6, 4))
    plt.plot(steps_base, Lnet_base, label="baseline")
    plt.plot(steps_lip, Lnet_lip, label="lip-reg")
    plt.xlabel("SVI step")
    plt.ylabel("network Lipschitz (mean)")
    plt.yscale("log")
    plt.title("Lipschitz evolution")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------
    # 5) Posterior Lipschitz distributions
    # ---------------------------------------------------------------------
    lip_samples_base = clf_base.sample_lipschitz(
        X_test[:1], sites=("Lip_network", "Lip_spec1", "Lip_head"), num_samples=500
    )
    lip_samples_lip = clf_lip.sample_lipschitz(
        X_test[:1], sites=("Lip_network", "Lip_spec1", "Lip_head"), num_samples=500
    )

    Lip_net_base = np.asarray(lip_samples_base["Lip_network"])
    Lip_net_lip = np.asarray(lip_samples_lip["Lip_network"])

    print("\n=== Posterior Lipschitz statistics (network) ===")
    print(f"Baseline: mean={Lip_net_base.mean():.3f}, std={Lip_net_base.std():.3f}")
    print(f"Lip-reg : mean={Lip_net_lip.mean():.3f}, std={Lip_net_lip.std():.3f}")

    # ---------------------------------------------------------------------
    # 6) Clean accuracy
    # ---------------------------------------------------------------------
    probs_base = clf_base.predict_proba(X_test)[:, 1]
    probs_lip = clf_lip.predict_proba(X_test)[:, 1]
    yhat_base = (probs_base > 0.5).astype(int)
    yhat_lip = (probs_lip > 0.5).astype(int)

    acc_base = float((yhat_base == np.asarray(y_test)).mean())
    acc_lip = float((yhat_lip == np.asarray(y_test)).mean())

    print("\n=== Clean test accuracy ===")
    print(f"Baseline: {acc_base:.3f}")
    print(f"Lip-reg : {acc_lip:.3f}")

    # ---------------------------------------------------------------------
    # 7) Adversarial accuracy (FGSM and PGD)
    # ---------------------------------------------------------------------
    eps = 0.1

    acc_fgsm_base = clf_base.evaluate_adversarial_accuracy(
        X_test, y_test, attack="fgsm", epsilon=eps, num_draws=1
    )
    acc_fgsm_lip = clf_lip.evaluate_adversarial_accuracy(
        X_test, y_test, attack="fgsm", epsilon=eps, num_draws=1
    )

    acc_pgd_base = clf_base.evaluate_adversarial_accuracy(
        X_test,
        y_test,
        attack="pgd",
        epsilon=eps,
        step_size=eps / 5.0,
        num_steps=10,
        num_draws=1,
    )
    acc_pgd_lip = clf_lip.evaluate_adversarial_accuracy(
        X_test,
        y_test,
        attack="pgd",
        epsilon=eps,
        step_size=eps / 5.0,
        num_steps=10,
        num_draws=1,
    )

    print("\n=== Adversarial accuracy (ε = {:.3f}) ===".format(eps))
    print(f"FGSM baseline: {acc_fgsm_base:.3f}")
    print(f"FGSM lip-reg : {acc_fgsm_lip:.3f}")
    print(f"PGD  baseline: {acc_pgd_base:.3f}")
    print(f"PGD  lip-reg : {acc_pgd_lip:.3f}")
