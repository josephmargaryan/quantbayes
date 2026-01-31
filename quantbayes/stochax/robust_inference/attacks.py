# quantbayes/stochax/robust_inference/attacks.py
from __future__ import annotations
import jax, jax.numpy as jnp, jax.random as jr

Array = jnp.ndarray


def runnerup_attack(P: Array, y_true: int, f: int, key) -> Array:
    """
    One-hot to runner-up class of the *global mean* probit; flips f random clients.
    """
    n, K = P.shape
    Pbar = P.mean(axis=0)
    top2 = jnp.argsort(Pbar)[-2:]
    alt = int(top2[0]) if int(top2[-1]) == y_true else int(top2[-1])
    alt = alt if alt != y_true else (y_true + 1) % K
    target = jnp.zeros((K,), P.dtype).at[alt].set(1.0)
    idx = jr.permutation(key, n)[:f]
    return P.at[idx].set(target[None, :])


def cwtm_aware_attack(P: Array, f: int, eps: float = 1e-3) -> Array:
    """
    Choose a probit u whose coordinates lie inside the central trimmed band so it survives CWTM.
    Deterministic for reproducibility.
    """
    n, K = P.shape
    central = []
    for k in range(K):
        vals = jnp.sort(P[:, k])
        c = vals[f]  # inside kept region [f, n-f)
        central.append(c)
    u = jnp.array(central)
    u = jnp.clip(u + eps, 1e-8, 1.0)
    u = u / (u.sum() + 1e-12)
    idx = jnp.arange(n)[:f]
    return P.at[idx].set(u[None, :])


def sia_blackbox(P: Array, y_true: int, f: int, key) -> Array:
    """
    Strongest Inverted Attack (black-box): each selected adversary sets a one-hot
    vector to its *own* second-most-probable class not equal to y_true.
    """
    n, K = P.shape
    idx = jr.permutation(key, n)[:f]

    def one_row(pi):
        order = jnp.argsort(pi)
        top, second = order[-1], order[-2]
        alt = jnp.where(top == y_true, second, top)
        alt = jnp.where(alt == y_true, (y_true + 1) % K, alt)
        return jnp.eye(K, dtype=P.dtype)[alt]

    P_adv = jax.vmap(one_row)(P[idx])
    return P.at[idx].set(P_adv)


def sia_whitebox(P: Array, y_true: int, f: int, agg, key) -> Array:
    """
    White-box SIA: select f clients; set them to the runner-up class chosen
    from the aggregator’s logits (global view) excluding y_true.
    """
    logits, _ = agg(P, None, None)
    order = jnp.argsort(logits)
    top, second = int(order[-1]), int(order[-2])
    alt = second if top == y_true else top
    n, K = P.shape
    idx = jr.permutation(key, n)[:f]
    target = jnp.eye(K, dtype=P.dtype)[alt]
    return P.at[idx].set(target[None, :])


def loss_max_attack(P: Array, f: int, agg) -> Array:
    """
    LMA (white-box): set adversary rows to one-hot of least likely class under the aggregator.
    """
    logits, _ = agg(P, None, None)
    tgt = int(jnp.argmin(logits))
    n, K = P.shape
    idx = jnp.arange(n)[:f]
    target = jnp.eye(K, dtype=P.dtype)[tgt]
    return P.at[idx].set(target[None, :])
