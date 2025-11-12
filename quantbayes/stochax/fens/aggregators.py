# quantbayes/stochax/fens/aggregators.py
from __future__ import annotations
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from quantbayes.stochax.layers.specnorm import SpectralNorm

Array = jnp.ndarray
PRNG = jax.Array


class MLPConcatAgg(eqx.Module):
    """
    FENS "stacked generalizer" = shallow MLP on concatenated client logits.
    Input x has shape (n_clients, K). We flatten → (n_clients*K,) and map:
        h = ReLU(W1 x + b1), logits = W2 h + b2
    Returns (logits[K], state).
    """

    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    n_clients: int
    n_classes: int
    hidden: int

    def __init__(
        self,
        n_clients: int,
        n_classes: int,
        key: PRNG,
        hidden: int = 64,
        bias: bool = True,
    ):
        k1, k2 = jax.random.split(key, 2)
        self.n_clients = int(n_clients)
        self.n_classes = int(n_classes)
        self.hidden = int(hidden)
        self.l1 = eqx.nn.Linear(
            self.n_clients * self.n_classes, self.hidden, key=k1, use_bias=bias
        )
        self.l2 = eqx.nn.Linear(self.hidden, self.n_classes, key=k2, use_bias=bias)

    def __call__(self, x: Array, key: Optional[PRNG], state: None):
        v = jnp.reshape(x, (self.n_clients * self.n_classes,))
        h = jax.nn.relu(self.l1(v))
        logits = self.l2(h)
        return logits, state


class MLPConcatAgg_SN(eqx.Module):
    """
    FENS "stacked generalizer" = shallow MLP on concatenated client logits.
    Input x has shape (n_clients, K). We flatten → (n_clients*K,) and map:
        h = ReLU(W1 x + b1), logits = W2 h + b2
    Returns (logits[K], state).
    """

    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    n_clients: int
    n_classes: int
    hidden: int
    target: float

    def __init__(
        self,
        n_clients: int,
        n_classes: int,
        key: PRNG,
        hidden: int = 64,
        bias: bool = True,
        target: float = 1.0,
    ):
        k1, k2 = jax.random.split(key, 2)
        self.n_clients = int(n_clients)
        self.n_classes = int(n_classes)
        self.target = target
        self.hidden = int(hidden)
        self.l1 = SpectralNorm(
            eqx.nn.Linear(
                self.n_clients * self.n_classes, self.hidden, key=k1, use_bias=bias
            ),
            target=self.target,
            mode="force",
        )
        self.l2 = SpectralNorm(
            eqx.nn.Linear(self.hidden, self.n_classes, key=k2, use_bias=bias),
            mode="force",
            target=self.target,
        )

    def __call__(self, x: Array, key: Optional[PRNG], state: None):
        v = jnp.reshape(x, (self.n_clients * self.n_classes,))
        h = jax.nn.relu(self.l1(v))
        logits = self.l2(h)
        return logits, state


class PerClientClassWeightsAgg(eqx.Module):
    """
    Paper-faithful: f(z1,...,zM) = sum_i (Lambda[i] ⊙ z_i).
    Operates on client *logits*. No normalization; returns logits.
    """

    W: Array  # (n_clients, n_classes)

    def __init__(self, n_clients: int, n_classes: int, key: PRNG):
        # Initialize near an average combiner to avoid huge initial scales.
        self.W = jnp.full(
            (int(n_clients), int(n_classes)),
            1.0 / float(n_clients),
            dtype=jnp.float32,
        )

    def __call__(self, x: Array, key: Optional[PRNG], state: None):
        # x: (n_clients, n_classes) logits from clients
        logits = jnp.sum(self.W * x, axis=0)  # (n_classes,)
        return logits, state


class PerClientScalarWeightsAgg(eqx.Module):
    """
    f(z1,...,zM) = sum_i w_i * z_i  (same w_i across classes).
    Cheaper than per-class; used in ablations.
    """

    w: Array  # (n_clients,)

    def __init__(self, n_clients: int, key: PRNG):
        self.w = jnp.ones((int(n_clients),), dtype=jnp.float32) / float(n_clients)

    def __call__(self, x: Array, key: Optional[PRNG], state: None):
        # x: (n_clients, n_classes) logits
        logits = jnp.tensordot(self.w, x, axes=(0, 0))  # (n_classes,)
        return logits, state


class MeanLogitsAgg(eqx.Module):
    """Simple average of client logits."""

    def __call__(self, x: Array, key: Optional[PRNG], state: None):
        return jnp.mean(x, axis=0), state


class WeightedAvgByCountsAgg(eqx.Module):
    """
    Per-class weights proportional to client class counts (fixed, non-trainable).
    weights[i, c] = n_i,c / sum_j n_j,c  ; operates on logits.
    """

    weights: Array = eqx.field(static=True)  # freeze from training/optim updates

    def __init__(self, class_counts: Array):
        # class_counts: (n_clients, n_classes)
        counts = jnp.asarray(class_counts, dtype=jnp.float32)
        denom = jnp.clip(jnp.sum(counts, axis=0, keepdims=True), 1.0, jnp.inf)
        self.weights = counts / denom  # (n_clients, n_classes)

    def __call__(self, x: Array, key: Optional[PRNG], state: None):
        logits = jnp.sum(self.weights * x, axis=0)
        return logits, state


class DeepSetAgg(eqx.Module):
    """
    Permutation-invariant aggregator over client *logits*.
      ρ: K→d→d applied per client, mean pool over clients, then μ: d→d→K.
    """

    rho1: eqx.nn.Linear
    rho2: eqx.nn.Linear
    mu1: eqx.nn.Linear
    mu2: eqx.nn.Linear
    K: int
    d: int

    def __init__(self, K: int, hidden: int, key: PRNG, bias: bool = True):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.rho1 = eqx.nn.Linear(K, hidden, key=k1, use_bias=bias)
        self.rho2 = eqx.nn.Linear(hidden, hidden, key=k2, use_bias=bias)
        self.mu1 = eqx.nn.Linear(hidden, hidden, key=k3, use_bias=bias)
        self.mu2 = eqx.nn.Linear(hidden, K, key=k4, use_bias=bias)
        self.K = int(K)
        self.d = int(hidden)

    # ---- pieces exposed for potential composition ----
    def encode_rows(self, z_row: Array) -> Array:
        # z_row: (K,) client logits for one example
        h = jax.nn.relu(self.rho1(z_row))
        h = jax.nn.relu(self.rho2(h))
        return h  # (d,)

    def decode_pool(self, hbar: Array) -> Array:
        t = jax.nn.relu(self.mu1(hbar))
        return self.mu2(t)  # (K,)

    # ---- standard forward (DeepSet with mean pooling) ----
    def __call__(self, x: Array, key: Optional[PRNG], state: None):
        # x: (n_clients, K) logits
        H = jax.vmap(self.encode_rows, in_axes=0)(x)  # (n_clients, d)
        hbar = jnp.mean(H, axis=0)  # (d,)
        logits = self.decode_pool(hbar)  # (K,)
        return logits, state


def make_fens_aggregator(
    name: str,
    *,
    n_clients: int,
    n_classes: int,
    key: PRNG,
    hidden: int = 64,
    target: float = 1.0,
    class_counts: Optional[Array] = None,
):
    name = name.lower().strip()

    if name in {"mlp", "concat_mlp", "nn"}:
        return MLPConcatAgg(
            n_clients=n_clients, n_classes=n_classes, key=key, hidden=hidden
        )
    if name in {"mlp_sn", "concat_mlp_sn", "nn_sn"}:
        return MLPConcatAgg_SN(
            n_clients=n_clients,
            n_classes=n_classes,
            key=key,
            hidden=hidden,
            target=target,
        )

    if name in {"per_client_class", "pccw", "pccw_logits"}:
        return PerClientClassWeightsAgg(
            n_clients=n_clients, n_classes=n_classes, key=key
        )

    if name in {"per_client_scalar", "linear"}:
        return PerClientScalarWeightsAgg(n_clients=n_clients, key=key)

    if name in {"mean", "avg"}:
        return MeanLogitsAgg()

    if name in {"weighted_avg_counts", "wavg_counts"}:
        if class_counts is None:
            raise ValueError(
                "Provide class_counts (n_clients, n_classes) for WeightedAvgByCountsAgg."
            )
        return WeightedAvgByCountsAgg(class_counts=class_counts)

    if name in {"deepset", "ds", "set"}:
        return DeepSetAgg(K=n_classes, hidden=hidden, key=key)

    raise ValueError(f"Unknown FENS aggregator '{name}'")
