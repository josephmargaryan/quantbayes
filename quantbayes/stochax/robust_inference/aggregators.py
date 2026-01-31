# quantbayes/stochax/robust_inference/aggregators.py
from __future__ import annotations
from typing import Any, Optional
import equinox as eqx
import jax
import jax.numpy as jnp

from quantbayes.stochax.robust_inference.probits import cwtm
from quantbayes.stochax.layers.specnorm import SpectralNorm

Array = jnp.ndarray
PRNG = jax.Array


def _safe_log(u: Array, eps: float) -> Array:
    """Stable log for nonnegative vectors on the simplex."""
    return jnp.log(jnp.maximum(u, eps))


def _cwtm_axis0(H: Array, f: int) -> Array:
    """
    Coordinate-wise trimmed mean along axis 0.
    H: (n, d) → returns (d,)
    Drops f smallest and f largest per feature.
    """
    n = H.shape[0]
    assert 2 * f < n, f"CWTM requires 2f < n; got f={f}, n={n}"
    idx = jnp.argsort(H, axis=0)
    Hs = jnp.take_along_axis(H, idx, axis=0)  # sorted per column
    Ht = Hs[f : n - f, :]
    return jnp.mean(Ht, axis=0)


# -------------------- Static baselines (paper) -------------------- #


class MeanAgg(eqx.Module):
    """Mean over clients (per class), returns log-probs as logits. No params."""

    K: int
    eps: float = 1e-12

    def __init__(self, K: int, eps: float = 1e-12):
        self.K = int(K)
        self.eps = float(eps)

    def __call__(self, x: Array, key: Optional[PRNG], state: Any):
        u = jnp.mean(x, axis=0)  # (K,)
        logits = _safe_log(u, self.eps)
        return logits, state


class CWTMAgg(eqx.Module):
    """Coordinate-wise trimmed mean across clients, then log. No trainable params."""

    f: int
    eps: float = 1e-12

    def __init__(self, f: int, eps: float = 1e-12):
        self.f = int(f)
        self.eps = float(eps)

    def __call__(self, x: Array, key: Optional[PRNG], state: Any):
        n = x.shape[0]
        assert 2 * self.f < n, f"CWTM requires 2f < n; got f={self.f}, n={n}"
        u = cwtm(x, self.f)  # (K,)
        logits = _safe_log(u, self.eps)
        return logits, state


class CWMedAgg(eqx.Module):
    """Coordinate-wise median across clients, then log. No trainable params."""

    eps: float = 1e-12

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def __call__(self, x: Array, key: Optional[PRNG], state: Any):
        u = jnp.median(x, axis=0)  # (K,)
        u = jnp.clip(u, 1e-12, 1.0)
        u = u / (jnp.sum(u) + 1e-12)
        logits = _safe_log(u, self.eps)
        return logits, state


# -------------------- Linear & DeepSet (paper) -------------------- #


class LinearAgg(eqx.Module):
    """Order-sensitive baseline: flatten (n*K) → K logits."""

    lin: eqx.nn.Linear
    n: int
    K: int

    def __init__(self, n_clients: int, K: int, key: PRNG, bias: bool = True):
        self.n = int(n_clients)
        self.K = int(K)
        self.lin = eqx.nn.Linear(self.n * self.K, self.K, key=key, use_bias=bias)

    def __call__(self, x: Array, key: Optional[PRNG], state: Any):
        v = jnp.reshape(x, (self.n * self.K,))
        logits = self.lin(v)  # (K,)
        return logits, state


class DeepSetAgg(eqx.Module):
    """
    Permutation-invariant aggregator:
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

    # ---- pieces exposed for inference-time CWTM composition ----
    def encode_rows(self, p_row: Array) -> Array:
        h = jax.nn.relu(self.rho1(p_row))
        h = jax.nn.relu(self.rho2(h))
        return h  # (d,)

    def decode_pool(self, hbar: Array) -> Array:
        t = jax.nn.relu(self.mu1(hbar))
        return self.mu2(t)  # (K,)

    # ---- standard forward (DeepSet with mean pooling) ----
    def __call__(self, x: Array, key: Optional[PRNG], state: Any):
        H = jax.vmap(self.encode_rows, in_axes=0)(x)  # (n, d)
        hbar = jnp.mean(H, axis=0)  # (d,)
        logits = self.decode_pool(hbar)  # (K,)
        return logits, state


class DeepSetAgg_SN(eqx.Module):
    """
    Permutation-invariant aggregator:
      ρ: K→d→d applied per client, mean pool over clients, then μ: d→d→K.
    """

    # You can leave these type hints as eqx.nn.Linear if you like;
    # at runtime they will actually hold SpectralNorm-wrapped linears.
    rho1: eqx.nn.Linear
    rho2: eqx.nn.Linear
    mu1: eqx.nn.Linear
    mu2: eqx.nn.Linear
    K: int
    d: int
    target: float

    def __init__(
        self, K: int, hidden: int, key: PRNG, bias: bool = True, target: float = 1.0
    ):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.target = float(target)
        self.rho1 = SpectralNorm(
            eqx.nn.Linear(K, hidden, key=k1, use_bias=bias),
            target=self.target,
            mode="force",
        )
        self.rho2 = SpectralNorm(
            eqx.nn.Linear(hidden, hidden, key=k2, use_bias=bias),
            target=self.target,
            mode="force",
        )
        self.mu1 = SpectralNorm(
            eqx.nn.Linear(hidden, hidden, key=k3, use_bias=bias),
            target=self.target,
            mode="force",
        )
        self.mu2 = SpectralNorm(
            eqx.nn.Linear(hidden, K, key=k4, use_bias=bias),
            target=self.target,
            mode="force",
        )
        self.K = int(K)
        self.d = int(hidden)

    # ---- pieces exposed for inference-time CWTM composition ----
    # Keep the signature EXACTLY like the non-SN version so DeepSetTM can vmap it.
    # Internally, we still call the SpectralNorm-wrapped layers with a dummy state.
    def encode_rows(self, p_row: Array) -> Array:
        # SpectralNorm.__call__(x, *, state) -> (y, state)
        h, _ = self.rho1(p_row, state=None)
        h = jax.nn.relu(h)
        h, _ = self.rho2(h, state=None)
        h = jax.nn.relu(h)
        return h  # (d,)

    def decode_pool(self, hbar: Array) -> Array:
        t, _ = self.mu1(hbar, state=None)
        t = jax.nn.relu(t)
        logits, _ = self.mu2(t, state=None)
        return logits  # (K,)

    # ---- standard forward (DeepSet with mean pooling) ----
    def __call__(self, x: Array, key: Optional[PRNG], state: Any):
        # This still matches the non-SN version structurally:
        # same arguments, same return type (logits, state).
        H = jax.vmap(self.encode_rows, in_axes=0)(x)  # (n, d)
        hbar = jnp.mean(H, axis=0)  # (d,)
        logits = self.decode_pool(hbar)  # (K,)
        return logits, state


class DeepSetCWTMAgg(eqx.Module):
    """
    Robust DeepSet (Section 4 & Eq. (10) at inference): ρ per client →
    CWTM pool across ρ-embeddings → μ.
    """

    rho1: eqx.nn.Linear
    rho2: eqx.nn.Linear
    mu1: eqx.nn.Linear
    mu2: eqx.nn.Linear
    K: int
    d: int
    f: int

    def __init__(self, K: int, hidden: int, f: int, key: PRNG, bias: bool = True):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.rho1 = eqx.nn.Linear(K, hidden, key=k1, use_bias=bias)
        self.rho2 = eqx.nn.Linear(hidden, hidden, key=k2, use_bias=bias)
        self.mu1 = eqx.nn.Linear(hidden, hidden, key=k3, use_bias=bias)
        self.mu2 = eqx.nn.Linear(hidden, K, key=k4, use_bias=bias)
        self.K = int(K)
        self.d = int(hidden)
        self.f = int(f)

    def _enc_one(self, p_row: Array) -> Array:
        h = jax.nn.relu(self.rho1(p_row))
        h = jax.nn.relu(self.rho2(h))
        return h

    def __call__(self, x: Array, key: Optional[PRNG], state: Any):
        n = x.shape[0]
        assert 2 * self.f < n, f"CWTM requires 2f < n; got f={self.f}, n={n}"
        H = jax.vmap(self._enc_one, in_axes=0)(x)  # (n, d)
        hbar = _cwtm_axis0(H, self.f)  # (d,)
        t = jax.nn.relu(self.mu1(hbar))
        logits = self.mu2(t)  # (K,)
        return logits, state


class DeepSetCWTMAgg_SN(eqx.Module):
    """
    Robust DeepSet (Section 4 & Eq. (10) at inference): ρ per client →
    CWTM pool across ρ-embeddings → μ.
    """

    rho1: eqx.nn.Linear
    rho2: eqx.nn.Linear
    mu1: eqx.nn.Linear
    mu2: eqx.nn.Linear
    K: int
    d: int
    f: int
    target: float

    def __init__(
        self,
        K: int,
        hidden: int,
        f: int,
        key: PRNG,
        bias: bool = True,
        target: float = 1.0,
    ):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.target = float(target)
        self.rho1 = SpectralNorm(
            eqx.nn.Linear(K, hidden, key=k1, use_bias=bias),
            target=self.target,
            mode="force",
        )
        self.rho2 = SpectralNorm(
            eqx.nn.Linear(hidden, hidden, key=k2, use_bias=bias),
            target=self.target,
            mode="force",
        )
        self.mu1 = SpectralNorm(
            eqx.nn.Linear(hidden, hidden, key=k3, use_bias=bias),
            target=self.target,
            mode="force",
        )
        self.mu2 = SpectralNorm(
            eqx.nn.Linear(hidden, K, key=k4, use_bias=bias),
            target=self.target,
            mode="force",
        )
        self.K = int(K)
        self.d = int(hidden)
        self.f = int(f)

    # Stateless helper, like the non-SN version; used with vmap.
    def _enc_one(self, p_row: Array) -> Array:
        h, _ = self.rho1(p_row, state=None)
        h = jax.nn.relu(h)
        h, _ = self.rho2(h, state=None)
        h = jax.nn.relu(h)
        return h

    def __call__(self, x: Array, key: Optional[PRNG], state: Any):
        n = x.shape[0]
        assert 2 * self.f < n, f"CWTM requires 2f < n; got f={self.f}, n={n}"
        H = jax.vmap(self._enc_one, in_axes=0)(x)  # (n, d)
        hbar = _cwtm_axis0(H, self.f)  # (d,)

        t, _ = self.mu1(hbar, state=None)
        t = jax.nn.relu(t)
        logits, _ = self.mu2(t, state=None)  # (K,)

        return logits, state


class DeepSetTM(eqx.Module):
    """
    Inference-only wrapper: CWTM across ρ-embeddings, then μ.
    Train the base DeepSet with mean pooling; wrap at eval time.
    """

    base: eqx.Module  # DeepSetAgg
    f: int

    def __init__(self, base: eqx.Module, f: int):
        assert hasattr(base, "encode_rows") and hasattr(base, "decode_pool")
        self.base = base
        self.f = int(f)

    def __call__(self, x: Array, key: Optional[PRNG], state: Any):
        n = x.shape[0]
        assert 2 * self.f < n, f"CWTM requires 2f < n; got f={self.f}, n={n}"
        H = jax.vmap(self.base.encode_rows, in_axes=0)(x)  # (n, d)
        idx = jnp.argsort(H, axis=0)
        Hs = jnp.take_along_axis(H, idx, axis=0)[self.f : n - self.f, :]
        hbar_tm = jnp.mean(Hs, axis=0)
        logits = self.base.decode_pool(hbar_tm)
        return logits, state


# -------------------- Factory -------------------- #


def make_aggregator(
    name: str,
    *,
    n_clients: int,
    K: int,
    key: PRNG,
    hidden: int = 64,
    f: int = 1,
    bias: bool = True,
    target: float = 1.0,
):
    """
    Returns an Equinox Module with signature: (x: (n,K), key, state) -> (logits[K], state).

    NOTE: For DeepSetTM (inference-only composition), instantiate DeepSetAgg for training
    and wrap it with DeepSetTM at evaluation time.
    """
    name = name.lower()
    if name == "mean":
        return MeanAgg(K=K)
    if name == "cwtm":
        return CWTMAgg(f=f)
    if name == "cwmed":
        return CWMedAgg()
    if name == "linear":
        return LinearAgg(n_clients=n_clients, K=K, key=key, bias=bias)
    if name == "deepset":
        return DeepSetAgg(K=K, hidden=hidden, key=key, bias=bias)
    if name == "deepset_sn":
        return DeepSetAgg_SN(K=K, hidden=hidden, key=key, bias=bias, target=target)
    if name == "deepset_cwtm":
        return DeepSetCWTMAgg(K=K, hidden=hidden, f=f, key=key, bias=bias)
    if name == "deepset_cwtm_sn":
        return DeepSetCWTMAgg_SN(
            K=K, hidden=hidden, f=f, key=key, bias=bias, target=target
        )
    raise ValueError(f"Unknown aggregator '{name}'")
