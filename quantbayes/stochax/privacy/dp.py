# quantbayes/stochax/privacy/dp.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jax.scipy.special import gammaln

Array = jnp.ndarray
Pytree = Any

# Use a single dtype to avoid x64 warnings in default JAX installs.
DT = jnp.float32


# =========================
# Utilities (vector norms)
# =========================


def _per_example_l2_norms(grad_batched: Pytree) -> Array:
    leaves = jax.tree_util.tree_leaves(grad_batched)
    parts = [
        jnp.sum(jnp.square(g), axis=tuple(range(1, g.ndim))) for g in leaves
    ]  # (B,)
    total_sq = parts[0]
    for s in parts[1:]:
        total_sq = total_sq + s
    return jnp.sqrt(total_sq + 1e-12)  # (B,)


def _scale_like_batch(g: Array, scale: Array) -> Array:
    return g * scale.reshape((scale.shape[0],) + (1,) * (g.ndim - 1))


def _per_example_clip(grad_batched: Pytree, C: float) -> Pytree:
    norms = _per_example_l2_norms(grad_batched)  # (B,)
    scale = jnp.minimum(1.0, C / (norms + 1e-12))
    return jax.tree_util.tree_map(lambda g: _scale_like_batch(g, scale), grad_batched)


def _sum_over_batch(grad_batched: Pytree) -> Pytree:
    return jax.tree_util.tree_map(lambda g: jnp.sum(g, axis=0), grad_batched)


def _tree_add(a: Pytree, b: Pytree) -> Pytree:
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _tree_scale(a: Pytree, s: float | Array) -> Pytree:
    return jax.tree_util.tree_map(lambda x: x * s, a)


def rdp_epsilon_for_sgm(
    q: float,
    sigma: float,
    steps: int,
    delta: float,
    orders: Sequence[int] = tuple(list(range(2, 65)) + [80, 96, 128, 256, 512]),
) -> float:
    """Cumulative ε for Poisson-sampled Gaussian mechanism after `steps` DP steps."""
    rdp = _rdp_total(float(q), float(sigma), int(steps), orders)
    return _eps_from_rdp(rdp, orders, float(delta))


# ===============================================
# RDP accountant for Sampled Gaussian Mechanism
# (Poisson subsampling; integer Renyi orders)
# ===============================================


def _log_binom(n, k) -> Array:
    """log C(n,k) implemented in float; n,k may be Python ints or JAX arrays."""
    n = jnp.asarray(n, DT)
    k = jnp.asarray(k, DT)
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


def _compute_log_a_int(q: float, sigma: float, alpha: int) -> Array:
    """
    log A_alpha for Poisson-sampled Gaussian mechanism (integer alpha >= 2):
      A_alpha = sum_{k=0}^alpha C(alpha,k) q^k (1-q)^{alpha-k} exp(k(k-1) / (2 sigma^2))
    All computations are tracer-safe (no Python int() on tracers).
    """
    q = jnp.asarray(q, DT)
    s2 = jnp.asarray(sigma, DT) ** 2
    alpha_f = jnp.asarray(alpha, DT)
    ks = jnp.arange(0, alpha + 1, dtype=DT)  # [0, 1, ..., alpha]
    logq = jnp.log(jnp.clip(q, 1e-300, 1.0))
    log1mq = jnp.log(jnp.clip(1.0 - q, 1e-300, 1.0))

    def term(k):
        # k is DT; no int cast anywhere
        logw = _log_binom(alpha_f, k) + k * logq + (alpha_f - k) * log1mq
        loge = (k * (k - 1.0)) / (2.0 * s2)
        return logw + loge

    logs = jax.vmap(term)(ks)  # (alpha+1,)
    m = jnp.max(logs)
    return m + jnp.log(jnp.sum(jnp.exp(logs - m)))


def _rdp_sgm_step(q: float, sigma: float, alpha: int) -> Array:
    """RDP (order alpha) for one step of Poisson-sampled Gaussian mechanism."""
    if alpha < 2:
        return jnp.asarray(jnp.inf, DT)
    logA = _compute_log_a_int(q, sigma, alpha)
    return (logA / (jnp.asarray(alpha - 1.0, DT))).astype(DT)


def _rdp_total(q: float, sigma: float, steps: int, orders: Iterable[int]) -> Array:
    """Vector of total RDP over `steps` for each integer order in `orders`."""
    orders_list = [int(o) for o in orders if int(o) >= 2]
    per_list = [_rdp_sgm_step(q, sigma, a) for a in orders_list]
    per = jnp.stack(per_list).astype(DT)  # (len(orders),)
    return jnp.asarray(steps, DT) * per


def _eps_from_rdp(rdp: Array, orders: Sequence[int], delta: float) -> float:
    """Convert vector of RDPs at different orders to (ε, δ)."""
    orders_arr = jnp.asarray([int(o) for o in orders if int(o) >= 2], DT)
    rdp = jnp.asarray(rdp, DT)
    delta = jnp.asarray(delta, DT)
    eps = rdp + jnp.log(1.0 / jnp.maximum(delta, 1e-300)) / (orders_arr - 1.0)
    return float(jnp.min(eps))


@dataclass
class DPSGDConfig:
    clipping_norm: float
    noise_multiplier: float
    delta: float = 1e-5
    # SOTA toggles/additions
    poisson_sampling: bool = False  # True => Bernoulli(q) subsampling each step
    sampling_rate: Optional[float] = None  # if None, trainer will set to batch_size/N
    microbatch_size: Optional[int] = None  # split batches inside a DP step
    # RDP orders (integer); default is a rich grid
    orders: Tuple[int, ...] = tuple(list(range(2, 65)) + [80, 96, 128, 256, 512])


class RDPAccountant:
    """
    RDP accountant for Poisson-sampled Gaussian mechanism.
    Tracks total RDP across steps, and returns the best ε for a given δ.
    """

    def __init__(
        self, orders: Sequence[int] = tuple(range(2, 65)), delta: float = 1e-5
    ):
        self.orders = tuple(int(o) for o in orders if int(o) >= 2)
        self.delta = float(delta)
        self._steps = 0
        self._rdp = jnp.zeros((len(self.orders),), dtype=DT)
        self._q = None
        self._sigma = None

    def reset(self, *, sigma: float, q: float):
        self._steps = 0
        self._rdp = jnp.zeros((len(self.orders),), dtype=DT)
        self._sigma = float(sigma)
        self._q = float(q)

    def accumulate(
        self,
        num_steps: int = 1,
        *,
        q: Optional[float] = None,
        sigma: Optional[float] = None,
    ):
        q_use = float(self._q if q is None else q)
        s_use = float(self._sigma if sigma is None else sigma)
        add = _rdp_total(q_use, s_use, int(num_steps), self.orders)  # (len(orders),)
        self._rdp = self._rdp + add
        self._steps += int(num_steps)

    def epsilon(self, delta: Optional[float] = None) -> float:
        return _eps_from_rdp(
            self._rdp, self.orders, float(self.delta if delta is None else delta)
        )


class DPPrivacyEngine:
    """
    DP-SGD engine: per-example clipping, Gaussian noise on the **sum** of clipped grads.
    - Supports Poisson sampling (Bernoulli(q)) and microbatching.
    - Uses RDP accountant for Sampled Gaussian Mechanism.
    """

    def __init__(self, cfg: DPSGDConfig):
        self.cfg = cfg
        self.accountant = RDPAccountant(orders=cfg.orders, delta=cfg.delta)
        self._q_default = cfg.sampling_rate if cfg.sampling_rate is not None else None
        # initialize accountant (trainer will update q each step)
        self.accountant.reset(
            sigma=float(cfg.noise_multiplier), q=float(self._q_default or 0.0)
        )

    def _per_example_grads(
        self, loss_fn, model, state, xb: Array, yb: Array, key: jax.Array
    ):
        """Per-example grads in inference mode (state not mutated)."""
        infer_model = eqx.nn.inference_mode(model, value=True)

        def single(x1, y1, k1):
            (val, _), g = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                infer_model, state, x1[None, ...], y1[None, ...], k1
            )
            return g

        keys = jr.split(key, xb.shape[0])
        grads_batched = jax.vmap(single)(xb, yb, keys)  # PyTree with leading batch axis
        return grads_batched

    def noisy_grad(
        self,
        loss_fn,
        model: eqx.Module,
        state: Any,
        xb: Array,
        yb: Array,
        *,
        key: jax.Array,
        sample_rate: Optional[float] = None,
    ) -> Tuple[Pytree, Any]:
        """
        Returns (noisy_mean_grad, new_state). Noise is calibrated to the **sum** of
        clipped per-example grads (std = sigma*C), then divided by batch size.
        """
        C = float(self.cfg.clipping_norm)
        sigma = float(self.cfg.noise_multiplier)

        B = int(xb.shape[0])
        assert B > 0, "Empty batch encountered in DP step."

        # Microbatching
        mbs = (
            int(self.cfg.microbatch_size) if (self.cfg.microbatch_size or 0) > 0 else B
        )
        num_chunks = (B + mbs - 1) // mbs

        grad_sum = None
        rng = key
        for i in range(num_chunks):
            s = i * mbs
            e = min(B, s + mbs)
            if e <= s:
                break
            xmb = xb[s:e]
            ymb = yb[s:e]
            rng, sub = jr.split(rng)

            g_batched = self._per_example_grads(loss_fn, model, state, xmb, ymb, sub)
            g_clipped = _per_example_clip(g_batched, C)
            g_sum_mb = _sum_over_batch(g_clipped)
            grad_sum = g_sum_mb if grad_sum is None else _tree_add(grad_sum, g_sum_mb)

        # Add Gaussian noise to the **sum**, then divide by B
        keys = jr.split(rng, len(jax.tree_util.tree_leaves(grad_sum)))
        it = iter(keys)

        def add_noise(leaf):
            return leaf + sigma * C * jr.normal(
                next(it), shape=leaf.shape, dtype=leaf.dtype
            )

        noisy_sum = jax.tree_util.tree_map(add_noise, grad_sum)
        noisy_mean = _tree_scale(noisy_sum, 1.0 / float(B))

        # Update accountant if we know q
        q_use = float(
            sample_rate if (sample_rate is not None) else (self._q_default or 0.0)
        )
        if q_use > 0.0:
            self.accountant.accumulate(1, q=q_use, sigma=sigma)

        # Update mutable state once (e.g., BN running stats) using a single forward
        _, new_state = loss_fn(model, state, xb, yb, jr.fold_in(key, 12345))
        return noisy_mean, new_state

    def epsilon(self) -> float:
        return self.accountant.epsilon()
