# quantbayes/ball_dp/heads/prototypes_eqx.py
from __future__ import annotations
from typing import Tuple
import equinox as eqx
import jax
import jax.numpy as jnp


class RidgePrototypesEqx(eqx.Module):
    mus: jnp.ndarray  # (K,d)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # squared distances to each prototype
        d2 = jnp.sum((self.mus - x[None, :]) ** 2, axis=1)  # (K,)
        return jnp.argmin(d2)

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self.__call__)(X)


def fit_ridge_prototypes_eqx(
    X: jnp.ndarray,
    y: jnp.ndarray,
    *,
    num_classes: int,
    lam: float,
) -> Tuple[RidgePrototypesEqx, jnp.ndarray]:
    X = jnp.asarray(X, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.int32)
    K = int(num_classes)
    n_total = X.shape[0]

    mus_list = []
    counts_list = []

    for c in range(K):
        mask = y == c  # (n,)
        nc = jnp.sum(mask).astype(jnp.float32)
        counts_list.append(nc)

        s = jnp.sum(jnp.where(mask[:, None], X, 0.0), axis=0)  # (d,)
        denom = 2.0 * nc + float(lam) * float(n_total)
        mu_c = jnp.where(nc > 0, (2.0 * s) / denom, jnp.zeros_like(s))
        mus_list.append(mu_c)

    mus = jnp.stack(mus_list, axis=0)  # (K,d)
    counts = jnp.stack(counts_list).astype(jnp.int32)  # (K,)
    return RidgePrototypesEqx(mus=mus), counts
