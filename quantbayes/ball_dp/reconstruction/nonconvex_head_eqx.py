# quantbayes/ball_dp/reconstruction/nonconvex_head_eqx.py
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


def _static_field():
    if hasattr(eqx, "field"):
        return eqx.field(static=True)
    return eqx.static_field()


class EmbeddingMLPClassifier(eqx.Module):
    """
    Simple MLP head on embeddings.
    Conforms to stochax trainer interface:
      (x, key, state) -> (logits, state)
    """

    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    act: str = _static_field()

    def __init__(
        self,
        d_in: int,
        n_classes: int,
        hidden: int,
        *,
        key: jr.PRNGKey,
        act: str = "elu",
    ):
        k1, k2 = jr.split(key)
        self.l1 = eqx.nn.Linear(d_in, hidden, key=k1)
        self.l2 = eqx.nn.Linear(hidden, n_classes, key=k2)
        self.act = str(act).lower()

    def _phi(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.act == "relu":
            return jax.nn.relu(x)
        if self.act == "gelu":
            return jax.nn.gelu(x)
        return jax.nn.elu(x)

    def __call__(self, x: jnp.ndarray, key: jr.PRNGKey, state):
        h = self._phi(self.l1(x))
        logits = self.l2(h)
        return logits, state


def init_embedding_mlp_classifier(
    *,
    d_in: int,
    n_classes: int,
    hidden: int = 32,
    act: str = "elu",
    seed: int = 0,
) -> EmbeddingMLPClassifier:
    return EmbeddingMLPClassifier(
        d_in, n_classes, hidden, key=jr.PRNGKey(int(seed)), act=act
    )


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.normal(size=(4, 8)).astype(np.float32))
    mdl = init_embedding_mlp_classifier(d_in=8, n_classes=3, hidden=16, seed=0)

    # vmap returns only logits because we index [0] inside lambda
    logits = jax.vmap(lambda x: mdl(x, jr.PRNGKey(0), None)[0])(X)
    print("logits shape:", logits.shape)
    assert logits.shape == (4, 3)
    print("[OK] nonconvex head forward works.")
