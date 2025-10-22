import jax
import numpy as np
import jax.numpy as jnp
import equinox as eqx

from quantbayes.stochax.distributed_training import (
    LocalGDServerEqx,
    make_star_with_server_edges,
    dgd_safe_alpha,
    make_polynomial_decay,
    safe_alpha,
)

# ... build parts, X_full, y_full ...


class LR(eqx.Module):
    lin: eqx.nn.Linear

    def __init__(self, key):
        self.lin = eqx.nn.Linear(d, 1, key=key)

    def __call__(self, x, key, state):
        return self.lin(x), state


def model_init_fn(key: jax.Array) -> eqx.Module:

    return LR(key)


rng = np.random.RandomState(0)
n_total, d = 4000, 50
X = rng.randn(n_total, d).astype(np.float32)
w_true = (rng.randn(d) / np.sqrt(d)).astype(np.float32)
logits = X @ w_true
p = 1.0 / (1.0 + np.exp(-logits))
y = (rng.rand(n_total) < p).astype(np.float32)

# Shuffle, split, standardize on train
idx = rng.permutation(n_total)
X, y = X[idx], y[idx]
n_train = int(0.8 * n_total)
X_tr_np, X_te_np = X[:n_train], X[n_train:]
y_tr_np, y_te_np = y[:n_train], y[n_train:]

mu = X_tr_np.mean(axis=0, keepdims=True)
sd = X_tr_np.std(axis=0, keepdims=True) + 1e-8
X_tr_np = (X_tr_np - mu) / sd
X_te_np = (X_te_np - mu) / sd

X_tr = jnp.array(X_tr_np)
y_tr = jnp.array(y_tr_np)
X_te = jnp.array(X_te_np)
y_te = jnp.array(y_te_np)
X_full = jnp.concatenate([X_tr, X_te], axis=0)
y_full = jnp.concatenate([y_tr, y_te], axis=0)


def uniform_partition(X: jnp.ndarray, y: jnp.ndarray, n_nodes: int):
    N = X.shape[0]
    base, rem = divmod(N, n_nodes)
    parts = []
    start = 0
    for i in range(n_nodes):
        size = base + (1 if i < rem else 0)
        parts.append((X[start : start + size], y[start : start + size]))
        start += size
    return parts


n_nodes = 4
parts = uniform_partition(X_tr, y_tr, n_nodes)

edges_ring = [(0, 1), (1, 2), (2, 3), (3, 0)]
edges_star0 = [(0, 1), (0, 2), (0, 3)]
alpha = safe_alpha(edges_ring, n_nodes)


def estimate_gamma_full(X: jnp.ndarray) -> float:
    """Heuristic for logistic loss: L ≈ 0.25 * λ_max((X^T X)/n)."""
    n = X.shape[0]
    XtX = (X.T @ X) / max(1, n)
    v = jnp.ones((XtX.shape[0],), dtype=X.dtype)
    for _ in range(25):
        v = XtX @ v
        v = v / (jnp.linalg.norm(v) + 1e-12)
    lam_max = float(v @ (XtX @ v))
    L_smooth = 0.25 * lam_max
    return 0.9 / max(L_smooth, 1e-8)


gamma_full = estimate_gamma_full(X_tr)
gamma_sgd = 0.5 * gamma_full
T = 200

edges_A = make_star_with_server_edges(n_clients=4, server_id=4)
alpha_A = dgd_safe_alpha(edges_A, 5)

# Diminishing LR:
sched = make_polynomial_decay(gamma0=0.08, power=1.0, t0=1.0)

trainer = LocalGDServerEqx(
    model_init_fn=model_init_fn,
    n_clients=4,
    tau=10,
    edges_A=edges_A,
    alpha_A=alpha_A,
    gamma=sched,  # <--- schedule
    T=450,
    server_id=4,
)

# Spectral star policy:
trainer.star_policy = {
    "name": "cheby",
    "K": 3,
}  # or {"name":"repeat","K":3} / {"name":"single"}

hist = trainer.fit(parts=parts, X_full=X_full, y_full=y_full, log=True)
print("mean rho_star:", sum(hist["rho_star"]) / len(hist["rho_star"]))
