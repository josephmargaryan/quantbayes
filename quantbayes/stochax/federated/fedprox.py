# quantbayes/stochax/federated/fedprox.py
from __future__ import annotations
from typing import Any, Callable, Optional, List
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
import optax

from quantbayes.stochax.trainer.train import (
    train as eqx_train,
    eval_step,
    predict,
    binary_loss,
)
from quantbayes.stochax.utils.equinox_helpers import clone_module


class FedProxTrainer:
    """
    FedProx: local objective L_i(θ) + μ/2 ||θ - θ_global||^2. Same outer loop as FedAvg.
    """

    def __init__(
        self,
        model_init_fn: Callable[[jax.Array], eqx.Module],
        n_nodes: int = 5,
        outer_rounds: int = 10,
        inner_epochs: int = 1,
        batch_size: Optional[int] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        mu: float = 0.0,  # <-- proximal strength
        patience: int = 5,
        key: Optional[jax.Array] = None,
        loss_fn=binary_loss,
    ):
        self.model_init_fn = model_init_fn
        self.n_nodes = n_nodes
        self.R = outer_rounds
        self.E = inner_epochs
        self.batch = batch_size
        self.lr = lr
        self.wd = weight_decay
        self.mu = mu
        self.patience = patience
        self.key = jr.PRNGKey(0) if key is None else key
        self.loss_fn = loss_fn
        self.global_model: Optional[eqx.Module] = None
        self.global_state: Optional[Any] = None
        self.test_losses: List[float] = []

    @staticmethod
    def _weighted_avg(models: List[eqx.Module], sizes: List[int]) -> eqx.Module:
        ws = jnp.array(sizes, dtype=jnp.float32)
        ws = ws / (jnp.sum(ws) + 1e-12)
        param_trees = [eqx.filter(m, eqx.is_inexact_array) for m in models]

        def wsum(*leaves):
            return sum(w * leaf for w, leaf in zip(ws, leaves))

        avg_params = jax.tree_util.tree_map(wsum, *param_trees)
        _, static = eqx.partition(models[0], eqx.is_inexact_array)
        return eqx.combine(avg_params, static)

    def _fedprox_loss(
        self, theta: eqx.Module, state: Any, x, y, k, theta_global_params
    ):
        base, new_state = self.loss_fn(theta, state, x, y, k)
        # μ/2 ||θ - θ_global||^2 on trainable params only
        p_theta = eqx.filter(theta, eqx.is_inexact_array)
        prox = (
            0.5
            * self.mu
            * sum(
                jnp.sum((a - b) ** 2)
                for a, b in zip(
                    jax.tree_util.tree_leaves(p_theta),
                    jax.tree_util.tree_leaves(theta_global_params),
                )
            )
        )
        return base + prox, new_state

    def train(self, X_train, y_train, X_test, y_test):
        """
        FedProx outer loop with local prox objective:
        L_i(θ) + (μ/2)||θ - θ_global||^2
        Uses data-size weighted FedAvg aggregation of local models.
        """
        from quantbayes.stochax.utils.equinox_helpers import clone_module

        # Shard evenly by count
        N = X_train.shape[0]
        base, rem = divmod(N, self.n_nodes)
        Xs, ys, sizes = [], [], []
        s = 0
        for i in range(self.n_nodes):
            sz = base + (1 if i < rem else 0)
            Xs.append(X_train[s : s + sz])
            ys.append(y_train[s : s + sz])
            sizes.append(sz)
            s += sz

        # Init global
        self.key, init_key = jr.split(self.key)
        self.global_model, self.global_state = eqx.nn.make_with_state(
            self.model_init_fn
        )(init_key)

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=self.lr, weight_decay=self.wd),
        )

        self.test_losses = []

        for r in range(1, self.R + 1):
            local_models, local_states = [], []
            round_key, *client_keys = jr.split(self.key, self.n_nodes + 1)
            self.key = round_key

            theta_g_params = eqx.filter(self.global_model, eqx.is_inexact_array)

            for i in range(self.n_nodes):
                # structural clone (Equinox best practice)
                m_i = clone_module(self.global_model)
                s_i = clone_module(self.global_state)

                opt_state = optimizer.init(eqx.filter(m_i, eqx.is_inexact_array))

                def prox_loss(m, s, xb, yb, k):
                    base, new_state = self._fedprox_loss(
                        m, s, xb, yb, k, theta_g_params
                    )
                    return base, new_state

                m_i, s_i, _, _ = eqx_train(
                    m_i,
                    s_i,
                    opt_state,
                    optimizer,
                    prox_loss,
                    Xs[i],
                    ys[i],
                    Xs[i],
                    ys[i],
                    batch_size=self.batch or Xs[i].shape[0],
                    num_epochs=self.E,
                    patience=self.patience,
                    key=client_keys[i],
                )
                local_models.append(m_i)
                local_states.append(s_i)

            # Weighted average aggregation
            self.global_model = self._weighted_avg(local_models, sizes)

            # State handling: pick any client state as "global"; BN aggregation is typically avoided
            self.global_state = local_states[0]

            # Eval
            self.key, evk = jr.split(self.key)
            loss = float(
                eval_step(
                    self.global_model,
                    self.global_state,
                    X_test,
                    y_test,
                    evk,
                    self.loss_fn,
                )
            )
            self.test_losses.append(loss)
            print(f"[FedProx] round {r}/{self.R} | test loss={loss:.4f}")

        return self.global_model, self.test_losses


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx

    # -------- synthetic binary logistic data --------
    rng = np.random.RandomState(0)
    n_total, d = 3000, 20
    X = rng.randn(n_total, d).astype(np.float32)
    w_true = (rng.randn(d) / np.sqrt(d)).astype(np.float32)
    logits = X @ w_true
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.rand(n_total) < p).astype(np.float32)

    # shuffle, split, standardize on train
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

    # -------- simple LR model --------
    class LR(eqx.Module):
        lin: eqx.nn.Linear

        def __init__(self, key):
            self.lin = eqx.nn.Linear(d, 1, key=key)

        def __call__(self, x, key=None, state=None):
            return self.lin(x), state

    def model_init_fn(key: jax.Array) -> eqx.Module:
        return LR(key)

    # -------- run FedProx vs FedAvg --------
    configs = [
        {"name": "FedAvg (μ=0.0)", "mu": 0.0, "color": None},
        {"name": "FedProx (μ=0.1)", "mu": 0.1, "color": None},
    ]
    histories = {}

    for cfg in configs:
        trainer = FedProxTrainer(
            model_init_fn=model_init_fn,
            n_nodes=6,
            outer_rounds=12,
            inner_epochs=2,
            batch_size=128,
            lr=3e-3,
            weight_decay=1e-4,
            mu=cfg["mu"],
            patience=3,
            key=jr.PRNGKey(0 if cfg["mu"] == 0 else 1),
        )
        _, test_losses = trainer.train(X_tr, y_tr, X_te, y_te)
        histories[cfg["name"]] = test_losses

    # -------- plot --------
    plt.figure(figsize=(7.6, 4.4))
    for name, loss in histories.items():
        plt.plot(np.arange(1, len(loss) + 1), loss, linewidth=2, label=name)
    plt.yscale("log")
    plt.xlabel("Round")
    plt.ylabel("Global test loss")
    plt.title("FedAvg vs FedProx (synthetic logistic)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # final numbers
    for name, loss in histories.items():
        print(f"{name:18s}  final test loss = {loss[-1]:.4f}")
