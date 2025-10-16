import copy
from typing import Callable, Optional, List, Any
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import numpy as np

from quantbayes.stochax.trainer.train import (
    train as eqx_train,
    binary_loss,
    eval_step,
    predict,
)
from quantbayes.stochax.privacy.dp import DPSGDConfig
from quantbayes.stochax.privacy.dp_train import dp_eqx_train

__all__ = ["FederatedTrainer"]


def _tree_zeros_like(a):
    return jax.tree_util.tree_map(jnp.zeros_like, a)


def _tree_scale(a, s: float):
    return jax.tree_util.tree_map(lambda x: s * x, a)


class FedAdamServer:
    def __init__(self, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = (
            float(lr),
            float(beta1),
            float(beta2),
            float(eps),
        )
        self.m = None
        self.v = None
        self.t = 0

    def apply(
        self, theta_g: eqx.Module, local_models: List[eqx.Module], weights: List[float]
    ) -> eqx.Module:
        p_g = eqx.filter(theta_g, eqx.is_inexact_array)
        p_loc = [eqx.filter(m, eqx.is_inexact_array) for m in local_models]

        def wsum(*leaves):
            return sum(w * leaf for w, leaf in zip(weights, leaves))

        avg_params = jax.tree_util.tree_map(wsum, *p_loc)
        delta = jax.tree_util.tree_map(lambda a, b: a - b, avg_params, p_g)

        if self.m is None:
            self.m, self.v = _tree_zeros_like(delta), _tree_zeros_like(delta)
        self.t += 1
        self.m = jax.tree_util.tree_map(
            lambda m, d: self.b1 * m + (1 - self.b1) * d, self.m, delta
        )
        self.v = jax.tree_util.tree_map(
            lambda v, d: self.b2 * v + (1 - self.b2) * (d * d), self.v, delta
        )
        mhat = _tree_scale(self.m, 1.0 / (1 - self.b1**self.t))
        vhat = _tree_scale(self.v, 1.0 / (1 - self.b2**self.t))
        update = jax.tree_util.tree_map(
            lambda m, v: self.lr * m / (jnp.sqrt(v) + self.eps), mhat, vhat
        )
        new_params = jax.tree_util.tree_map(lambda a, u: a + u, p_g, update)
        _, static = eqx.partition(theta_g, eqx.is_inexact_array)
        return eqx.combine(new_params, static)


def coord_median(models: List[eqx.Module]) -> eqx.Module:
    params = [eqx.filter(m, eqx.is_inexact_array) for m in models]

    def med(*leaves):
        stack = jnp.stack(leaves, axis=0)
        return jnp.median(stack, axis=0)

    med_params = jax.tree_util.tree_map(med, *params)
    _, static = eqx.partition(models[0], eqx.is_inexact_array)
    return eqx.combine(med_params, static)


def trimmed_mean(models: List[eqx.Module], trim_ratio: float = 0.1) -> eqx.Module:
    params = [eqx.filter(m, eqx.is_inexact_array) for m in models]
    k = len(models)
    t = int(jnp.floor(trim_ratio * k))

    def tmean(*leaves):
        stack = jnp.sort(jnp.stack(leaves, axis=0), axis=0)
        trimmed = stack[t : k - t] if (k - 2 * t) > 0 else stack
        return jnp.mean(trimmed, axis=0)

    tm_params = jax.tree_util.tree_map(tmean, *params)
    _, static = eqx.partition(models[0], eqx.is_inexact_array)
    return eqx.combine(tm_params, static)


class FederatedTrainer:
    """
    FedAvg + (optional) DP local updates, FedProx, robust aggregation, FedAdam, FedBN.
    Defaults reproduce old behavior.
    """

    def __init__(
        self,
        model_init_fn: Callable[[jax.Array], eqx.Module],
        n_nodes: int = 5,
        outer_rounds: int = 10,
        inner_epochs: int = 1,
        batch_size: Optional[int] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_spec: float = 0.0,
        patience: int = 5,
        key: Optional[jax.Array] = None,
        loss_fn=binary_loss,
        # NEW toggles (all default to legacy behavior):
        mu: float = 0.0,  # FedProx prox strength (0 => off)
        dp_config: Optional[DPSGDConfig] = None,  # DP-SGD local training
        aggregator: Optional[
            dict
        ] = None,  # {"name":"mean"|"median"|"trimmed","trim_ratio":0.1}
        server_opt: Optional[
            dict
        ] = None,  # {"name":"none"|"fedadam", "lr":..., "beta1":..., "beta2":..., "eps":...}
        keep_bn_local: bool = True,  # FedBN (True => do not aggregate BN buffers)
    ):
        self.model_init_fn = model_init_fn
        self.n_nodes = n_nodes
        self.outer_rounds = outer_rounds
        self.inner_epochs = inner_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_spec = lambda_spec
        self.patience = patience
        self.key = key if key is not None else jr.PRNGKey(0)
        self.loss_fn = loss_fn

        self.mu = float(mu)
        self.dp_config = dp_config
        self.aggregator = aggregator or {"name": "mean", "trim_ratio": 0.1}
        self.server_opt = server_opt or {
            "name": "none",
            "lr": 1e-2,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        }
        self.keep_bn_local = bool(keep_bn_local)

        self.train_histories: List[List[List[float]]] = []
        self.val_histories: List[List[List[float]]] = []
        self.test_losses: List[float] = []
        self.global_model: Optional[eqx.Module] = None
        self.global_state: Optional[Any] = None
        self.client_states: Optional[List[Any]] = None

    @staticmethod
    def shard_data(X: jnp.ndarray, y: jnp.ndarray, n_nodes: int):
        N = X.shape[0]
        base, rem = divmod(N, n_nodes)
        shards_x, shards_y, sizes = [], [], []
        idx = 0
        for i in range(n_nodes):
            sz = base + (1 if i < rem else 0)
            shards_x.append(X[idx : idx + sz])
            shards_y.append(y[idx : idx + sz])
            sizes.append(sz)
            idx += sz
        return shards_x, shards_y, sizes

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

    def _fedprox_loss(self, model, state, xb, yb, k, theta_g_params):
        base, new_state = self.loss_fn(model, state, xb, yb, k)
        if self.mu > 0.0:
            p = eqx.filter(model, eqx.is_inexact_array)
            prox = (
                0.5
                * self.mu
                * sum(
                    jnp.sum((a - b) ** 2)
                    for a, b in zip(
                        jax.tree_util.tree_leaves(p),
                        jax.tree_util.tree_leaves(theta_g_params),
                    )
                )
            )
            base = base + prox
        return base, new_state

    def _aggregate(
        self, local_models: List[eqx.Module], sizes: List[int]
    ) -> eqx.Module:
        name = self.aggregator.get("name", "mean")
        if name == "mean":
            return self._weighted_avg(local_models, sizes)
        elif name == "median":
            return coord_median(local_models)
        elif name == "trimmed":
            r = float(self.aggregator.get("trim_ratio", 0.1))
            return trimmed_mean(local_models, r)
        else:
            raise ValueError(f"Unknown aggregator {name}")

    def train(
        self,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        X_test: jnp.ndarray,
        y_test: jnp.ndarray,
    ):
        """
        FedAvg/FedProx training with optional DP, robust aggregators, FedAdam, and FedBN.

        FedBN semantics (when keep_bn_local=True):
        - BN running stats are NOT aggregated; we keep per-client BN buffers.
        - evaluation uses a weighted average of per-client losses, each with its own BN state.

        Returns
        -------
        (global_model, test_losses)
        """
        # Local import to keep copy-paste friction low
        from quantbayes.stochax.utils.equinox_helpers import clone_module

        # Shard data
        X_shards, y_shards, sizes = self.shard_data(X_train, y_train, self.n_nodes)

        # Init global model/state
        self.key, init_key = jr.split(self.key)
        self.global_model, self.global_state = eqx.nn.make_with_state(
            self.model_init_fn
        )(init_key)

        # Optimizer for local client training
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=self.learning_rate, weight_decay=self.weight_decay
            ),
        )

        # Optional FedAdam server
        server = None
        if self.server_opt.get("name", "none") == "fedadam":
            server = FedAdamServer(
                lr=self.server_opt.get("lr", 1e-2),
                beta1=self.server_opt.get("beta1", 0.9),
                beta2=self.server_opt.get("beta2", 0.999),
                eps=self.server_opt.get("eps", 1e-8),
            )

        # Bookkeeping
        self.train_histories = []
        self.val_histories = []
        self.test_losses = []
        self.client_states = None  # for FedBN

        for rnd in range(1, self.outer_rounds + 1):
            # Split RNG for this round
            self.key, *subkeys = jr.split(self.key, self.n_nodes + 1)

            # Snapshot global params for FedProx proximal term
            theta_g_params = eqx.filter(self.global_model, eqx.is_inexact_array)

            round_tr, round_va = [], []
            local_models, local_states = [], []

            # ----- Local client training -----
            for i in range(self.n_nodes):
                # structural clone of the global model/state (best-practice in Equinox)
                lm = clone_module(self.global_model)
                ls = clone_module(self.global_state)

                opt_state = optimizer.init(eqx.filter(lm, eqx.is_inexact_array))

                def prox_loss(m, s, xb, yb, k):
                    # L_i(θ) + (μ/2)||θ - θ_g||^2   if self.mu>0
                    base, new_state = self._fedprox_loss(
                        m, s, xb, yb, k, theta_g_params
                    )
                    return base, new_state

                if self.dp_config is None:
                    lm_trained, ls_trained, tr_hist, va_hist = eqx_train(
                        lm,
                        ls,
                        opt_state,
                        optimizer,
                        prox_loss,
                        X_shards[i],
                        y_shards[i],
                        X_shards[i],
                        y_shards[i],
                        batch_size=self.batch_size or X_shards[i].shape[0],
                        num_epochs=self.inner_epochs,
                        patience=self.patience,
                        key=subkeys[i],
                        lambda_spec=self.lambda_spec,
                    )
                else:
                    lm_trained, ls_trained, tr_hist, va_hist = dp_eqx_train(
                        lm,
                        ls,
                        opt_state,
                        optimizer,
                        prox_loss,
                        X_shards[i],
                        y_shards[i],
                        X_shards[i],
                        y_shards[i],
                        dp_config=self.dp_config,
                        batch_size=self.batch_size or X_shards[i].shape[0],
                        num_epochs=self.inner_epochs,
                        patience=self.patience,
                        key=subkeys[i],
                    )

                local_models.append(lm_trained)
                local_states.append(ls_trained)
                round_tr.append(tr_hist)
                round_va.append(va_hist)

            # ----- Server aggregation of weights -----
            if server is None:
                self.global_model = self._aggregate(local_models, sizes)
            else:
                ws = [s / (sum(sizes) + 1e-12) for s in sizes]
                self.global_model = server.apply(self.global_model, local_models, ws)

            # ----- State/Buffers (FedBN by default) -----
            if self.keep_bn_local:
                # Do NOT aggregate BN buffers; keep each client's buffers.
                self.client_states = local_states
                # Keep a nominal global_state for APIs that expect one.
                self.global_state = local_states[0]
            else:
                # (Optional) implement BN aggregation here if desired.
                self.global_state = local_states[0]

            # ----- Evaluation -----
            self.key, eval_key = jr.split(self.key)
            if self.keep_bn_local and self.client_states is not None:
                # Weighted average of per-client eval losses (each with its BN buffers)
                total = float(sum(sizes))
                loss = 0.0
                for s_i, n_i in zip(self.client_states, sizes):
                    loss_i = float(
                        eval_step(
                            self.global_model,
                            s_i,
                            X_test,
                            y_test,
                            eval_key,
                            self.loss_fn,
                        )
                    )
                    loss += (n_i / (total + 1e-12)) * loss_i
            else:
                # Standard eval with single (global) state
                loss = float(
                    eval_step(
                        self.global_model,
                        self.global_state,
                        X_test,
                        y_test,
                        eval_key,
                        self.loss_fn,
                    )
                )

            # ----- Logging -----
            self.test_losses.append(loss)
            self.train_histories.append(round_tr)
            self.val_histories.append(round_va)
            print(f"[Fed] round {rnd}/{self.outer_rounds} | test loss={loss:.4f}")

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
    n_total, d = 4000, 25
    X = rng.randn(n_total, d).astype(np.float32)
    w_true = (rng.randn(d) / np.sqrt(d)).astype(np.float32)
    logits = X @ w_true
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.rand(n_total) < p).astype(np.float32)

    idx = rng.permutation(n_total)
    X, y = X[idx], y[idx]
    n_train = int(0.8 * n_total)
    X_tr_np, X_te_np = X[:n_train], X[n_train:]
    y_tr_np, y_te_np = y[:n_train], y[n_train:]

    mu = X_tr_np.mean(axis=0, keepdims=True)
    sd = X_tr_np.std(axis=0, keepdims=True) + 1e-8
    X_tr_np = (X_tr_np - mu) / sd
    X_te_np = (X_te_np - mu) / sd

    X_train = jnp.array(X_tr_np)
    y_train = jnp.array(y_tr_np)
    X_test = jnp.array(X_te_np)
    y_test = jnp.array(y_te_np)

    # -------- simple MLP --------
    class SimpleMLP(eqx.Module):
        fc1: eqx.nn.Linear
        fc2: eqx.nn.Linear

        def __init__(self, key):
            k1, k2 = jr.split(key)
            self.fc1 = eqx.nn.Linear(d, 64, key=k1)
            self.fc2 = eqx.nn.Linear(64, 1, key=k2)

        def __call__(self, x, key=None, state=None):
            x = jax.nn.relu(self.fc1(x))
            return self.fc2(x), state

    # -------- try a couple aggregators --------
    agg_cfgs = [
        {"name": "mean", "kwargs": {}},
        {"name": "median", "kwargs": {}},
        {"name": "trimmed", "kwargs": {"trim_ratio": 0.2}},
    ]
    curves = {}

    for i, cfg in enumerate(agg_cfgs):
        trainer = FederatedTrainer(
            model_init_fn=SimpleMLP,
            n_nodes=6,
            outer_rounds=10,
            inner_epochs=3,
            batch_size=128,
            learning_rate=5e-3,
            weight_decay=1e-4,
            lambda_spec=0.0,
            patience=3,
            key=jr.PRNGKey(100 + i),
            aggregator={"name": cfg["name"], **cfg["kwargs"]},
            keep_bn_local=True,  # FedBN semantics (no BN in this toy model)
        )
        _, test_losses = trainer.train(X_train, y_train, X_test, y_test)
        curves[cfg["name"]] = test_losses

    # -------- plot global loss vs round --------
    rounds = np.arange(1, max(len(v) for v in curves.values()) + 1)
    plt.figure(figsize=(7.6, 4.4))
    for name, loss in curves.items():
        plt.plot(rounds[: len(loss)], loss, label=name, linewidth=2)
    plt.yscale("log")
    plt.xlabel("Round")
    plt.ylabel("Global test loss")
    plt.title("FedAvg — Aggregator comparison (synthetic)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------- print finals --------
    for name, loss in curves.items():
        print(f"[{name:7s}] final test loss = {loss[-1]:.4f}")
