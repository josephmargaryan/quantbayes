# quantbayes/stochax/federated/fedavg.py
from __future__ import annotations
from typing import Callable, Optional, List, Any, Dict
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

from quantbayes.stochax.trainer.train import (
    train as eqx_train,
    binary_loss,
    eval_step,
)
from quantbayes.stochax.privacy.dp import DPSGDConfig
from quantbayes.stochax.privacy.dp_train import dp_eqx_train


# ---------- FedOpt server (Adam/Yogi/Adagrad/Momentum) ----------
class FedOptServer:
    """
    FedOpt family on server deltas Δ = (avg_local_params - global_params).
    name ∈ {"adam","yogi","adagrad","momentum"}.
    """

    def __init__(
        self, name="adam", lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, momentum=0.9
    ):
        self.name = str(name).lower()
        self.lr, self.b1, self.b2, self.eps, self.mu = (
            float(lr),
            float(beta1),
            float(beta2),
            float(eps),
            float(momentum),
        )
        self.m = None  # first moment / momentum
        self.v = None  # second moment
        self.t = 0

    @staticmethod
    def _zeros_like(a):
        return jax.tree_util.tree_map(jnp.zeros_like, a)

    @staticmethod
    def _scale(a, s):
        return jax.tree_util.tree_map(lambda x: s * x, a)

    def apply(
        self, theta_g: eqx.Module, local_models: List[eqx.Module], weights: List[float]
    ) -> eqx.Module:
        p_g = eqx.filter(theta_g, eqx.is_inexact_array)
        p_loc = [eqx.filter(m, eqx.is_inexact_array) for m in local_models]

        def wsum(*leaves):
            return sum(w * leaf for w, leaf in zip(weights, leaves))

        avg_params = jax.tree_util.tree_map(wsum, *p_loc)
        delta = jax.tree_util.tree_map(
            lambda a, b: a - b, avg_params, p_g
        )  # Δ = avg - global

        if self.name == "momentum":
            if self.m is None:
                self.m = self._zeros_like(delta)
            self.m = jax.tree_util.tree_map(lambda m, d: self.mu * m + d, self.m, delta)
            update = self._scale(self.m, self.lr)
        else:
            if self.m is None:
                self.m = self._zeros_like(delta)
            if self.v is None:
                self.v = self._zeros_like(delta)
            self.t += 1
            self.m = jax.tree_util.tree_map(
                lambda m, d: self.b1 * m + (1 - self.b1) * d, self.m, delta
            )
            if self.name == "adam":
                self.v = jax.tree_util.tree_map(
                    lambda v, d: self.b2 * v + (1 - self.b2) * (d * d), self.v, delta
                )
            elif self.name == "yogi":
                self.v = jax.tree_util.tree_map(
                    lambda v, d: v - (1 - self.b2) * jnp.sign(v - d * d) * (d * d),
                    self.v,
                    delta,
                )
            elif self.name == "adagrad":
                self.v = jax.tree_util.tree_map(lambda v, d: v + d * d, self.v, delta)
            else:
                raise ValueError(f"Unknown FedOpt name '{self.name}'")

            if self.name in {"adam", "yogi"}:
                mhat = self._scale(self.m, 1.0 / (1 - self.b1**self.t))
                vhat = self._scale(self.v, 1.0 / (1 - self.b2**self.t))
                update = jax.tree_util.tree_map(
                    lambda m, v: self.lr * m / (jnp.sqrt(v) + self.eps), mhat, vhat
                )
            else:  # Adagrad
                update = jax.tree_util.tree_map(
                    lambda m, v: self.lr * m / (jnp.sqrt(v) + self.eps), self.m, self.v
                )

        new_params = jax.tree_util.tree_map(lambda a, u: a + u, p_g, update)
        _, static = eqx.partition(theta_g, eqx.is_inexact_array)
        return eqx.combine(new_params, static)


# ---------- robust aggregators ----------
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


def _tree_norm(a):
    leaves = jax.tree_util.tree_leaves(a)
    return jnp.sqrt(sum(jnp.sum(x * x) for x in leaves))


def _sub(a, b):
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def _scale(a, s):
    return jax.tree_util.tree_map(lambda x: s * x, a)


def geometric_median(
    models: List[eqx.Module], max_iter: int = 80, eps: float = 1e-6
) -> eqx.Module:
    params = [eqx.filter(m, eqx.is_inexact_array) for m in models]
    x = params[0]
    for _ in range(max_iter):
        dists = [float(_tree_norm(_sub(x, p))) + 1e-12 for p in params]
        inv = [1.0 / d for d in dists]
        s = sum(inv)
        w = [v / s for v in inv]
        x_new = jax.tree_util.tree_map(
            lambda *leaves: sum(wi * leaf for wi, leaf in zip(w, leaves)), *params
        )
        if float(_tree_norm(_sub(x_new, x))) < eps:
            x = x_new
            break
        x = x_new
    _, static = eqx.partition(models[0], eqx.is_inexact_array)
    return eqx.combine(x, static)


# ---------- Federated Trainer (FedAvg/FedBN/DP + FedOpt, robust, q-FedAvg) ----------
class FederatedTrainer:
    """
    FedAvg + (optional) DP local updates, FedProx (μ=0 here), robust aggregation (coord-median/trimmed/geomed),
    FedOpt server (Adam/Yogi/Adagrad/Momentum), FedBN, and q-FedAvg fairness weighting.
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
        patience: int = 5,
        key: Optional[jax.Array] = None,
        loss_fn=binary_loss,
        dp_config: Optional[DPSGDConfig] = None,  # DP-SGD local training
        aggregator: Optional[
            dict
        ] = None,  # {"name":"mean"|"median"|"trimmed"|"geom_median", "trim_ratio":0.1, "q":0.0}
        server_opt: Optional[
            dict
        ] = None,  # {"name":"none"|"adam"|"yogi"|"adagrad"|"momentum", ...}
        keep_bn_local: bool = True,  # FedBN (True => do not aggregate BN buffers)
    ):
        self.model_init_fn = model_init_fn
        self.n_nodes = int(n_nodes)
        self.R = int(outer_rounds)
        self.E = int(inner_epochs)
        self.batch_size = batch_size
        self.lr_local = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.key = jr.PRNGKey(0) if key is None else key
        self.loss_fn = loss_fn

        self.dp_config = dp_config
        self.aggregator = aggregator or {"name": "mean", "trim_ratio": 0.1, "q": 0.0}
        self.server_opt = server_opt or {
            "name": "adam",
            "lr": 1e-2,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        }
        self.keep_bn_local = bool(keep_bn_local)

        self.test_losses: List[float] = []
        self.global_model: Optional[eqx.Module] = None
        self.global_state: Optional[Any] = None

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
    def _weighted_avg(models: List[eqx.Module], weights: np.ndarray) -> eqx.Module:
        param_trees = [eqx.filter(m, eqx.is_inexact_array) for m in models]
        w = jnp.asarray(weights, dtype=jnp.float32)

        def wsum(*leaves):
            return sum(wi * leaf for wi, leaf in zip(w, leaves))

        avg_params = jax.tree_util.tree_map(wsum, *param_trees)
        _, static = eqx.partition(models[0], eqx.is_inexact_array)
        return eqx.combine(avg_params, static)

    def _aggregate_models(
        self,
        local_models: List[eqx.Module],
        sizes: List[int],
        local_losses: Optional[List[float]],
    ) -> eqx.Module:
        name = self.aggregator.get("name", "mean").lower()
        if name == "mean":
            # q-FedAvg weighting (q=0 => size-weighted mean)
            q = float(self.aggregator.get("q", 0.0))
            if q > 0.0 and local_losses is not None:
                arr = np.asarray([(li + 1e-8) ** q for li in local_losses], dtype=float)
                w = arr / (arr.sum() + 1e-12)
            else:
                w = np.asarray([s / (sum(sizes) + 1e-12) for s in sizes], dtype=float)
            return self._weighted_avg(local_models, w)
        elif name == "median":
            return coord_median(local_models)
        elif name == "trimmed":
            r = float(self.aggregator.get("trim_ratio", 0.1))
            return trimmed_mean(local_models, r)
        elif name == "geom_median":
            return geometric_median(local_models)
        else:
            raise ValueError(f"Unknown aggregator {name}")

    def train(
        self,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        X_test: jnp.ndarray,
        y_test: jnp.ndarray,
    ) -> tuple[eqx.Module, List[float]]:
        """
        FedAvg/FedBN/DP with robust aggregators, FedOpt server (for mean aggregation),
        and optional q-FedAvg fairness weighting.
        """
        from quantbayes.stochax.utils.equinox_helpers import clone_module

        Xs, ys, sizes = self.shard_data(X_train, y_train, self.n_nodes)

        self.key, init_key = jr.split(self.key)
        self.global_model, self.global_state = eqx.nn.make_with_state(
            self.model_init_fn
        )(init_key)

        # local optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=self.lr_local, weight_decay=self.weight_decay),
        )

        # optional FedOpt server (only meaningful with mean aggregation)
        server = None
        if self.server_opt.get("name", "none").lower() in {
            "adam",
            "yogi",
            "adagrad",
            "momentum",
        }:
            server = FedOptServer(**self.server_opt)

        self.test_losses = []

        for rnd in range(1, self.R + 1):
            # snapshot RNG & global
            self.key, *subkeys = jr.split(self.key, self.n_nodes + 1)
            local_models, local_states = [], []
            client_losses_for_q: List[float] = []

            for i in range(self.n_nodes):
                # clone global
                lm = clone_module(self.global_model)
                ls = clone_module(self.global_state)
                opt_state = optimizer.init(eqx.filter(lm, eqx.is_inexact_array))

                if self.dp_config is None:
                    lm_trained, ls_trained, *_ = eqx_train(
                        lm,
                        ls,
                        opt_state,
                        optimizer,
                        self.loss_fn,
                        Xs[i],
                        ys[i],
                        Xs[i],
                        ys[i],
                        batch_size=self.batch_size or Xs[i].shape[0],
                        num_epochs=self.E,
                        patience=self.patience,
                        key=subkeys[i],
                    )
                else:
                    lm_trained, ls_trained, *_ = dp_eqx_train(
                        lm,
                        ls,
                        opt_state,
                        optimizer,
                        self.loss_fn,
                        Xs[i],
                        ys[i],
                        Xs[i],
                        ys[i],
                        dp_config=self.dp_config,
                        batch_size=self.batch_size or Xs[i].shape[0],
                        num_epochs=self.E,
                        patience=self.patience,
                        key=subkeys[i],
                    )

                # local shard loss (for q-FedAvg if used)
                self.key, lkey = jr.split(self.key)
                li = float(
                    eval_step(lm_trained, ls_trained, Xs[i], ys[i], lkey, self.loss_fn)
                )
                client_losses_for_q.append(li)

                local_models.append(lm_trained)
                local_states.append(ls_trained)

            name = self.aggregator.get("name", "mean").lower()
            if server is not None and name == "mean":
                # server FedOpt via deltas with (possibly) fairness weights
                q = float(self.aggregator.get("q", 0.0))
                if q > 0.0 and client_losses_for_q:
                    arr = np.asarray(
                        [(li + 1e-8) ** q for li in client_losses_for_q], dtype=float
                    )
                    weights = (arr / (arr.sum() + 1e-12)).tolist()
                else:
                    weights = [s / (sum(sizes) + 1e-12) for s in sizes]
                self.global_model = server.apply(
                    self.global_model, local_models, weights
                )
                # FedBN: keep BN buffers per-client
                self.global_state = local_states[0]  # nominal state
                client_states = local_states
            else:
                # robust aggregation or mean without FedOpt
                self.global_model = self._aggregate_models(
                    local_models, sizes, client_losses_for_q
                )
                self.global_state = local_states[0]
                client_states = local_states

            # evaluation (FedBN: weighted avg of per-client losses using their BN states)
            total = float(sum(sizes))
            if self.keep_bn_local and client_states is not None:
                loss = 0.0
                for s_i, n_i in zip(client_states, sizes):
                    self.key, evk = jr.split(self.key)
                    loss_i = float(
                        eval_step(
                            self.global_model, s_i, X_test, y_test, evk, self.loss_fn
                        )
                    )
                    loss += (n_i / (total + 1e-12)) * loss_i
            else:
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
            print(f"[FedAvg+] round {rnd}/{self.R} | test loss={loss:.4f}")

        return self.global_model, self.test_losses


# ------------------------------- MAIN ---------------------------------
if __name__ == "__main__":
    import os
    import numpy as np
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx
    from quantbayes.stochax.distributed_training.helpers import (
        load_mnist_38,
        estimate_gamma_logistic,
        make_polynomial_decay,  # unused here, but available if you want LR schedules
        plot_global_loss_q3,
        summarize_histories,
        print_publication_summary,
        latex_table_from_summary,
    )

    # ----- data -----
    try:
        X, y = load_mnist_38(seed=0, flatten=True, standardize=True)
    except Exception:
        rng = np.random.RandomState(0)
        n, d = 6000, 50
        X = rng.randn(n, d).astype(np.float32)
        w = (rng.randn(d) / np.sqrt(d)).astype(np.float32)
        logits = X @ w
        p = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.rand(n) < p).astype(np.float32)
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
        X = (X - mu) / sd
        X, y = jnp.asarray(X), jnp.asarray(y)

    # train/test split
    n = int(X.shape[0])
    ntr = int(0.8 * n)
    Xtr, Xte = X[:ntr], X[ntr:]
    ytr, yte = y[:ntr], y[ntr:]

    # model
    d = int(X.shape[1])

    class LR(eqx.Module):
        lin: eqx.nn.Linear

        def __init__(self, d: int, key):
            self.lin = eqx.nn.Linear(d, 1, key=key)

        def __call__(self, x, key=None, state=None):
            return self.lin(x).squeeze(-1), state

    model_init = lambda k: LR(d, k)

    histories: Dict[str, Dict[str, List[float]]] = {}

    # A) FedAvg + FedOpt(Adam) mean aggregation (baseline modern)
    trainer_A = FederatedTrainer(
        model_init_fn=model_init,
        n_nodes=8,
        outer_rounds=25,
        inner_epochs=1,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-4,
        aggregator={"name": "mean", "q": 0.0},
        server_opt={
            "name": "adam",
            "lr": 5e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        },
        keep_bn_local=True,
        key=jr.PRNGKey(0),
    )
    _, losses_A = trainer_A.train(Xtr, ytr, Xte, yte)
    histories["FedOpt-Adam-mean"] = {"loss_node1": losses_A}

    # B) Robust geometric median (Byzantine-robust aggregation)
    trainer_B = FederatedTrainer(
        model_init_fn=model_init,
        n_nodes=8,
        outer_rounds=25,
        inner_epochs=1,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-4,
        aggregator={"name": "geom_median"},
        server_opt={"name": "none"},
        keep_bn_local=True,
        key=jr.PRNGKey(1),
    )
    _, losses_B = trainer_B.train(Xtr, ytr, Xte, yte)
    histories["GeomMedian"] = {"loss_node1": losses_B}

    # C) q-FedAvg fairness (q=0.5) + FedOpt(Adam)
    trainer_C = FederatedTrainer(
        model_init_fn=model_init,
        n_nodes=8,
        outer_rounds=25,
        inner_epochs=1,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-4,
        aggregator={"name": "mean", "q": 0.5},
        server_opt={
            "name": "adam",
            "lr": 5e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        },
        keep_bn_local=True,
        key=jr.PRNGKey(2),
    )
    _, losses_C = trainer_C.train(Xtr, ytr, Xte, yte)
    histories["qFedAvg(q=0.5)+Adam"] = {"loss_node1": losses_C}

    # ---- plots + summary ----
    os.makedirs("figs_fedavg", exist_ok=True)
    plot_global_loss_q3(
        histories,
        title="FedAvg+ variants",
        save="figs_fedavg/loss.png",
        style="accessible",
    )
    print("Saved figs_fedavg/loss.png")

    os.makedirs("tables", exist_ok=True)
    summary = summarize_histories(histories)
    print("\nNumeric summary (FedAvg+):")
    print_publication_summary(summary, decimals=4)
    latex = latex_table_from_summary(
        summary,
        decimals=3,
        caption="FedAvg+ with FedOpt/robust/q-FedAvg.",
        label="tab:fedavg_plus",
    )
    with open("tables/fedavg_plus_summary.tex", "w") as f:
        f.write(latex)
    print("Wrote tables/fedavg_plus_summary.tex")
