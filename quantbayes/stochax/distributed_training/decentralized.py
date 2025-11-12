# quantbayes/stochax/distributed_training/decentralized.py
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map
import equinox as eqx
import optax
import numpy as np
from typing import List, Tuple, Dict, Optional

from quantbayes.stochax.trainer.train import (
    train as eqx_train,
    predict,
    eval_step,
    binary_loss,
    multiclass_loss,
)
from quantbayes.stochax.privacy.dp import DPSGDConfig
from quantbayes.stochax.privacy.dp_train import dp_eqx_train

__all__ = ["DecentralizedTrainer"]


class DecentralizedTrainer:
    """
    Vanilla dual‐ascent decentralized trainer.
    """

    def __init__(
        self,
        model_init_fn,
        undirected_edges: List[Tuple[int, int]],
        n_nodes: int,
        outer_rounds: int,
        inner_epochs: int,
        gamma: float,
        lr: float,
        weight_decay: float = 0.0,
        patience: int = 3,
        batch_size: Optional[int] = None,
        key: jax.Array = jr.PRNGKey(0),
        loss_fn=binary_loss,
        dp_config: Optional[DPSGDConfig] = None,  # NEW
    ):
        self.model_init_fn = model_init_fn
        self.undirected_edges = undirected_edges
        self.n_nodes = n_nodes
        self.outer_rounds = outer_rounds
        self.inner_epochs = inner_epochs
        self.gamma = gamma
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.batch_size = batch_size
        self.key = key
        self.loss_fn = loss_fn
        self.dp_config = dp_config  # NEW

        self.train_histories: List[List[List[float]]] = []
        self.val_histories: List[List[List[float]]] = []
        self.test_losses: List[float] = []
        self.w_list = None
        self.nu = None

    @staticmethod
    def shard_data(
        X: jnp.ndarray, y: jnp.ndarray, n_nodes: int
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        N = X.shape[0]
        base, rem = divmod(N, n_nodes)
        shards_x, shards_y = [], []
        start = 0
        for i in range(n_nodes):
            size = base + (1 if i < rem else 0)
            end = start + size
            shards_x.append(X[start:end])
            shards_y.append(y[start:end])
            start = end
        return shards_x, shards_y

    @staticmethod
    def init_primal_dual(
        model_init_fn,
        n_nodes: int,
        undirected_edges: List[Tuple[int, int]],
        key: jax.Array,
    ):
        directed = []
        for i, j in undirected_edges:
            directed.append((i, j))
            directed.append((j, i))

        w_list, state_list = [], []
        for _ in range(n_nodes):
            key, subkey = jr.split(key)
            m, s = eqx.nn.make_with_state(model_init_fn)(subkey)
            w_list.append(m)
            state_list.append(s)

        nu = {}
        for i, j in directed:
            params_i = eqx.filter(w_list[i], eqx.is_inexact_array)
            nu[(i, j)] = tree_map(lambda x: jnp.zeros_like(x), params_i)

        return w_list, state_list, nu, directed, key

    def local_vanilla_train(
        self,
        w_i,
        state_i,
        optimizer,
        nu_ijs: List[Dict],
        w_neighbors: List[eqx.Module],
        X_local: jnp.ndarray,
        y_local: jnp.ndarray,
        key: jax.Array,
    ):
        # 90/10 split
        N = X_local.shape[0]
        split = max(1, int(0.9 * N))
        X_tr, X_val = X_local[:split], X_local[split:]
        y_tr, y_val = y_local[:split], y_local[split:]

        def vanilla_loss(model, state, xb, yb, k):
            base, new_state = self.loss_fn(model, state, xb, yb, k)
            params_i = eqx.filter(model, eqx.is_inexact_array)
            penalty = 0.0
            for nu_ij, w_j in zip(nu_ijs, w_neighbors):
                params_j = eqx.filter(w_j, eqx.is_inexact_array)

                def per_leaf(nu_l, pi_l, pj_l):
                    return jnp.vdot(nu_l, pi_l - pj_l)

                penalty += jax.tree_util.tree_reduce(
                    lambda a, b: a + b,
                    jax.tree_util.tree_map(per_leaf, nu_ij, params_i, params_j),
                    initializer=0.0,
                )
            return base + penalty, new_state

        opt_state = optimizer.init(eqx.filter(w_i, eqx.is_inexact_array))

        if self.dp_config is None:
            w_best, s_best, train_hist, val_hist = eqx_train(
                w_i,
                state_i,
                opt_state,
                optimizer,
                vanilla_loss,
                X_tr,
                y_tr,
                X_val,
                y_val,
                batch_size=self.batch_size or X_tr.shape[0],
                num_epochs=self.inner_epochs,
                patience=self.patience,
                key=key,
            )
        else:
            w_best, s_best, train_hist, val_hist = dp_eqx_train(
                w_i,
                state_i,
                opt_state,
                optimizer,
                vanilla_loss,
                X_tr,
                y_tr,
                X_val,
                y_val,
                dp_config=self.dp_config,
                batch_size=self.batch_size or X_tr.shape[0],
                num_epochs=self.inner_epochs,
                patience=self.patience,
                key=key,
            )
        return w_best, s_best, train_hist, val_hist

    def train(self, X_train, y_train, X_test, y_test):
        X_shards, y_shards = self.shard_data(X_train, y_train, self.n_nodes)
        w_list, state_list, nu, directed_edges, key = self.init_primal_dual(
            self.model_init_fn, self.n_nodes, self.undirected_edges, self.key
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=self.lr, weight_decay=self.weight_decay),
        )

        self.train_histories = []
        self.val_histories = []
        self.test_losses = []

        for rnd in range(self.outer_rounds):
            key, *subkeys = jr.split(key, self.n_nodes + 1)

            new_w, new_s = [], []
            round_tr, round_val = [], []

            # --- primal updates ---
            for i in range(self.n_nodes):
                nu_ijs = [
                    nu[(i, j)] for (i_, j) in directed_edges if i_ == i and j != i
                ]
                neighbors = [
                    w_list[j] for (i_, j) in directed_edges if i_ == i and j != i
                ]

                w_i_new, s_i_new, th, vh = self.local_vanilla_train(
                    w_list[i],
                    state_list[i],
                    optimizer,
                    nu_ijs,
                    neighbors,
                    X_shards[i],
                    y_shards[i],
                    subkeys[i],
                )
                new_w.append(w_i_new)
                new_s.append(s_i_new)
                round_tr.append(th)
                round_val.append(vh)

            w_list, state_list = new_w, new_s
            self.train_histories.append(round_tr)
            self.val_histories.append(round_val)

            # --- dual updates ---
            for i, j in self.undirected_edges:
                pi = eqx.filter(w_list[i], eqx.is_inexact_array)
                pj = eqx.filter(w_list[j], eqx.is_inexact_array)
                nu_ij = tree_map(
                    lambda ν, a, b: ν + self.gamma * (a - b), nu[(i, j)], pi, pj
                )
                nu[(i, j)] = nu_ij
                nu[(j, i)] = tree_map(lambda x: -x, nu_ij)

            # --- eval on node 0 ---
            loss = float(
                eval_step(
                    w_list[0],
                    state_list[0],
                    X_test,
                    y_test,
                    jr.split(key)[1],
                    self.loss_fn,
                )
            )
            self.test_losses.append(loss)
            print(f"Round {rnd+1}/{self.outer_rounds} — Test Loss: {loss:.4f}")

        self.w_list = w_list
        self.state_list = state_list
        self.nu = nu
        return w_list, nu, self.test_losses

    def _predict_node(
        self, w: eqx.Module, state: Optional[dict], X: jnp.ndarray, key: jax.Array
    ) -> jnp.ndarray:
        """
        Compute raw output (logits) for a single node.
        """
        return predict(w, state, X, key)

    def predict_proba(
        self,
        X: jnp.ndarray,
        *,
        node: Optional[int] = None,
        key: Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        """
        Return probabilities.
          - If node is None: ensemble average over all nodes.
          - Otherwise: per-node probabilities.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, len(self.w_list))

        probs = []
        for i, (w_i, s_i, k_i) in enumerate(zip(self.w_list, self.state_list, keys)):
            logits = self._predict_node(w_i, s_i, X, k_i)
            if logits.shape[-1] == 1:
                p = jax.nn.sigmoid(logits).squeeze(-1)
            else:
                p = jax.nn.softmax(logits, axis=-1)
            probs.append(p)

        probs = jnp.stack(probs, axis=0)

        if node is None:
            return jnp.mean(probs, axis=0)
        else:
            return probs[node]

    def predict(
        self,
        X: jnp.ndarray,
        *,
        node: Optional[int] = None,
        threshold: float = 0.5,
        key: Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        """
        Return class predictions.
          - If binary: a (batch,) bool array: proba > threshold.
          - If multiclass: an (batch,) int array of argmax.
          - node: same semantics as in predict_proba.
        """
        proba = self.predict_proba(X, node=node, key=key)
        if proba.ndim == 1:
            return (proba > threshold).astype(jnp.int32)
        else:
            return jnp.argmax(proba, axis=-1)


if __name__ == "__main__":
    import numpy as onp
    import matplotlib.pyplot as plt

    num_samples, feature_dim = 800, 6
    rng = onp.random.RandomState(0)
    X_np = rng.randn(num_samples, feature_dim).astype(onp.float32)
    true_w = rng.randn(feature_dim, 1)
    logits = X_np @ true_w
    probs = 1 / (1 + onp.exp(-logits))
    y_np = (rng.rand(num_samples, 1) < probs).astype(onp.float32)

    X = jnp.array(X_np)
    y = jnp.array(y_np)
    split = int(0.8 * num_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    class SimpleLR(eqx.Module):
        lin: eqx.nn.Linear

        def __init__(self, key):
            k1, _ = jr.split(key)
            self.lin = eqx.nn.Linear(feature_dim, 1, key=k1)

        def __call__(self, x, key=None, state=None):
            return self.lin(x), state

    trainer = DecentralizedTrainer(
        model_init_fn=SimpleLR,
        undirected_edges=[(0, 1), (1, 2), (2, 3)],  # a line of 4 nodes
        n_nodes=4,
        outer_rounds=8,
        inner_epochs=8,
        gamma=0.02,
        lr=5e-3,
        weight_decay=1e-4,
        patience=3,
        batch_size=64,
        key=jr.PRNGKey(1),
    )
    w_list, nu, test_losses = trainer.train(X_train, y_train, X_test, y_test)
    print("Test losses per round:", [float(v) for v in test_losses])

    # plot loss vs round
    plt.figure(figsize=(7.2, 4.0))
    plt.plot(np.arange(1, len(test_losses) + 1), test_losses, linewidth=2)
    plt.yscale("log")
    plt.xlabel("Round")
    plt.ylabel("Test loss (node 0)")
    plt.title("Decentralized dual-ascent — loss curve")
    plt.tight_layout()
    plt.show()

    # quick accuracy report
    key_pred = jr.PRNGKey(42)
    preds_node_0 = trainer.predict(X_test, node=0, key=key_pred)
    y1 = jnp.squeeze(y_test, axis=-1)
    acc0 = float((preds_node_0 == y1).mean())
    print(f"[Decentralized] node 0 test acc: {acc0*100:.2f}%")
