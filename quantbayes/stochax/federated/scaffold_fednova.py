# quantbayes/stochax/federated/scaffold_fednova.py
from __future__ import annotations
from typing import Callable, Optional, List, Any, Dict, Tuple
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
import optax

from quantbayes.stochax.trainer.train import binary_loss, eval_step
from quantbayes.stochax.utils.equinox_helpers import clone_module

Array = jnp.ndarray


# ---- small tree algebra ----
def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_sub(a, b):
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def tree_scale(a, s):
    return jax.tree_util.tree_map(lambda x: s * x, a)


def tree_zeros_like(a):
    return jax.tree_util.tree_map(jnp.zeros_like, a)


def tree_vdot(a, b):
    leaves = zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    return sum(jnp.vdot(x, y) for x, y in leaves)


# ---- local dataloader ----
def minibatches(X: Array, y: Array, batch_size: int, key: jax.Array):
    n = int(X.shape[0])
    idx = jnp.arange(n)
    key, sub = jr.split(key)
    perm = jr.permutation(sub, idx)
    for start in range(0, n, batch_size):
        sl = perm[start : start + batch_size]
        yield X[sl], y[sl]
    return


class ScaffoldFedNovaTrainer:
    """
    SCAFFOLD (control variates) + FedNova (step-count normalization).
    Toggles:
      - use_scaffold: apply gradient correction ∇f_i(θ) + (c - c_i)
      - use_fednova: aggregate normalized pseudo-gradients G_i = (θ_start - θ_end)/(τ_i * η_local)
      - server_lr: global step size on aggregated pseudo-gradient (only when use_fednova==True).
    """

    def __init__(
        self,
        model_init_fn: Callable[[jax.Array], eqx.Module],
        n_nodes: int = 8,
        outer_rounds: int = 25,
        inner_epochs: int = 1,
        batch_size: int = 256,
        lr_local: float = 1e-3,
        server_lr: float = 5e-3,
        weight_decay: float = 0.0,
        patience: int = 5,  # unused in this explicit loop; kept for API symmetry
        use_scaffold: bool = True,
        use_fednova: bool = True,
        key: Optional[jax.Array] = None,
        loss_fn=binary_loss,
    ):
        self.model_init_fn = model_init_fn
        self.n_nodes = int(n_nodes)
        self.R = int(outer_rounds)
        self.E = int(inner_epochs)
        self.batch = int(batch_size)
        self.lr_local = float(lr_local)
        self.server_lr = float(server_lr)
        self.weight_decay = float(weight_decay)
        self.use_scaffold = bool(use_scaffold)
        self.use_fednova = bool(use_fednova)
        self.key = jr.PRNGKey(0) if key is None else key
        self.loss_fn = loss_fn

        self.global_model: Optional[eqx.Module] = None
        self.global_state: Optional[Any] = None
        self.test_losses: List[float] = []

    @staticmethod
    def shard_even(X: Array, y: Array, n_nodes: int):
        N = X.shape[0]
        base, rem = divmod(N, n_nodes)
        Xs, ys, sizes = [], [], []
        s = 0
        for i in range(n_nodes):
            sz = base + (1 if i < rem else 0)
            Xs.append(X[s : s + sz])
            ys.append(y[s : s + sz])
            sizes.append(sz)
            s += sz
        return Xs, ys, sizes

    def _local_train_one(
        self,
        model: eqx.Module,
        state: Any,
        X: Array,
        y: Array,
        key: jax.Array,
        c_global_p,
        c_client_p,
    ) -> Tuple[eqx.Module, Any, int]:
        """Explicit local loop with (optional) SCAFFOLD correction."""
        params, static = eqx.partition(model, eqx.is_inexact_array)
        opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=self.lr_local, weight_decay=self.weight_decay),
        )
        opt_state = opt.init(params)
        steps = 0

        @eqx.filter_value_and_grad(has_aux=True)  # <<< FIX: has_aux=True
        def loss_and_grad(p, s, xb, yb, k):
            m = eqx.combine(p, static)
            base, new_s = self.loss_fn(m, s, xb, yb, k)  # base is scalar array
            if self.use_scaffold:
                # add linear term <p, (c - c_i)> so grad wrt p adds (c - c_i)
                offset = tree_sub(c_global_p, c_client_p)
                base = base + tree_vdot(p, offset)
            return base, new_s

        for _ in range(self.E):
            key, sub = jr.split(key)
            for xb, yb in minibatches(X, y, self.batch, sub):
                (loss_val, new_state), g = loss_and_grad(params, state, xb, yb, key)
                updates, opt_state = opt.update(g, opt_state, params)
                params = optax.apply_updates(params, updates)
                state = new_state
                steps += 1

        new_model = eqx.combine(params, static)
        return new_model, state, steps

    def train(
        self, X_train: Array, y_train: Array, X_test: Array, y_test: Array
    ) -> Tuple[eqx.Module, List[float]]:
        # shard
        Xs, ys, sizes = self.shard_even(X_train, y_train, self.n_nodes)

        # init global
        self.key, k0 = jr.split(self.key)
        self.global_model, self.global_state = eqx.nn.make_with_state(
            self.model_init_fn
        )(k0)
        p_g = eqx.filter(self.global_model, eqx.is_inexact_array)

        # SCAFFOLD control variates
        c_global = tree_zeros_like(p_g)
        c_clients = [tree_zeros_like(p_g) for _ in range(self.n_nodes)]

        self.test_losses = []

        for rnd in range(1, self.R + 1):
            local_models, local_states = [], []
            tau_list: List[int] = []
            delta_c_list: List[Any] = []  # for c_global update

            # per-client local training
            keys = jr.split(self.key, self.n_nodes + 1)
            self.key = keys[0]
            for i in range(self.n_nodes):
                # clones
                m_i = clone_module(self.global_model)
                s_i = clone_module(self.global_state)
                p_start = eqx.filter(m_i, eqx.is_inexact_array)

                # local train
                m_i_new, s_i_new, steps = self._local_train_one(
                    m_i,
                    s_i,
                    Xs[i],
                    ys[i],
                    keys[i + 1],
                    c_global_p=c_global,
                    c_client_p=c_clients[i],
                )
                p_end = eqx.filter(m_i_new, eqx.is_inexact_array)

                local_models.append(m_i_new)
                local_states.append(s_i_new)
                tau_list.append(int(max(1, steps)))

                # SCAFFOLD client update: c_i^new = c_i - c + (p_start - p_end) / (τ η)
                if self.use_scaffold:
                    delta_theta = tree_sub(p_start, p_end)
                    coef = 1.0 / (float(steps) * self.lr_local + 1e-12)
                    c_i_new = tree_add(
                        tree_sub(c_clients[i], c_global), tree_scale(delta_theta, coef)
                    )
                    delta_c = tree_sub(c_i_new, c_clients[i])
                    c_clients[i] = c_i_new
                    delta_c_list.append(delta_c)

            # Server aggregation
            if self.use_fednova:
                # FedNova: G_i = (p_start - p_end)/(τ_i * η_local); p_start == p_g for all clients in this round
                Gi_list = [
                    tree_scale(
                        tree_sub(
                            p_g, eqx.filter(local_models[i], eqx.is_inexact_array)
                        ),
                        1.0 / (tau_list[i] * self.lr_local + 1e-12),
                    )
                    for i in range(self.n_nodes)
                ]
                tot = float(sum(sizes))
                G = None
                for i in range(self.n_nodes):
                    Gi_w = tree_scale(Gi_list[i], float(sizes[i]) / (tot + 1e-12))
                    G = Gi_w if G is None else tree_add(G, Gi_w)
                # global update
                p_g = tree_sub(p_g, tree_scale(G, self.server_lr))
                _, static = eqx.partition(self.global_model, eqx.is_inexact_array)
                self.global_model = eqx.combine(p_g, static)
                self.global_state = local_states[0]  # nominal buffers
            else:
                # parameter weighted averaging (FedAvg)
                weights = [s / (sum(sizes) + 1e-12) for s in sizes]

                def wsum(*leaves):
                    return sum(w * leaf for w, leaf in zip(weights, leaves))

                avg_params = jax.tree_util.tree_map(
                    wsum, *[eqx.filter(m, eqx.is_inexact_array) for m in local_models]
                )
                _, static = eqx.partition(self.global_model, eqx.is_inexact_array)
                self.global_model = eqx.combine(avg_params, static)
                p_g = avg_params
                self.global_state = local_states[0]

            # SCAFFOLD global control update: c <- c + mean_i Δc_i
            if self.use_scaffold and delta_c_list:
                mean_delta_c = None
                for dc in delta_c_list:
                    mean_delta_c = (
                        dc if mean_delta_c is None else tree_add(mean_delta_c, dc)
                    )
                mean_delta_c = tree_scale(
                    mean_delta_c, 1.0 / max(1.0, float(len(delta_c_list)))
                )
                c_global = tree_add(c_global, mean_delta_c)

            # eval
            self.key, evk = jr.split(self.key)
            loss = float(
                eval_step(
                    self.global_model,
                    self.global_state,
                    X_test,
                    y_test,
                    evk,
                    binary_loss,
                )
            )
            self.test_losses.append(loss)
            print(f"[SCAFFOLD+FedNova] round {rnd}/{self.R} | test loss={loss:.4f}")

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

    # A) FedAvg baseline (no scaffold, no fednova)
    trainer_A = ScaffoldFedNovaTrainer(
        model_init_fn=model_init,
        n_nodes=8,
        outer_rounds=25,
        inner_epochs=1,
        batch_size=256,
        lr_local=1e-3,
        server_lr=5e-3,
        weight_decay=0.0,
        use_scaffold=False,
        use_fednova=False,
        key=jr.PRNGKey(0),
    )
    _, losses_A = trainer_A.train(Xtr, ytr, Xte, yte)
    histories["FedAvg"] = {"loss_node1": losses_A}

    # B) SCAFFOLD only
    trainer_B = ScaffoldFedNovaTrainer(
        model_init_fn=model_init,
        n_nodes=8,
        outer_rounds=25,
        inner_epochs=1,
        batch_size=256,
        lr_local=1e-3,
        server_lr=5e-3,
        weight_decay=0.0,
        use_scaffold=True,
        use_fednova=False,
        key=jr.PRNGKey(1),
    )
    _, losses_B = trainer_B.train(Xtr, ytr, Xte, yte)
    histories["SCAFFOLD"] = {"loss_node1": losses_B}

    # C) FedNova only
    trainer_C = ScaffoldFedNovaTrainer(
        model_init_fn=model_init,
        n_nodes=8,
        outer_rounds=25,
        inner_epochs=1,
        batch_size=256,
        lr_local=1e-3,
        server_lr=5e-3,
        weight_decay=0.0,
        use_scaffold=False,
        use_fednova=True,
        key=jr.PRNGKey(2),
    )
    _, losses_C = trainer_C.train(Xtr, ytr, Xte, yte)
    histories["FedNova"] = {"loss_node1": losses_C}

    # D) SCAFFOLD + FedNova
    trainer_D = ScaffoldFedNovaTrainer(
        model_init_fn=model_init,
        n_nodes=8,
        outer_rounds=25,
        inner_epochs=1,
        batch_size=256,
        lr_local=1e-3,
        server_lr=5e-3,
        weight_decay=0.0,
        use_scaffold=True,
        use_fednova=True,
        key=jr.PRNGKey(3),
    )
    _, losses_D = trainer_D.train(Xtr, ytr, Xte, yte)
    histories["SCAFFOLD+FedNova"] = {"loss_node1": losses_D}

    # ---- plots + summary ----
    os.makedirs("figs_scaffold_fednova", exist_ok=True)
    plot_global_loss_q3(
        histories,
        title="SCAFFOLD / FedNova variants",
        save="figs_scaffold_fednova/loss.png",
        style="accessible",
    )
    print("Saved figs_scaffold_fednova/loss.png")

    os.makedirs("tables", exist_ok=True)
    summary = summarize_histories(histories)
    print("\nNumeric summary (SCAFFOLD/FedNova):")
    print_publication_summary(summary, decimals=4)
    latex = latex_table_from_summary(
        summary,
        decimals=3,
        caption="SCAFFOLD and FedNova variants.",
        label="tab:scaffold_fednova",
    )
    with open("tables/scaffold_fednova_summary.tex", "w") as f:
        f.write(latex)
    print("Wrote tables/scaffold_fednova_summary.tex")
