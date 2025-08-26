import copy
from typing import Callable, Optional, List, Tuple
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

__all__ = ["FederatedTrainer"]


class FederatedTrainer:
    """
    Standard Federated Averaging (FedAvg) trainer.
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

        self.train_histories: List[List[List[float]]] = []
        self.val_histories: List[List[List[float]]] = []
        self.test_losses: List[float] = []

        self.global_model: Optional[eqx.Module] = None
        self.global_state: Optional[any] = None

    @staticmethod
    def shard_data(
        X: jnp.ndarray, y: jnp.ndarray, n_nodes: int
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """Evenly split X,y into n_nodes shards."""
        N = X.shape[0]
        base, rem = divmod(N, n_nodes)
        shards_x, shards_y = [], []
        idx = 0
        for i in range(n_nodes):
            size = base + (1 if i < rem else 0)
            shards_x.append(X[idx : idx + size])
            shards_y.append(y[idx : idx + size])
            idx += size
        return shards_x, shards_y

    @staticmethod
    def federated_avg(models: List[eqx.Module]) -> eqx.Module:
        """Average trainable parameters only."""
        param_trees = [eqx.filter(m, eqx.is_inexact_array) for m in models]
        avg_params = jax.tree_util.tree_map(lambda *a: sum(a) / len(a), *param_trees)
        _, static = eqx.partition(models[0], eqx.is_inexact_array)
        return eqx.combine(avg_params, static)

    def train(
        self,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        X_test: jnp.ndarray,
        y_test: jnp.ndarray,
    ) -> Tuple[eqx.Module, List[float]]:
        X_shards, y_shards = self.shard_data(X_train, y_train, self.n_nodes)

        self.key, init_key = jr.split(self.key)
        self.global_model, self.global_state = eqx.nn.make_with_state(
            self.model_init_fn
        )(init_key)

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
            ),
        )

        self.train_histories.clear()
        self.val_histories.clear()
        self.test_losses.clear()

        for rnd in range(1, self.outer_rounds + 1):
            self.key, *subkeys = jr.split(self.key, self.n_nodes + 1)

            round_tr, round_va = [], []
            local_models, local_states = [], []

            for i in range(self.n_nodes):
                lm = copy.deepcopy(self.global_model)
                ls = copy.deepcopy(self.global_state)
                opt_state = optimizer.init(eqx.filter(lm, eqx.is_inexact_array))

                lm_trained, ls_trained, tr_hist, va_hist = eqx_train(
                    lm,
                    ls,
                    opt_state,
                    optimizer,
                    self.loss_fn,
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

                local_models.append(lm_trained)
                local_states.append(ls_trained)
                round_tr.append(tr_hist)
                round_va.append(va_hist)

            self.global_model = self.federated_avg(local_models)
            self.global_state = local_states[0]

            self.key, eval_key = jr.split(self.key)
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

            self.train_histories.append(round_tr)
            self.val_histories.append(round_va)
            self.test_losses.append(loss)
            print(f"Round {rnd}/{self.outer_rounds} | Global Test Loss: {loss:.4f}")

        return self.global_model, self.test_losses

    def predict_proba(
        self, X: jnp.ndarray, *, key: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """
        Compute predicted probabilities using the global model.
        - For binary tasks returns shape (batch,) with sigmoid outputs.
        - For multiclass tasks returns shape (batch, num_classes) with softmax outputs.
        """
        if self.global_model is None or self.global_state is None:
            raise ValueError("Model has not been trained yet.")
        if key is None:
            key = jr.PRNGKey(0)
        logits = predict(self.global_model, self.global_state, X, key)
        if logits.shape[-1] == 1:
            return jax.nn.sigmoid(logits).squeeze(-1)
        else:
            return jax.nn.softmax(logits, axis=-1)

    def predict(
        self, X: jnp.ndarray, *, threshold: float = 0.5, key: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """
        Return discrete predictions.
        - Binary: dtype int32 array of 0/1 via `proba > threshold`.
        - Multiclass: argmax over softmax probabilities.
        """
        proba = self.predict_proba(X, key=key)
        if proba.ndim == 1:
            return (proba > threshold).astype(jnp.int32)
        else:
            return jnp.argmax(proba, axis=-1)


if __name__ == "__main__":
    # Synthetic example—3-node FedAvg
    import numpy as onp

    num_samples, feature_dim = 800, 5
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

    class SimpleMLP(eqx.Module):
        fc1: eqx.nn.Linear
        fc2: eqx.nn.Linear

        def __init__(self, key):
            k1, k2 = jr.split(key)
            self.fc1 = eqx.nn.Linear(feature_dim, 16, key=k1)
            self.fc2 = eqx.nn.Linear(16, 1, key=k2)

        def __call__(self, x, key=None, state=None):
            x = jax.nn.relu(self.fc1(x))
            return self.fc2(x), state

    trainer = FederatedTrainer(
        model_init_fn=SimpleMLP,
        n_nodes=3,
        outer_rounds=5,
        inner_epochs=3,
        batch_size=64,
        learning_rate=1e-2,
        weight_decay=1e-4,
        lambda_spec=0.0,
        patience=3,
        key=jr.PRNGKey(0),
    )
    final_model, test_losses = trainer.train(X_train, y_train, X_test, y_test)
    print("Final Test Losses:", test_losses)

    key_pred = jr.PRNGKey(42)
    preds = trainer.predict(X_test, key=key_pred)
    y_true = jnp.squeeze(y_test, axis=-1).astype(jnp.int32)
    accuracy = (preds == y_true).mean()
    print(f"Test Accuracy: {float(accuracy * 100):.2f}%")
