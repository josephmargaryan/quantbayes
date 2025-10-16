from typing import Callable, Optional, List, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import numpy as np

from quantbayes.stochax.trainer.train import (
    train as eqx_train,
    predict,
    binary_loss,
    eval_step,
    multiclass_loss,
)
from quantbayes.stochax.privacy.dp import DPSGDConfig
from quantbayes.stochax.privacy.dp_train import dp_eqx_train

__all__ = ["CentralizedTrainer"]


class CentralizedTrainer:
    def __init__(
        self,
        model_init_fn: Callable[[jax.Array], eqx.Module],
        epochs: int = 10,
        batch_size: Optional[int] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        lambda_spec: float = 0.0,
        patience: int = 5,
        key: Optional[jax.Array] = None,
        loss_fn=binary_loss,
        dp_config: Optional[DPSGDConfig] = None,  # NEW (default keeps old behavior)
    ):
        self.model_init_fn = model_init_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_spec = lambda_spec
        self.patience = patience
        self.key = key if key is not None else jr.PRNGKey(0)
        self.loss_fn = loss_fn
        self.dp_config = dp_config

        self.train_history: List[float] = []
        self.val_history: List[float] = []
        self.test_loss: Optional[float] = None
        self.model: Optional[eqx.Module] = None
        self.state: Optional[any] = None

    def train(
        self,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        X_test: jnp.ndarray,
        y_test: jnp.ndarray,
    ) -> Tuple[eqx.Module, any, float]:
        self.key, init_key = jr.split(self.key)
        model, state = eqx.nn.make_with_state(self.model_init_fn)(init_key)

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=self.learning_rate, weight_decay=self.weight_decay
            ),
        )
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        if self.dp_config is None:
            model_best, state_best, train_hist, val_hist = eqx_train(
                model,
                state,
                opt_state,
                optimizer,
                self.loss_fn,
                X_train,
                y_train,
                X_train,
                y_train,
                batch_size=self.batch_size or X_train.shape[0],
                num_epochs=self.epochs,
                patience=self.patience,
                key=self.key,
                lambda_spec=self.lambda_spec,
            )
        else:
            # DP training (same API; loss_fn unchanged)
            model_best, state_best, train_hist, val_hist = dp_eqx_train(
                model,
                state,
                opt_state,
                optimizer,
                self.loss_fn,
                X_train,
                y_train,
                X_train,
                y_train,
                dp_config=self.dp_config,
                batch_size=self.batch_size or X_train.shape[0],
                num_epochs=self.epochs,
                patience=self.patience,
                key=self.key,
            )

        self.key, eval_key = jr.split(self.key)
        test_loss = float(
            eval_step(model_best, state_best, X_test, y_test, eval_key, self.loss_fn)
        )

        self.model = model_best
        self.state = state_best
        self.train_history = train_hist
        self.val_history = val_hist
        self.test_loss = test_loss

        print(f"Centralized Training Complete | Final Test Loss: {test_loss:.4f}")
        return model_best, state_best, test_loss

    def _predict_raw(
        self, X: jnp.ndarray, key: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """
        Return raw logits from your trained model.
        """
        if self.model is None or self.state is None:
            raise RuntimeError("Must call .train() before .predict_raw()")
        if key is None:
            key = jr.PRNGKey(0)
        return predict(self.model, self.state, X, key)

    def predict_proba(
        self, X: jnp.ndarray, *, key: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """
        Return class‐probabilities for X:
          - Binary: shape (batch,) with P(y=1)
          - Multiclass: shape (batch, n_classes)
        """
        logits = self._predict_raw(X, key)
        if logits.shape[-1] == 1:
            return jax.nn.sigmoid(logits).squeeze(-1)
        return jax.nn.softmax(logits, axis=-1)

    def predict(
        self, X: jnp.ndarray, *, threshold: float = 0.5, key: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """
        Return hard predictions:
          - Binary: 0/1 via proba > threshold
          - Multiclass: integer argmax
        """
        proba = self.predict_proba(X, key=key)
        if proba.ndim == 1:
            return (proba > threshold).astype(jnp.int32)
        return jnp.argmax(proba, axis=-1)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx

    num_samples, feature_dim = 2000, 12
    rng = np.random.RandomState(0)
    X_np = rng.randn(num_samples, feature_dim).astype(np.float32)
    true_w = rng.randn(feature_dim, 1) / np.sqrt(feature_dim)
    logits = X_np @ true_w
    probs = 1 / (1 + np.exp(-logits))
    y_np = (rng.rand(num_samples, 1) < probs).astype(np.float32)

    # standardize
    split = int(0.8 * num_samples)
    X_tr_np, X_te_np = X_np[:split], X_np[split:]
    y_tr_np, y_te_np = y_np[:split], y_np[split:]
    mu = X_tr_np.mean(axis=0, keepdims=True)
    sd = X_tr_np.std(axis=0, keepdims=True) + 1e-8
    X_tr_np = (X_tr_np - mu) / sd
    X_te_np = (X_te_np - mu) / sd

    X_tr = jnp.array(X_tr_np)
    y_tr = jnp.array(y_tr_np)
    X_te = jnp.array(X_te_np)
    y_te = jnp.array(y_te_np)

    class SimpleMLP(eqx.Module):
        fc1: eqx.nn.Linear
        fc2: eqx.nn.Linear

        def __init__(self, key):
            k1, k2 = jr.split(key)
            self.fc1 = eqx.nn.Linear(feature_dim, 32, key=k1)
            self.fc2 = eqx.nn.Linear(32, 1, key=k2)

        def __call__(self, x, key=None, state=None):
            x = jax.nn.relu(self.fc1(x))
            return self.fc2(x), state

    def model_init(k):
        return SimpleMLP(k)

    trainer = CentralizedTrainer(
        model_init_fn=model_init,
        epochs=12,
        batch_size=128,
        learning_rate=2e-3,
        weight_decay=1e-4,
        lambda_spec=0.0,
        patience=3,
        key=jr.PRNGKey(42),
    )
    model, state, test_loss = trainer.train(X_tr, y_tr, X_te, y_te)
    print("Final Test Loss:", trainer.test_loss)

    # ---- plot train/val loss curves ----
    epochs = np.arange(1, len(trainer.train_history) + 1)
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(epochs, trainer.train_history, label="train", linewidth=2)
    plt.plot(epochs, trainer.val_history, label="val", linewidth=2)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Centralized training loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
