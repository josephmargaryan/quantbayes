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
    predict,
    binary_loss,
    eval_step,
    multiclass_loss,
)

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

        # Attributes populated after .train()
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
        # Initialize model & state
        self.key, init_key = jr.split(self.key)
        model, state = eqx.nn.make_with_state(self.model_init_fn)(init_key)

        # Setup optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=self.learning_rate, weight_decay=self.weight_decay
            ),
        )
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        # Run training
        model_best, state_best, train_hist, val_hist = eqx_train(
            model,
            state,
            opt_state,
            optimizer,
            self.loss_fn,
            X_train,
            y_train,
            X_train,
            y_train,  # using train set for validation by default
            batch_size=self.batch_size or X_train.shape[0],
            num_epochs=self.epochs,
            patience=self.patience,
            key=self.key,
            lambda_spec=self.lambda_spec,
        )

        # Evaluate on test set
        self.key, eval_key = jr.split(self.key)
        test_loss = float(
            eval_step(model_best, state_best, X_test, y_test, eval_key, self.loss_fn)
        )

        # Store attributes
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
        # use stored key if none provided
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
        # binary classification
        if logits.shape[-1] == 1:
            return jax.nn.sigmoid(logits).squeeze(-1)
        # multiclass classification
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
        # binary
        if proba.ndim == 1:
            return (proba > threshold).astype(jnp.int32)
        # multiclass
        return jnp.argmax(proba, axis=-1)


if __name__ == "__main__":
    # Synthetic example
    num_samples, feature_dim = 1000, 5
    rng = np.random.RandomState(0)
    X_np = rng.randn(num_samples, feature_dim).astype(np.float32)
    true_w = rng.randn(feature_dim, 1)
    logits = X_np @ true_w
    probs = 1 / (1 + np.exp(-logits))
    y_np = (rng.rand(num_samples, 1) < probs).astype(np.float32)

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

    def model_init(k):
        return SimpleMLP(k)

    trainer = CentralizedTrainer(
        model_init_fn=model_init,
        epochs=10,
        batch_size=None,  # full-batch, or set to e.g. 64
        learning_rate=1e-2,
        weight_decay=0.0,
        lambda_spec=0.0,
        patience=3,
        key=jr.PRNGKey(42),
    )
    model, state, test_loss = trainer.train(X_train, y_train, X_test, y_test)
    print("Train Loss History:", trainer.train_history)
    print("Val   Loss History:", trainer.val_history)
    print("Test  Loss:", trainer.test_loss)
