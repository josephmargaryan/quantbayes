# quantbayes/stochax/distributed_training/centralized.py
from typing import Callable, Optional, List, Tuple, Any

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

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
    """
    Centralized trainer with optional DP-SGD.

    Best-practice notes:
    - Validation data must be supplied explicitly (no in-training reuse of train as val).
    - Optimizer uses AdamW (decoupled weight decay) + gradient clipping.
    - For DP training, we wrap the loss with an L2 penalty so DP/non-DP regularization semantics match.
      (Non-DP path can still rely on your existing `eqx_train(..., lambda_spec=...)` if that
       implements a custom/spectral penalty internally.)
    """

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
        loss_fn: Callable = binary_loss,
        dp_config: Optional[DPSGDConfig] = None,
    ):
        self.model_init_fn = model_init_fn
        self.epochs = int(epochs)
        self.batch_size = batch_size
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.lambda_spec = float(lambda_spec)
        self.patience = int(patience)
        self.key = key if key is not None else jr.PRNGKey(0)
        self.loss_fn = loss_fn
        self.dp_config = dp_config

        self.train_history: List[float] = []
        self.val_history: List[float] = []
        self.test_loss: Optional[float] = None
        self.model: Optional[eqx.Module] = None
        self.state: Optional[Any] = None

    # --------------------------- utilities ---------------------------

    @staticmethod
    def _l2_penalty(params: eqx.Module) -> jnp.ndarray:
        """Compute 0.5 * ||θ||^2 over inexact parameter leaves."""
        leaves = jax.tree_util.tree_leaves(eqx.filter(params, eqx.is_inexact_array))
        if not leaves:
            return jnp.array(0.0, dtype=jnp.float32)
        sq_sum = sum([jnp.sum(jnp.square(p)) for p in leaves])
        return 0.5 * sq_sum

    def _reg_wrapped_loss(self, base_loss_fn: Callable) -> Callable:
        """
        Wrap loss_fn to add L2 penalty with coefficient self.lambda_spec.
        (Used for DP path to mirror non-DP regularization semantics.)
        """
        lam = self.lambda_spec

        if lam <= 0.0:
            return base_loss_fn

        def loss_with_l2(
            m: eqx.Module, s: Any, xb: jnp.ndarray, yb: jnp.ndarray, k: jax.Array
        ):
            base, new_s = base_loss_fn(m, s, xb, yb, k)
            reg = lam * self._l2_penalty(m)
            return base + reg, new_s

        return loss_with_l2

    # ----------------------------- API ------------------------------

    def train(
        self,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        X_val: jnp.ndarray,
        y_val: jnp.ndarray,
        X_test: jnp.ndarray,
        y_test: jnp.ndarray,
    ) -> Tuple[eqx.Module, Any, float]:
        """
        Train with explicit train/val sets; evaluate on test.
        Returns (best_model, best_state, test_loss).
        """
        # Init model + state
        self.key, init_key = jr.split(self.key)
        model, state = eqx.nn.make_with_state(self.model_init_fn)(init_key)

        # Optimizer (decoupled weight decay)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=self.learning_rate, weight_decay=self.weight_decay
            ),
        )
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        # Effective batch size
        bsz = self.batch_size or int(X_train.shape[0])

        if self.dp_config is None:
            # Non-DP path; keep legacy behavior (pass lambda_spec through)
            model_best, state_best, train_hist, val_hist = eqx_train(
                model,
                state,
                opt_state,
                optimizer,
                self.loss_fn,
                X_train,
                y_train,
                X_val,
                y_val,
                batch_size=bsz,
                num_epochs=self.epochs,
                patience=self.patience,
                key=self.key,
                lambda_spec=self.lambda_spec,
            )
        else:
            # DP path; wrap loss with L2 penalty to match non-DP regularization semantics
            reg_loss_fn = self._reg_wrapped_loss(self.loss_fn)

            model_best, state_best, train_hist, val_hist = dp_eqx_train(
                model,
                state,
                opt_state,
                optimizer,
                reg_loss_fn,
                X_train,
                y_train,
                X_val,
                y_val,
                dp_config=self.dp_config,
                batch_size=bsz,
                num_epochs=self.epochs,
                patience=self.patience,
                key=self.key,
            )

        # Test evaluation
        self.key, eval_key = jr.split(self.key)
        test_loss = float(
            eval_step(model_best, state_best, X_test, y_test, eval_key, self.loss_fn)
        )

        # Persist
        self.model = model_best
        self.state = state_best
        self.train_history = train_hist
        self.val_history = val_hist
        self.test_loss = test_loss

        print(f"Centralized Training Complete | Final Test Loss: {test_loss:.4f}")
        return model_best, state_best, test_loss

    # --------------------------- inference --------------------------

    def _predict_raw(
        self, X: jnp.ndarray, key: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """
        Return raw logits from the trained model.
        """
        if self.model is None or self.state is None:
            raise RuntimeError("Must call .train() before ._predict_raw()")
        if key is None:
            key = jr.PRNGKey(0)
        return predict(self.model, self.state, X, key)

    def predict_proba(
        self, X: jnp.ndarray, *, key: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """
        Return probabilities for X:
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


# ------------------------------- Demo -------------------------------
# Smoke test that exercises BOTH non-DP and DP paths with a proper
# train/val/test split. Safe to comment out if you don't need it.
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    from quantbayes.stochax.privacy.dp import DPSGDConfig

    # ----- synthetic binary data -----
    N, D = 2000, 12
    rng = np.random.RandomState(0)
    X_np = rng.randn(N, D).astype(np.float32)
    true_w = rng.randn(D, 1) / np.sqrt(D)
    logits = X_np @ true_w
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_np = (rng.rand(N, 1) < probs).astype(np.float32)

    # splits: 70% train, 15% val, 15% test
    n_train = int(0.70 * N)
    n_val = int(0.15 * N)
    X_tr_np, y_tr_np = X_np[:n_train], y_np[:n_train]
    X_val_np, y_val_np = (
        X_np[n_train : n_train + n_val],
        y_np[n_train : n_train + n_val],
    )
    X_te_np, y_te_np = X_np[n_train + n_val :], y_np[n_train + n_val :]

    # standardize using train stats only
    mu = X_tr_np.mean(axis=0, keepdims=True)
    sd = X_tr_np.std(axis=0, keepdims=True) + 1e-8
    X_tr_np = (X_tr_np - mu) / sd
    X_val_np = (X_val_np - mu) / sd
    X_te_np = (X_te_np - mu) / sd

    X_tr = jnp.array(X_tr_np)
    y_tr = jnp.array(y_tr_np)
    X_val = jnp.array(X_val_np)
    y_val = jnp.array(y_val_np)
    X_te = jnp.array(X_te_np)
    y_te = jnp.array(y_te_np)

    # ----- tiny MLP -----
    class SimpleMLP(eqx.Module):
        fc1: eqx.nn.Linear
        fc2: eqx.nn.Linear

        def __init__(self, key):
            k1, k2 = jr.split(key)
            self.fc1 = eqx.nn.Linear(D, 32, key=k1)
            self.fc2 = eqx.nn.Linear(32, 1, key=k2)

        def __call__(self, x, key=None, state=None):
            x = jax.nn.relu(self.fc1(x))
            return self.fc2(x), state

    def model_init(k):
        return SimpleMLP(k)

    # ----- Non-DP -----
    trainer = CentralizedTrainer(
        model_init_fn=model_init,
        epochs=12,
        batch_size=128,
        learning_rate=2e-3,
        weight_decay=1e-4,
        lambda_spec=0.0,
        patience=3,
        key=jr.PRNGKey(42),
        dp_config=None,
    )
    model, state, test_loss = trainer.train(X_tr, y_tr, X_val, y_val, X_te, y_te)
    print(f"[Non-DP] Final Test Loss: {test_loss:.4f}")

    # ----- DP -----
    dp_cfg = DPSGDConfig(clipping_norm=1.0, noise_multiplier=1.0, delta=1e-5)
    trainer_dp = CentralizedTrainer(
        model_init_fn=model_init,
        epochs=12,
        batch_size=128,
        learning_rate=2e-3,
        weight_decay=1e-4,
        lambda_spec=0.0,  # If >0, DP path applies loss-level L2 of strength lambda_spec
        patience=3,
        key=jr.PRNGKey(12345),
        dp_config=dp_cfg,
    )
    model_dp, state_dp, test_loss_dp = trainer_dp.train(
        X_tr, y_tr, X_val, y_val, X_te, y_te
    )
    print(
        f"[DP]     Final Test Loss: {test_loss_dp:.4f} "
        f"(C={dp_cfg.clipping_norm}, σ={dp_cfg.noise_multiplier}, δ={dp_cfg.delta})"
    )

    # ----- plot both -----
    import numpy as np

    plt.figure(figsize=(7.6, 4.4))
    ep_base = np.arange(1, len(trainer.train_history) + 1)
    ep_dp = np.arange(1, len(trainer_dp.train_history) + 1)

    if len(ep_base):
        plt.plot(ep_base, trainer.train_history, label="train (non-DP)", linewidth=2)
        plt.plot(ep_base, trainer.val_history, label="val (non-DP)", linewidth=2)
    if len(ep_dp):
        plt.plot(
            ep_dp,
            trainer_dp.train_history,
            label="train (DP)",
            linewidth=2,
            linestyle="--",
        )
        plt.plot(
            ep_dp, trainer_dp.val_history, label="val (DP)", linewidth=2, linestyle="--"
        )

    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Centralized training: non-DP vs DP")
    plt.legend()
    plt.tight_layout()
    plt.show()
