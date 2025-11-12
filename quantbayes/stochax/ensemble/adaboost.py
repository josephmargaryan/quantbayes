import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from jax.tree_util import tree_flatten, tree_unflatten
import numpy as np
from typing import Callable, List, Tuple, Optional
import matplotlib.pyplot as plt

from quantbayes.stochax.trainer.train import *

__all__ = ["AdaBoost"]


# -------------------------------------------------------------------
# 1) PER-SAMPLE WEIGHTED LOSS FUNCTIONS
# -------------------------------------------------------------------
def weighted_binary_loss_per_sample(logits: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute elementwise binary‐cross‐entropy (logistic) loss, given:
      - logits: shape [batch_size], raw scores (float)
      - y:      shape [batch_size], targets ∈ {0., 1.}
    Returns a vector of length batch_size with per-example losses.
    """
    # optax.sigmoid_binary_cross_entropy returns a vector of per-sample losses
    return optax.sigmoid_binary_cross_entropy(logits, y)


def weighted_multiclass_loss_per_sample(
    logits: jnp.ndarray, y: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute elementwise softmax‐cross‐entropy loss, given:
      - logits: shape [batch_size, K], raw scores
      - y:      shape [batch_size], integer labels ∈ {0,…,K−1}
    Returns a vector of length batch_size with per-example losses.
    """
    return optax.softmax_cross_entropy_with_integer_labels(logits, y)


# -------------------------------------------------------------------
# 2) AdaBoost CLASS (EXACT SAMPLE-WEIGHT, STAGED-ERROR TRACKING)
# -------------------------------------------------------------------
class AdaBoost:
    """
    AdaBoost ensemble that trains each weak learner on the full dataset
    using exact sample weights (no bootstrap).  This matches scikit-learn’s
    AdaBoostClassifier behavior.  It also records staged 0–1 train/val errors
    so you can plot convergence curves “à la” scikit-learn’s `staged_score`.
    """

    def __init__(
        self,
        model_constructor: Callable[[jax.random.PRNGKey], eqx.Module],
        num_estimators: int = 10,
        learning_rate: float = 1.0,
        batch_size: int = 64,
        num_epochs: int = 10,
        patience: int = 3,
        optimizer: optax.GradientTransformation = optax.adam(1e-2),
        *,
        num_classes: int = 2,
        rng_key: Optional[jax.random.PRNGKey] = None,
    ):
        """
        Parameters
        ----------
        model_constructor : function
            Given a JAX PRNGKey, returns a fresh Equinox Module.
            - If num_classes == 2: module outputs raw logits of shape (batch_size,) for binary classification.
            - If num_classes > 2: module outputs raw logits of shape (batch_size, K).
        num_estimators : int
            Number of boosting rounds (B).
        learning_rate : float
            Shrinkage factor on α_b (≤ 1.0). scikit-learn uses 1.0 by default.
        batch_size : int
            Mini-batch size when fitting each weak learner.
        num_epochs : int
            Maximum epochs per weak learner.
        patience : int
            Early stopping patience (in epochs) based on weighted loss not improving.
        optimizer : optax.GradientTransformation
            Optimizer for training each weak learner.
        num_classes : int
            If 2 → binary AdaBoost, targets y ∈ {0.,1.}. If >2 → SAMME multiclass, y ∈ {0,…,K−1}.
        rng_key : PRNGKey (optional)
            If provided, used for reproducibility; otherwise defaults to jr.PRNGKey(0).
        """
        self.model_constructor = model_constructor
        self.num_estimators = num_estimators
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.optimizer = optimizer
        self.num_classes = num_classes

        # List to store (trained_module, α_b) for each weak learner
        self.weak_learners: List[Tuple[eqx.Module, float]] = []

        # To track staged 0–1 errors
        self.train_errors: List[float] = []
        self.val_errors: List[float] = []

        # RNG key
        self.rng_key = jr.PRNGKey(0) if rng_key is None else rng_key

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        *,
        X_val: Optional[jnp.ndarray] = None,
        y_val: Optional[jnp.ndarray] = None,
    ):
        """
        Fit the AdaBoost ensemble on (X, y) using exact sample weights.

        - If num_classes == 2: y must be float32 ∈ {0., 1.}.
        - If num_classes  > 2: y must be int32 ∈ {0,…,K−1}, where K=num_classes.
        Optionally pass (X_val, y_val) to record validation errors per stage.
        """
        N = X.shape[0]
        # 1) Initialize sample weights uniformly
        w = jnp.ones(N) / N

        for b in range(self.num_estimators):
            # ─── 2) Instantiate a fresh weak learner ───
            self.rng_key, subkey = jr.split(self.rng_key)
            model = self.model_constructor(subkey)

            # Flatten the model into (leaves, treedef) so that we can gradient-update the leaves
            leaves, treedef = tree_flatten(
                model
            )  # :contentReference[oaicite:3]{index=3}
            params = leaves  # flat list of all leaves
            opt_state = self.optimizer.init(params)

            best_model = model
            best_loss = jnp.inf
            epochs_without_improve = 0

            # ─── 3) Train this weak learner with weighted loss ───
            for epoch in range(self.num_epochs):
                # 3a) Shuffle data each epoch
                self.rng_key, perm_key = jr.split(self.rng_key)
                perm = jr.permutation(perm_key, N)
                X_perm, y_perm, w_perm = X[perm], y[perm], w[perm]

                num_batches = int(jnp.ceil(N / self.batch_size))
                for i in range(num_batches):
                    start = i * self.batch_size
                    end = min((i + 1) * self.batch_size, N)
                    x_batch = X_perm[start:end]
                    y_batch = y_perm[start:end]
                    w_batch = w_perm[start:end]

                    # Provide a separate RNG for each example in the batch, if model is stochastic
                    self.rng_key, batch_key = jr.split(self.rng_key)
                    keys_batch = jr.split(batch_key, x_batch.shape[0])

                    # Define weighted loss on the batch
                    def loss_fn(params_, _state_, xb, yb, wb, keysb):
                        # Reconstruct module from flat leaves `params_`
                        model_ = tree_unflatten(
                            treedef, params_
                        )  # :contentReference[oaicite:4]{index=4}
                        logits, _ = jax.vmap(lambda xi, ki: model_(xi, ki, None))(
                            xb, keysb
                        )

                        if self.num_classes == 2:
                            logits1d = logits.squeeze()
                            per_example = weighted_binary_loss_per_sample(logits1d, yb)
                        else:
                            per_example = weighted_multiclass_loss_per_sample(
                                logits, yb
                            )

                        # Weighted average over the batch
                        weighted_loss = jnp.sum(per_example * wb) / jnp.sum(wb)
                        return weighted_loss, None

                    # Compute gradients w.r.t. `params`
                    (batch_loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                        params, None, x_batch, y_batch, w_batch, keys_batch
                    )
                    updates, opt_state = self.optimizer.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)

                # ─── After each epoch, compute weighted loss on the entire training set ───
                self.rng_key, eval_key = jr.split(self.rng_key)
                keys_full = jr.split(eval_key, N)
                logits_full, _ = jax.vmap(
                    lambda xi, ki: tree_unflatten(treedef, params)(
                        xi, ki, None
                    )  # rebuilt module
                )(X, keys_full)

                if self.num_classes == 2:
                    logits1d_full = logits_full.squeeze()
                    per_full = weighted_binary_loss_per_sample(logits1d_full, y)
                else:
                    per_full = weighted_multiclass_loss_per_sample(logits_full, y)

                full_loss = jnp.sum(per_full * w) / jnp.sum(w)

                # Early stopping
                if full_loss < best_loss - 1e-8:
                    best_loss = full_loss
                    best_model = tree_unflatten(treedef, params)
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1
                    if epochs_without_improve >= self.patience:
                        break

            # ─── 4) Evaluate ε_b on the full training set with best_model ───
            self.rng_key, pred_key = jr.split(self.rng_key)
            keys_pred = jr.split(pred_key, N)
            logits_best, _ = jax.vmap(lambda xi, ki: best_model(xi, ki, None))(
                X, keys_pred
            )

            if self.num_classes == 2:
                # Binary mode: convert logits → {−1, +1}
                logits1d_best = logits_best.squeeze()
                h_signed = jnp.sign(logits1d_best)
                h_signed = jnp.where(h_signed == 0, 1, h_signed)
                # Convert y ∈ {0,1} → y_signed ∈ {−1,+1}
                y_signed = 2 * y - 1
                incorrect = (h_signed != y_signed).astype(jnp.float32)
                epsilon = jnp.sum(w * incorrect)
                # If ε_b > 0.5, flip hypothesis so error becomes 1−ε_b
                if epsilon > 0.5:
                    h_signed = -h_signed
                    epsilon = 1.0 - epsilon
                alpha = 0.5 * jnp.log((1.0 - epsilon) / (epsilon + 1e-15))
                # Update sample weights: w_i ← w_i · exp(−α·y_i·h_i)
                w = w * jnp.exp(-alpha * y_signed * h_signed)
                w = w / jnp.sum(w)

            else:
                # SAMME multiclass
                K = self.num_classes
                preds = jnp.argmax(logits_best, axis=1)
                incorrect = (preds != y).astype(jnp.float32)
                epsilon = jnp.sum(w * incorrect)
                # Skip if no better than random
                if epsilon >= (K - 1) / K:
                    print(
                        f"Weak learner {b+1} skipped (ε={epsilon:.4f} ≥ {(K-1)/K:.4f})"
                    )
                    continue
                alpha = jnp.log((1.0 - epsilon) / (epsilon + 1e-15)) + jnp.log(K - 1.0)
                correct_mask = preds == y
                w = w * jnp.where(correct_mask, jnp.exp(-alpha), jnp.exp(alpha))
                w = w / jnp.sum(w)

            # ─── 5) Store (best_model, α_b) ───
            self.weak_learners.append((best_model, float(alpha)))
            print(
                f"Weak learner {b+1}/{self.num_estimators}  –  ε = {epsilon:.4f},  α = {alpha:.4f}"
            )

            # ─── 6) Record staged training error (0–1) after b+1 rounds ───
            train_pred = self._staged_predict(X, upto=b + 1)
            train_err = float(jnp.mean((train_pred != y).astype(jnp.float32)))
            self.train_errors.append(train_err)

            # If validation set supplied, record validation error
            if X_val is not None and y_val is not None:
                val_pred = self._staged_predict(X_val, upto=b + 1)
                val_err = float(jnp.mean((val_pred != y_val).astype(jnp.float32)))
                self.val_errors.append(val_err)

    def _staged_predict(self, X: jnp.ndarray, upto: int) -> jnp.ndarray:
        """
        Return 0–1 predictions using only the first `upto` weak learners.
        - If num_classes == 2: returns array of shape (N,) in {0,1}.
        - If num_classes  > 2: returns array of shape (N,) in {0,…,K−1}.
        """
        N = X.shape[0]
        if self.num_classes == 2:
            agg = jnp.zeros(N)
            for model, alpha in self.weak_learners[:upto]:
                self.rng_key, pred_key = jr.split(self.rng_key)
                keys_full = jr.split(pred_key, N)
                logits, _ = jax.vmap(lambda xi, ki: model(xi, ki, None))(X, keys_full)
                logits1d = logits.squeeze()
                h_signed = jnp.sign(logits1d)
                h_signed = jnp.where(h_signed == 0, 1, h_signed)
                agg = agg + alpha * h_signed

            final_signed = jnp.sign(agg)
            final_signed = jnp.where(final_signed == 0, 1, final_signed)
            # Convert back to {0,1}
            return ((final_signed + 1) // 2).astype(jnp.int32)

        else:
            K = self.num_classes
            scores = jnp.zeros((N, K))
            for model, alpha in self.weak_learners[:upto]:
                self.rng_key, pred_key = jr.split(self.rng_key)
                keys_full = jr.split(pred_key, N)
                logits, _ = jax.vmap(lambda xi, ki: model(xi, ki, None))(X, keys_full)
                preds = jnp.argmax(logits, axis=1)
                onehot_preds = jax.nn.one_hot(preds, K)
                scores = scores + alpha * onehot_preds
            return jnp.argmax(scores, axis=1)

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Final ensemble prediction using all trained weak learners.
        Returns same shape/type as `_staged_predict`.
        """
        return self._staged_predict(X, upto=len(self.weak_learners))


# -------------------------------------------------------------------
# 3) EXAMPLE USAGE + CONVERGENCE PLOT
# -------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # 3.1) Create synthetic binary data (labels ∈ {0,1})
    X_np, y_np = make_classification(
        n_samples=1000, n_features=10, n_informative=5, random_state=0
    )

    # 3.2) Train/test split
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np, y_np, test_size=0.2, random_state=0
    )

    # 3.3) Convert to JAX arrays
    X_train = jnp.array(X_train_np, dtype=jnp.float32)
    y_train = jnp.array(y_train_np, dtype=jnp.int32)
    X_test = jnp.array(X_test_np, dtype=jnp.float32)
    y_test = jnp.array(y_test_np, dtype=jnp.int32)

    # 3.4) Define a simple weak learner: a one-layer linear module
    class SimpleWeakLearner(eqx.Module):
        linear: eqx.nn.Linear

        def __init__(self, in_features: int, key: jax.random.PRNGKey):
            self.linear = eqx.nn.Linear(in_features, 1, key=key)

        def __call__(self, x: jnp.ndarray, key: jax.random.PRNGKey, state):
            # Return a single logit per example
            return jnp.squeeze(self.linear(x)), state

    def weak_model_constructor(key: jax.random.PRNGKey) -> eqx.Module:
        in_f = X_train.shape[-1]
        return SimpleWeakLearner(in_f, key)

    # 3.5) Instantiate AdaBoost, using binary mode
    ada = AdaBoost(
        model_constructor=weak_model_constructor,
        num_estimators=10,
        learning_rate=1.0,
        batch_size=128,
        num_epochs=50,
        patience=5,
        optimizer=optax.adam(1e-2),
        num_classes=2,
        rng_key=jr.PRNGKey(42),
    )

    # 3.6) Fit on training data (no validation split here)
    ada.fit(X_train, y_train)

    # 3.7) Predict on test set & report accuracy
    y_pred = ada.predict(X_test)
    acc = accuracy_score(y_test_np, np.array(y_pred))
    print(f"\nAdaBoost Test Accuracy: {acc:.4f}")

    # 3.8) Plot convergence (training error vs. number of weak learners)
    stages = np.arange(1, len(ada.train_errors) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(stages, ada.train_errors, label="Training Error", marker="o")
    plt.xlabel("Number of Weak Learners")
    plt.ylabel("0–1 Error Rate")
    plt.title("AdaBoost Convergence (Training Error)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
