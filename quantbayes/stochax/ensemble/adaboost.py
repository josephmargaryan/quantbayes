import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import numpy as np
from typing import Callable, List, Tuple

from quantbayes.stochax.trainer.train import *

__all__ = ["AdaBoost"]

# -------------------------------------------------------------------
# Utility: draw indices with replacement according to a probability vector
# -------------------------------------------------------------------
def weighted_sample_indices(
    key: jax.random.PRNGKey, weights: jnp.ndarray, num_samples: int
) -> jnp.ndarray:
    """
    Draw `num_samples` indices from {0, ..., num_samples-1} with replacement,
    according to the probability distribution `weights`.
    """
    # jax.random.choice can sample with replacement when shape=(num_samples,)
    return jr.choice(key, a=num_samples, shape=(num_samples,), p=weights)


# -------------------------------------------------------------------
# AdaBoostEquinox: trains B weak learners in sequence, updating sample weights
# -------------------------------------------------------------------
class AdaBoost:
    """
    AdaBoost implemented on top of Equinox + Optax.

    Each weak learner is trained on a weighted bootstrap sample. After training,
    we measure its weighted error on the *entire* training set, compute α, update
    sample weights, and store (model, state, α).

    At inference time, we sum α_b * sign(logit_b(x)) over all weak learners.
    """

    def __init__(
        self,
        model_constructor: Callable[[jax.random.PRNGKey], eqx.Module],
        loss_fn: Callable,
        optimizer: optax.GradientTransformation,
        num_estimators: int = 10,
        batch_size: int = 64,
        num_epochs: int = 50,
        patience: int = 5,
    ):
        """
        Parameters
        ----------
        model_constructor : function
            Takes a JAX PRNGKey and returns a freshly initialized Equinox Module
            that outputs a *single logit* per example (for binary classification).
        loss_fn : function
            Your existing binary‐classification loss (e.g. `binary_loss`). It must
            follow the signature (model, state, x, y, key) -> (loss, new_state).
        optimizer : optax.GradientTransformation
            Optimizer (e.g. optax.adam(...)) for training each weak learner.
        num_estimators : int
            Number of AdaBoost iterations (B).
        batch_size : int
            Batch size when training each weak learner.
        num_epochs : int
            Maximum epochs for each weak learner.
        patience : int
            Early‐stopping patience for each weak learner.
        """
        self.model_constructor = model_constructor
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.B = num_estimators
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience

        # After fitting, this list will hold tuples (model, state, alpha)
        self.weak_learners: List[Tuple[eqx.Module, any, float]] = []

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        key: jax.random.PRNGKey,
    ):
        """
        Train the AdaBoost ensemble on (X, y). Here y must be ±1.

        Parameters
        ----------
        X : jnp.ndarray, shape (N, d)
            Training features.
        y : jnp.ndarray, shape (N,)
            Training labels in {−1, +1}.
        key : jax.random.PRNGKey
            PRNG key for reproducibility.
        """
        N = X.shape[0]
        # 1) Initialize sample weights uniformly
        w = jnp.ones(N) / N

        # 2) Split the top‐level key into B subkeys, one per iteration
        keys = jr.split(key, self.B)

        for b in range(self.B):
            kb = keys[b]
            # 3) Bootstrap sample (with replacement) according to current w
            kb, ksamp = jr.split(kb)
            indices = weighted_sample_indices(ksamp, w, N)  # shape (N,)

            X_sample = X[indices]
            y_sample = y[indices]

            # 4) Build and train a fresh weak learner on (X_sample, y_sample)
            kb, kmodel = jr.split(kb)
            model = self.model_constructor(kmodel)
            state = None  # assume stateless; if your model tracks BatchNorm, modify accordingly
            opt_state = self.optimizer.init(eqx.filter(model, eqx.is_inexact_array))

            best_model, best_state, _, _ = train(
                model=model,
                state=state,
                opt_state=opt_state,
                optimizer=self.optimizer,
                loss_fn=self.loss_fn,
                # We pass X_sample as BOTH training and validation so that
                # early stopping is just on the sampled data itself.
                X_train=X_sample,
                y_train=y_sample,
                X_val=X_sample,
                y_val=y_sample,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                patience=self.patience,
                key=kmodel,
            )

            # 5) Compute predictions h_b on the *entire* training set X
            kb, kpred = jr.split(kb)
            logits = predict(best_model, best_state, X, kpred).ravel()  # shape (N,)
            h = jnp.sign(logits)
            # If a logit happens to be exactly zero, map it to +1:
            h = jnp.where(h == 0, 1, h)

            # 6) Compute weighted error: ε_b = ∑_i w_i * 1{h(x_i) ≠ y_i}
            incorrect = (h != y).astype(jnp.float32)
            epsilon = jnp.sum(w * incorrect)

            # 7) If ε_b > 0.5, flip h (so error becomes 1−ε)
            if epsilon > 0.5:
                h = -h
                epsilon = 1.0 - epsilon

            # 8) Compute α_b = ½⋅ln((1−ε)/ε)
            # Add 1e-15 in the denominator to avoid division by zero
            alpha = 0.5 * jnp.log((1.0 - epsilon) / (epsilon + 1e-15))

            # 9) Update sample weights: w_i ← w_i * exp(−α⋅y_i⋅h_i)
            w = w * jnp.exp(-alpha * y * h)
            w = w / jnp.sum(w)

            # 10) Store (weak learner, its state, α_b)
            self.weak_learners.append((best_model, best_state, float(alpha)))

            print(f"Weak learner {b+1}/{self.B} – ε = {epsilon:.4f}, α = {alpha:.4f}")

    def predict(self, X: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Make predictions on new data via sign(∑_b α_b ⋅ h_b(x)).
        
        Returns a NumPy array in {−1, +1}.
        """
        M = X.shape[0]
        agg = jnp.zeros(M)

        # Split key for each weak learner (for any stochastic layers at inference)
        keys = jr.split(key, len(self.weak_learners))

        for (model, state, alpha), kb in zip(self.weak_learners, keys):
            logits = predict(model, state, X, kb).ravel()  # shape (M,)
            h = jnp.sign(logits)
            h = jnp.where(h == 0, 1, h)
            agg = agg + alpha * h

        final_pred = jnp.sign(agg)
        final_pred = jnp.where(final_pred == 0, 1, final_pred)
        return np.array(final_pred)  # convert to CPU‐memory NumPy


# -------------------------------------------------------------------
# Example Usage (binary classification with synthetic data)
# -------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # 1) Create synthetic data (labels in {0,1}), then map to {−1,+1}
    X_np, y_np = make_classification(
        n_samples=1000, n_features=10, n_informative=5, random_state=0
    )
    y_np = 2 * y_np - 1  # convert 0→−1, 1→+1

    # 2) Train/test split
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np, y_np, test_size=0.2, random_state=0
    )

    # 3) Convert to JAX arrays
    X_train = jnp.array(X_train_np, dtype=jnp.float32)
    y_train = jnp.array(y_train_np, dtype=jnp.float32)
    X_test = jnp.array(X_test_np, dtype=jnp.float32)
    y_test = jnp.array(y_test_np, dtype=jnp.float32)

    # 4) Define a very simple weak learner: a one‐layer Equinox module that outputs a single logit
    class SimpleWeakLearner(eqx.Module):
        linear: eqx.nn.Linear

        def __init__(self, in_features: int, key: jax.random.PRNGKey):
            self.linear = eqx.nn.Linear(in_features, 1, key=key)

        def __call__(
            self, x: jnp.ndarray, key: jax.random.PRNGKey, state
        ) -> Tuple[jnp.ndarray, any]:
            """
            Forward pass: return a scalar logit per example.
            `state` is unused since this is stateless.
            """
            logits = jnp.squeeze(self.linear(x), axis=-1)
            return logits, state

    def weak_model_constructor(key: jax.random.PRNGKey) -> eqx.Module:
        in_f = X_train.shape[-1]
        return SimpleWeakLearner(in_f, key)

    # 5) Plug in your existing `binary_loss` and choose an optimizer
    # (Assumes `binary_loss` returns (loss, new_state) and works on a single‐logit model.)
    loss_fn = binary_loss
    optimizer = optax.adam(learning_rate=1e-2)

    # 6) Instantiate AdaBoost and fit
    ada = AdaBoost(
        model_constructor=weak_model_constructor,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_estimators=10,
        batch_size=128,
        num_epochs=100,
        patience=10,
    )
    main_key = jr.PRNGKey(42)
    ada.fit(X_train, y_train, main_key)

    # 7) Predict on the test set
    test_key = jr.PRNGKey(2025)
    y_pred = ada.predict(X_test, test_key)

    # 8) Evaluate
    acc = accuracy_score(y_test_np, y_pred)
    print(f"AdaBoost Test Accuracy: {acc:.4f}")
