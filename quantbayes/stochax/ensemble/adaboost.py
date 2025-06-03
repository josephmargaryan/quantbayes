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


def weighted_sample_indices(
    key: jax.random.PRNGKey, weights: jnp.ndarray, num_samples: int
) -> jnp.ndarray:
    """
    Draw `num_samples` indices from {0, ..., num_samples-1} with replacement,
    according to the probability distribution `weights`.
    """
    return jr.choice(key, a=num_samples, shape=(num_samples,), p=weights)


class AdaBoost:
    """
    AdaBoost that can handle:
      • Binary classification (num_classes = 2, labels ∈ {−1, +1})
      • Multiclass classification (num_classes = K > 2, labels ∈ {0,…,K−1})
      • (If you wanted, you could also plug in a regression loss, but standard
        AdaBoost is usually for classification.)

    Internally, we branch on self.num_classes == 2 vs. > 2 to pick the right α‐formula,
    weight update, and voting logic.
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
        *,
        num_classes: int = 2,
    ):
        """
        Parameters
        ----------
        model_constructor : function
            Takes a JAX PRNGKey and returns a freshly initialized Equinox Module.
            If num_classes == 2, this module must output a single scalar logit per example.
            If num_classes > 2, it must output a length‐K logit vector per example.
        loss_fn : function
            Either binary_loss (for num_classes == 2) or multiclass_loss (for num_classes > 2).
        optimizer : optax.GradientTransformation
            e.g. optax.adam(...)
        num_estimators : int
            Number of weak learners B.
        batch_size, num_epochs, patience : same as your existing train() parameters.
        num_classes : int
            If 2 → classic binary AdaBoost (labels ∈ {−1, +1}).
            If K > 2 → discrete multiclass AdaBoost (SAMME).
        """
        self.model_constructor = model_constructor
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.B = num_estimators
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.num_classes = num_classes

        # After fitting, we'll store a list of (model, state, alpha) tuples.
        self.weak_learners: List[Tuple[eqx.Module, any, float]] = []

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        key: jax.random.PRNGKey,
    ):
        """
        Train the AdaBoost ensemble on (X, y).

        For binary (num_classes=2), y must be in {−1, +1}.
        For multiclass (num_classes=K>2), y must be in {0,1,…,K−1}.
        """
        N = X.shape[0]
        # 1) Initialize sample weights uniformly
        w = jnp.ones(N) / N

        # 2) Split the top-level key into B subkeys
        keys = jr.split(key, self.B)

        for b in range(self.B):
            kb = keys[b]
            kb, ksamp = jr.split(kb)

            # 3) Weighted bootstrap sampling
            indices = weighted_sample_indices(ksamp, w, N)  # shape (N,)
            X_sample = X[indices]
            y_sample = y[indices]

            # 4) Build & train a fresh weak learner on (X_sample, y_sample)
            kb, kmodel = jr.split(kb)
            model = self.model_constructor(kmodel)
            state = None  # if your model is stateless
            opt_state = self.optimizer.init(eqx.filter(model, eqx.is_inexact_array))

            best_model, best_state, _, _ = train(
                model=model,
                state=state,
                opt_state=opt_state,
                optimizer=self.optimizer,
                loss_fn=self.loss_fn,
                X_train=X_sample,
                y_train=y_sample,
                X_val=X_sample,
                y_val=y_sample,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                patience=self.patience,
                key=kmodel,
            )

            # 5) Compute full‐set logits from this weak learner
            kb, kpred = jr.split(kb)
            logits_full = predict(best_model, best_state, X, kpred)
            # • If binary: shape (N, 1) or (N,), a single scalar per example.
            # • If multiclass: shape (N, K), a vector per example.

            # 6) Branch: binary vs. multiclass
            if self.num_classes == 2:
                # -------------------
                # BINARY AdaBoost
                # -------------------
                # Expect logits_full.ravel() is shape (N,)
                z = logits_full.ravel()
                # Convert logit → ±1
                h = jnp.sign(z)
                h = jnp.where(h == 0, 1, h)  # ties go to +1

                # Weighted error ε_b = ∑_i w_i · 1{h_i ≠ y_i}
                incorrect = (h != y).astype(jnp.float32)
                epsilon = jnp.sum(w * incorrect)

                # If ε_b > 0.5, “flip” h so that error becomes 1 − ε
                if epsilon > 0.5:
                    h = -h
                    epsilon = 1.0 - epsilon

                # α_b = ½·ln((1−ε)/ε)
                alpha = 0.5 * jnp.log((1.0 - epsilon) / (epsilon + 1e-15))

                # Update sample weights: w_i ← w_i · exp(−α·y_i·h_i)
                w = w * jnp.exp(-alpha * y * h)
                w = w / jnp.sum(w)

            else:
                # -------------------
                # MULTICLASS AdaBoost (SAMME)
                # -------------------
                # Expect logits_full has shape (N, K)
                K = self.num_classes
                # Discrete prediction: h_i = argmax_k logits_full[i, k]
                h = jnp.argmax(logits_full, axis=1)  # shape (N,), in {0,…,K−1}

                # Weighted error: ε_b = ∑_i w_i · 1{h_i ≠ y_i}
                incorrect = (h != y).astype(jnp.float32)
                epsilon = jnp.sum(w * incorrect)

                # If ε_b ≥ (K−1)/K, this weak learner is no better than random → skip it
                if epsilon >= (K - 1) / K:
                    print(
                        f"Weak learner {b+1}/{self.B} skipped: error ε = {epsilon:.4f} ≥ (K−1)/K"
                    )
                    continue

                # α_b = ln((1−ε)/ε) + ln(K−1)
                alpha = jnp.log((1.0 - epsilon) / (epsilon + 1e-15)) + jnp.log(K - 1.0)

                # Update sample weights:
                #   if h_i == y_i → multiply w_i by exp(−α),
                #   else           → multiply w_i by exp(+α).
                correct_mask = h == y
                w = w * jnp.where(correct_mask, jnp.exp(-alpha), jnp.exp(alpha))
                w = w / jnp.sum(w)

            # 7) Store (weak learner, its state, α_b)
            self.weak_learners.append((best_model, best_state, float(alpha)))
            print(f"Weak learner {b+1}/{self.B} – ε = {epsilon:.4f}, α = {alpha:.4f}")

        # end for b

    def predict(self, X: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Make final predictions on new data.

        Returns:
          • If num_classes == 2: array of shape (N,) in {−1,+1}.
          • If num_classes > 2: array of shape (N,) in {0,…,K−1}.
        """
        M = X.shape[0]
        keys = jr.split(key, len(self.weak_learners))

        if self.num_classes == 2:
            # -------------------
            # BINARY vote: sign(∑ α_b·h_b(x))
            # -------------------
            agg = jnp.zeros(M)
            for (model, state, alpha), kb in zip(self.weak_learners, keys):
                z = predict(model, state, X, kb).ravel()  # shape (M,)
                h = jnp.sign(z)
                h = jnp.where(h == 0, 1, h)
                agg = agg + alpha * h

            final_pred = jnp.sign(agg)
            final_pred = jnp.where(final_pred == 0, 1, final_pred)  # ties to +1
            return np.array(final_pred)  # in {−1,+1}

        else:
            # -------------------
            # MULTICLASS vote: accumulate α_b for whichever class is predicted by learner b
            # -------------------
            K = self.num_classes
            scores = jnp.zeros(
                (M, K)
            )  # score[i,k] = sum of α_b for all b that predicted class k

            for (model, state, alpha), kb in zip(self.weak_learners, keys):
                logits = predict(model, state, X, kb)  # shape (M, K)
                h = jnp.argmax(logits, axis=1)  # shape (M,), in {0,…,K−1}
                one_hot_h = jax.nn.one_hot(h, K)  # shape (M, K)
                scores = scores + alpha * one_hot_h  # add α to the predicted class slot

            final_pred = jnp.argmax(
                scores, axis=1
            )  # choose the class with highest total α
            return np.array(final_pred)  # shape (M,), in {0,…,K−1}


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
