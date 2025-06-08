import math
import copy
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from sklearn.metrics import zero_one_loss, mean_squared_error

from quantbayes.pacbayes.WMV import (
    PBLambdaCriterion,
    PBKLCriterion,
    TandemCriterion,
    PBBernsteinCriterion,
    SplitKLCriterion, 
    UnexpectedBernsteinCriterion
)
from quantbayes.stochax.trainer.train import (
    train,
    binary_loss,
    multiclass_loss,
    regression_loss,
)
from quantbayes.stochax.trainer.train import predict as _single_predict

__all__ = ["PacBayesEnsemble"]

CRITERIA = {
    "pblambda": PBLambdaCriterion,
    "pbkl": PBKLCriterion,
    "tandem": TandemCriterion,
    "pbbernstein": PBBernsteinCriterion,
    "split-kl": SplitKLCriterion,
    "unexpected-bernstein": UnexpectedBernsteinCriterion
}


class PacBayesEnsemble:
    """
    A PAC-Bayes ensemble of Equinox neural networks for binary, multiclass, or regression tasks.

    Parameters
    ----------
    model_constructors : list of callables
        Each callable should accept a JAX PRNGKey and return an Equinox Module.
        E.g. `lambda key: MyNet(key)`. If m > len(model_constructors), constructors are cycled.
    task : {"binary", "multiclass", "regression"}
        The learning task. Determines which loss/prediction logic to use.
    loss_fn : callable
        A function (model, state, x_batch, y_batch, key) -> (loss, new_state), returning a scalar ∈ [0,1].
        E.g. `binary_loss`, `multiclass_loss`, or `regression_loss`.
    optimizer : optax.GradientTransformation
        The optimizer (e.g. `optax.adam(1e-3)`).
    bound_type : {"pbkl", "pblambda", "tandem", "pbbernstein"}, default="pbkl"
        Which PAC-Bayes bound to optimize.
    delta : float, default=0.05
        Confidence parameter δ ∈ (0,1) for the PAC-Bayes bound.
    L_max : float, default=1.0
        Maximum regression loss (for clipping in regression tasks).
    seed : int, default=0
        Global random seed. All random draws (model init, training, hold-out eval, test eval)
        are derived from this key so that runs are reproducible.

    Attributes
    ----------
    results_ : list of dict
        For each m in m_values, a dict containing:
          - "m"          : int (ensemble size)
          - "hold_loss"  : float (weighted hold-out loss = Σ_i ρ_i ℓ_i)
          - "bound"      : float (final PAC-Bayes bound)
          - "test_loss"  : float or None (0-1 or MSE on test set)
    best_m_ : int
        The ensemble size m that achieved the lowest PAC-Bayes bound.
    best_bound_ : float
        The lowest PAC-Bayes bound achieved.
    best_rho_ : ndarray of shape (best_m_,)
        The Gibbs weights ρ_i for the best ensemble.
    best_models_states_ : list of (model, state) tuples
        The Equinox model and state for each of the best_m_ weak learners.
    is_fitted : bool
        Whether fit() has been called successfully.
    """

    def __init__(
        self,
        model_constructors,  # list of callables key -> Equinox module
        task,  # "binary", "multiclass", "regression"
        loss_fn,
        optimizer,
        bound_type="pbkl",
        delta=0.05,
        L_max=1.0,
        seed=0,
    ):
        if bound_type not in CRITERIA:
            raise ValueError(f"Unknown bound_type '{bound_type}'. Must be one of {list(CRITERIA)}.")
        if task not in ("binary", "multiclass", "regression"):
            raise ValueError("`task` must be 'binary', 'multiclass', or 'regression'.")

        self.model_constructors = model_constructors
        self.task = task
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.bound_type = bound_type
        self.delta = delta
        self.L_max = L_max
        self.seed = seed

        # Attributes to be set in fit()
        self.results_ = []
        self.best_m_ = None
        self.best_bound_ = np.inf
        self.best_rho_ = None
        self.best_models_states_ = None
        self.is_fitted = False

    def _compute_hold_loss(self, model, state, X_hold, y_hold_np, key):
        """
        Compute a single weak learner’s hold-out loss ℓ_i ∈ [0,1].
        """
        logits = _single_predict(model, state, X_hold, key)
        if self.task == "binary":
            probs = jax.nn.sigmoid(logits).ravel()
            preds = (probs >= 0.5).astype(int)
            return float(zero_one_loss(y_hold_np, preds))
        elif self.task == "multiclass":
            probs = jax.nn.softmax(logits, axis=-1)
            preds = np.array(jnp.argmax(probs, axis=-1))
            return float(zero_one_loss(y_hold_np, preds))
        else:  # regression
            preds = np.array(logits).ravel()
            mse = mean_squared_error(y_hold_np, preds)
            return float(min(mse, self.L_max) / self.L_max)

    def _predict_with_weights(self, X, models_states, rho, rng):
        """
        Perform majority-vote (binary/multiclass) or weighted-average (regression) using
        the provided list of (model, state) and weights rho. Returns numpy array of predictions.
        """
        X = jnp.asarray(X, dtype=jnp.float32)
        n_samples = X.shape[0]
        m = len(models_states)
        preds_np: np.ndarray

        if self.task == "binary":
            # Each net outputs a logit; convert to ±1
            P_votes = np.zeros((m, n_samples))
            rng, *keys = jr.split(rng, num=m + 1)
            keys = keys[:m]
            for i, (model, state) in enumerate(models_states):
                logits = _single_predict(model, state, X, keys[i])
                probs = jax.nn.sigmoid(logits).ravel()
                bit = np.where(np.array(probs) >= 0.5, +1, -1)
                P_votes[i, :] = bit
            agg = (rho[:, None] * P_votes).sum(axis=0)
            # Convert back to {0,1}
            preds_np = (agg >= 0).astype(int)

        elif self.task == "multiclass":
            # Each net outputs logits over C classes; aggregate probabilities
            # Determine C from first net’s output shape on a dummy batch
            rng, dummy_key = jr.split(rng)
            dummy_logits = _single_predict(models_states[0][0], models_states[0][1], X[:1], dummy_key)
            C = dummy_logits.shape[-1]
            votes = np.zeros((n_samples, C))
            rng, *keys = jr.split(rng, num=m + 1)
            keys = keys[:m]
            for i, (model, state) in enumerate(models_states):
                logits = _single_predict(model, state, X, keys[i])
                probs = jax.nn.softmax(logits, axis=-1)
                votes += rho[i] * np.array(probs)
            preds_np = votes.argmax(axis=1)

        else:  # regression
            preds = np.zeros(n_samples)
            rng, *keys = jr.split(rng, num=m + 1)
            keys = keys[:m]
            for i, (model, state) in enumerate(models_states):
                logits = _single_predict(model, state, X, keys[i])
                preds += rho[i] * np.array(logits).ravel()
            preds_np = preds

        return preds_np

    def fit(
        self,
        X_train,
        y_train,
        X_hold,
        y_hold,
        X_test,
        y_test,
        m_values,
        batch_size=64,
        num_epochs=200,
        patience=20,
    ):
        """
        Train the PAC-Bayes ensemble.

        Parameters
        ----------
        X_train : array-like, shape (N_train, d)
            Training data for weak learners.
        y_train : array-like, shape (N_train,) or (N_train,1)
            Training labels (binary: 0/1, multiclass: integers 0..C-1, regression: real).
        X_hold, y_hold : array-like
            Hold-out set used to compute each weak learner's hold-out loss ℓ_i (and pairwise losses if needed).
        X_test, y_test : array-like
            Test set used to evaluate the final majority-vote performance. If None, test_loss is set to None.
        m_values : list of int
            List of ensemble sizes to try, e.g. [1, 2, 5, 10].
        batch_size : int, default=64
            Batch size for training each weak learner.
        num_epochs : int, default=200
            Maximum epochs for training each weak learner.
        patience : int, default=20
            Early-stopping patience on hold-out loss during each weak learner’s train().

        Returns
        -------
        self
        """
        # Convert inputs to JAX arrays
        X_train = jnp.asarray(X_train, dtype=jnp.float32)
        y_train = jnp.asarray(y_train)
        X_hold = jnp.asarray(X_hold, dtype=jnp.float32)
        y_hold = jnp.asarray(y_hold)
        if X_test is not None and y_test is not None:
            X_test = jnp.asarray(X_test, dtype=jnp.float32)
            y_test = jnp.asarray(y_test)
        else:
            X_test = None
            y_test = None

        # Flatten hold-out/test labels for numpy-based metrics
        y_hold_np = np.array(y_hold).ravel()
        if y_test is not None:
            y_test_np = np.array(y_test).ravel()
        else:
            y_test_np = None

        rng = jr.PRNGKey(self.seed)
        self.results_ = []
        self.best_bound_ = np.inf
        self.best_m_ = None
        self.best_rho_ = None
        self.best_models_states_ = None

        # Loop over ensemble sizes
        for m in m_values:
            # Step 1: Train m weak learners
            models_states = []
            for i in range(m):
                rng, sk = jr.split(rng)
                # Initialize new model & state
                constructor = self.model_constructors[i % len(self.model_constructors)]
                model, state = eqx.nn.make_with_state(lambda key: constructor(key))(sk)
                opt_state = self.optimizer.init(eqx.filter(model, eqx.is_inexact_array))
                # Train with early stopping on (X_hold, y_hold)
                best_model, best_state, _, _ = train(
                    model,
                    state,
                    opt_state,
                    self.optimizer,
                    self.loss_fn,
                    X_train,
                    y_train,
                    X_hold,
                    y_hold,
                    batch_size,
                    num_epochs,
                    patience,
                    sk,
                )
                models_states.append((best_model, best_state))

            n_hold = X_hold.shape[0]

            # === Step 2: Compute hold-out losses (and pairwise if needed) ===

            if self.bound_type == "tandem":
                #
                # For tandem: build an (m × n_hold) matrix H[i,t] = 1_{[h_i(x_t) ≠ y_t]},
                # then pair_losses = (H @ H.T)/n_hold, and losses = diag(pair_losses).
                #
                H = np.zeros((m, n_hold), dtype=np.float32)
                yh_np = y_hold_np  # already flattened

                # We need a separate PRNGKey for each model’s prediction on the hold set
                rng, *all_keys = jr.split(rng, num=m + 1)
                keys = all_keys[:m]

                for i, (model, state) in enumerate(models_states):
                    logits = _single_predict(model, state, X_hold, keys[i])
                    if self.task == "binary":
                        probs = jax.nn.sigmoid(logits).ravel()
                        preds = (probs >= 0.5).astype(int)
                    else:  # "multiclass"
                        probs = jax.nn.softmax(logits, axis=-1)
                        preds = np.array(jnp.argmax(probs, axis=-1))
                    H[i, :] = (preds != yh_np).astype(np.float32)

                # Build the m × m matrix of tandem (pairwise) losses
                pair_losses = (H @ H.T) / float(n_hold)  # shape (m,m)
                # The diagonal entries are the marginal hold-out losses ℓ_i
                losses = np.diag(pair_losses).copy()
            else:
                #
                # For pbkl / pblambda / pbbernstein: we only need the vector of marginal hold-out losses
                #
                losses = np.zeros(m, dtype=np.float32)
                # create one PRNGKey per model
                rng, *hold_keys = jr.split(rng, num=m + 1)
                hold_keys = hold_keys[:m]
                for i, (model, state) in enumerate(models_states):
                    losses[i] = self._compute_hold_loss(model, state, X_hold, y_hold_np, hold_keys[i])

                # pair_losses is not used for these bound types; but define a placeholder
                pair_losses = None

            # === Step 3: PAC-Bayes optimize λ and ρ ===
            pi = np.full(m, 1.0 / m, dtype=np.float64)
            rho = pi.copy()
            n_r = n_hold
            lam = max(1.0 / math.sqrt(n_r), 0.5)
            Crit = CRITERIA[self.bound_type]()
            prev_bound = np.inf

            for _ in range(200):
                kl = float((rho * np.log(rho / pi)).sum())

                if self.bound_type == "tandem":
                    # Crit.compute expects: (pair_losses (m×m), rho (m), kl, n_r, delta, lam, full_n)
                    stat, bound = Crit.compute(pair_losses, rho, kl, n_r, self.delta, lam, n_r)
                else:
                    # For pbkl / pblambda / pbbernstein, Crit.compute expects:
                    #    (vector_of_losses (m,), rho (m,), kl, n_r, delta, lam, full_n)
                    stat, bound = Crit.compute(losses, rho, kl, n_r, self.delta, lam, n_r)

                if abs(prev_bound - bound) < 1e-6:
                    break
                prev_bound = bound

                # Update λ according to the same formula in your original code
                lam = 2.0 / (
                    math.sqrt(
                        1
                        + 2 * n_r * stat
                        / (kl + math.log(2 * math.sqrt(n_r) / self.delta))
                    )
                    + 1
                )

                # “shift” is always based on the marginal losses
                shift = losses.min()
                w = np.exp(-lam * n_r * (losses - shift))
                rho = w / w.sum()

            # === Step 4: Evaluate test loss (if provided) ===
            if X_test is not None and y_test_np is not None:
                rng, subkey = jr.split(rng)
                preds = self._predict_with_weights(X_test, models_states, rho, subkey)
                if self.task in ("binary", "multiclass"):
                    test_loss = float(zero_one_loss(y_test_np, preds))
                else:
                    test_loss = float(mean_squared_error(y_test_np, preds))
            else:
                test_loss = None

            # Weighted hold-out loss = sum_i rho_i * ℓ_i
            hold_loss = float((rho * losses).sum())

            self.results_.append({
                "m": m,
                "hold_loss": hold_loss,
                "bound": float(bound),
                "test_loss": test_loss,
            })

            # Keep track of the best‐bound ensemble
            if bound < self.best_bound_:
                self.best_bound_ = bound
                self.best_m_ = m
                self.best_rho_ = rho.copy()
                # Deepcopy models and states so they are not overwritten later
                self.best_models_states_ = [(copy.deepcopy(mdl), copy.deepcopy(st))
                                            for mdl, st in models_states]

        # After trying all m_values, store best ensemble for predict()
        self._last_models_states = self.best_models_states_
        self._last_rho = self.best_rho_
        self.is_fitted = True
        return self


    def predict(self, X, key=None):
        """
        Majority-vote (binary/multiclass) or weighted-average (regression) on new data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, d)
            Input features.
        key : jax.random.PRNGKey, optional
            If provided, used to seed any stochastic components (dropout, etc.).
            If None, uses a key derived from self.seed to ensure reproducibility.

        Returns
        -------
        preds : ndarray
            - If task == "binary": array of {0,1} of shape (n_samples,)
            - If task == "multiclass": array of int labels shape (n_samples,)
            - If task == "regression": array of floats shape (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        if key is None:
            key = jr.PRNGKey(self.seed)
        preds_np = self._predict_with_weights(X, self._last_models_states, self._last_rho, key)
        return preds_np

    def score(self, X, y, key=None):
        """
        Compute a score on (X, y). For classification tasks, returns accuracy.
        For regression, returns negative MSE (so higher is better).

        Parameters
        ----------
        X : array-like, shape (n_samples, d)
        y : array-like, shape (n_samples,) or (n_samples,1)
        key : jax.random.PRNGKey, optional

        Returns
        -------
        score : float
            - For binary/multiclass: 1 - zero_one_loss
            - For regression:  - mean_squared_error
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before score().")
        y_np = np.array(y).ravel()
        preds = self.predict(X, key)
        if self.task in ("binary", "multiclass"):
            return float(1.0 - zero_one_loss(y_np, preds))
        else:
            return float(-mean_squared_error(y_np, preds))

    def summary(self):
        """
        Print a table of hold-out loss, PAC-Bayes bound, and test loss for each m.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before summary().")
        header = f"{'m':>4s} | {'HoldLoss':>10s} | {'Bound':>10s} | {'TestLoss':>10s}"
        print(header)
        print("-" * len(header))
        for r in self.results_:
            test_str = f"{r['test_loss']:.4f}" if r["test_loss"] is not None else "None"
            print(f"{r['m']:4d} | {r['hold_loss']:.4f}   | {r['bound']:.4f}   | {test_str}")

    # Properties to access results directly
    @property
    def results(self):
        """List of dicts with keys 'm', 'hold_loss', 'bound', 'test_loss'."""
        return self.results_

    @property
    def hold_losses(self):
        """Return a dict mapping m -> hold_loss."""
        return {r["m"]: r["hold_loss"] for r in self.results_}

    @property
    def bounds(self):
        """Return a dict mapping m -> bound."""
        return {r["m"]: r["bound"] for r in self.results_}

    @property
    def test_losses(self):
        """Return a dict mapping m -> test_loss (None if no test data)."""
        return {r["m"]: r["test_loss"] for r in self.results_}


# ------------------------
# Define simple Equinox nets
# ------------------------
class EQXBinary(eqx.Module):
    l1: eqx.nn.Linear

    def __init__(self, in_features, key):
        self.l1 = eqx.nn.Linear(in_features, 1, key=key)

    def __call__(self, x, key, state):
        y = self.l1(x)
        return y, state


class EQXMulti(eqx.Module):
    l1: eqx.nn.Linear

    def __init__(self, in_features, n_classes, key):
        self.l1 = eqx.nn.Linear(in_features, n_classes, key=key)

    def __call__(self, x, key, state):
        y = self.l1(x)
        return y, state


class EQXReg(eqx.Module):
    l1: eqx.nn.Linear

    def __init__(self, in_features, key):
        self.l1 = eqx.nn.Linear(in_features, 1, key=key)

    def __call__(self, x, key, state):
        y = self.l1(x)
        return y, state


# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.metrics import zero_one_loss

    # Binary example
    Xc, yc = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=0
    )
    X = jnp.array(Xc, dtype=jnp.float32)
    y = jnp.array(yc.reshape(-1, 1), dtype=jnp.float32)
    split = int(0.8 * len(yc))
    Xe, ye = X[:split], y[:split]
    Xh, yh = X[split:], y[split:]

    # Define two simple constructors for diversity
    def ctor1(key):
        return EQXBinary(X.shape[1], key)

    def ctor2(key):
        return EQXBinary(X.shape[1], jr.split(key)[1])

    ensemble = PacBayesEnsemble(
        [ctor1, ctor2],
        task="binary",
        loss_fn=binary_loss,
        optimizer=optax.adam(1e-3),
        bound_type="tandem",
        delta=0.05,
        seed=42,
    )
    ensemble.fit(Xe, ye, Xh, yh, Xh, yh, m_values=[1, 2, 4, 8], batch_size=64, num_epochs=50, patience=10)
    ensemble.summary()

    # Access raw results
    print("\nHold losses by m:", ensemble.hold_losses)
    print("Bounds by m:    ", ensemble.bounds)
    print("Test losses by m:", ensemble.test_losses)
    print("Best m:", ensemble.best_m_, "with bound", ensemble.best_bound_)

    # Validate predict() agrees with reported test_loss for best_m_
    y_pred = ensemble.predict(Xh, jr.PRNGKey(999))
    y_h_np = np.array(yh).ravel()
    pred_err = zero_one_loss(y_h_np, y_pred)
    print(f"\n.predict() error on hold-out: {pred_err:.4f}")
