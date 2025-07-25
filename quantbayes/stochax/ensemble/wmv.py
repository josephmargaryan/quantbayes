"""
PAC-Bayes Ensemble for **Equinox** neural networks
==================================================

*   Each weak learner is an Equinox model trained with Optax.
*   Default subset size **r = d + 1** (can be overridden).
*   Supports every bound we implemented for the scikit-learn version:
        - PB-λ, PB-KL, empirical Bernstein, tandem, split-KL,
          unexpected-Bernstein
*   For the *tandem* bound the co–error is evaluated **only on the
    intersection of the two validation sets**, exactly as required by
    Theorem 3.36.

Only two helper functions are assumed to live in your code-base:

``predict(model, state, X, key)``   → logits (or numeric output)
``train(model, state, opt_state, …)`` → best_model, best_state, …

If your signatures differ, adapt the two calls marked with **# <- helper**.

Author  : 2025-06-29
"""

from __future__ import annotations

import copy
import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from sklearn.metrics import mean_squared_error, zero_one_loss
from sklearn.utils import check_random_state

# ----------------------------------------------------------------------
#  Bound criteria (re-used from your WMV module)
# ----------------------------------------------------------------------
from quantbayes.pacbayes.wmv import (  # noqa: F401
    PBBernsteinCriterion,
    PBKLCriterion,
    PBLambdaCriterion,
    SplitKLCriterion,
    TandemCriterion,
    UnexpectedBernsteinCriterion,
    SplitBernsteinCriterion,
)

# helper utilities you already have
from quantbayes.stochax.trainer.train import (  # noqa: F401
    binary_loss,  # example loss fn for smoke-test
    predict as _single_predict,  # <- helper
    train,  # <- helper
)

_CRITERIA: Dict[str, Callable] = {
    "pblambda": PBLambdaCriterion,
    "pbkl": PBKLCriterion,
    "pbbernstein": PBBernsteinCriterion,
    "tandem": TandemCriterion,
    "splitkl": SplitKLCriterion,
    "splitbernstein": SplitBernsteinCriterion,
    "unexpectedbernstein": UnexpectedBernsteinCriterion,
}


# ----------------------------------------------------------------------
#  Ensemble
# ----------------------------------------------------------------------
class PacBayesEnsemble:
    """
    PAC-Bayes ensemble of Equinox networks.

    Parameters
    ----------
    constructors : sequence of callables  key -> model  **or**
                   key -> (model, state)
    task         : "binary" | "multiclass" | "regression"
    loss_fn      : returns a scalar already normalised to [0,1]
    optimizer    : Optax optimizer
    bound_type   : one of _CRITERIA keys
    delta        : confidence level
    r            : subset size (default d+1)
    seed         : RNG seed
    """

    # -----------------------------------------------------------------
    def __init__(
        self,
        constructors: Sequence[Callable[[jr.PRNGKey], object]],
        *,
        task: str = "binary",
        loss_fn: Callable = binary_loss,
        optimizer: optax.GradientTransformation = optax.adam(1e-3),
        bound_type: str = "pblambda",
        delta: float = 0.05,
        r: Optional[int] = None,
        L_max: float = 1.0,
        seed: int = 0,
    ):
        if task not in {"binary", "multiclass", "regression"}:
            raise ValueError("task must be 'binary' | 'multiclass' | 'regression'")
        if bound_type not in _CRITERIA:
            raise ValueError(f"unknown bound_type '{bound_type}'")

        self.constructors = list(constructors)
        self.task = task
        self.loss_fn = loss_fn
        self.opt = optimizer
        self.bound_type = bound_type
        self.delta = float(delta)
        self.r_user = r
        self.L_max = float(L_max)
        self.seed = int(seed)

        # will be filled in fit()
        self.classes_: Optional[np.ndarray] = None
        self.results_: List[Dict] = []
        self.best_bound_: float = float("inf")
        self.best_m_: Optional[int] = None
        self.best_rho_: Optional[np.ndarray] = None
        self.best_models_: Optional[List[Tuple[eqx.Module, object]]] = None
        self.is_fitted = False

    # -----------------------------------------------------------------
    #                          private helpers
    # -----------------------------------------------------------------
    def _compute_hold_loss(
        self,
        model,
        state,
        Xv: jnp.ndarray,
        yv_idx_np: np.ndarray,
        key: jr.PRNGKey,
    ) -> float:
        logits = _single_predict(model, state, Xv, key)  # helper
        if self.task == "binary":
            preds = (jax.nn.sigmoid(logits).ravel() >= 0.5).astype(int)
            return float(zero_one_loss(yv_idx_np, preds))

        if self.task == "multiclass":
            preds = np.array(jnp.argmax(logits, axis=-1))
            return float(zero_one_loss(yv_idx_np, preds))

        preds = np.array(logits).ravel()
        mse = mean_squared_error(yv_idx_np, preds)
        return float(min(mse, self.L_max) / self.L_max)

    # ---------- aggregation of probabilities ------------------------
    def _aggregate_proba(
        self,
        X: jnp.ndarray,
        models_states: List[Tuple[eqx.Module, object]],
        rho: np.ndarray,
        key: jr.PRNGKey,
    ) -> np.ndarray:
        m = len(models_states)
        keys = jr.split(key, m)

        # binary ------------------------------------------------------
        if self.task == "binary":
            p1 = np.zeros(len(X))
            for (mdl, st), r_i, k in zip(models_states, rho, keys):
                logits = _single_predict(mdl, st, X, k)
                p1 += r_i * jax.nn.sigmoid(logits)
            p1 = np.array(p1)
            return np.vstack([1.0 - p1, p1]).T  # (n,2)

        # multiclass --------------------------------------------------
        C = len(self.classes_)
        probs = np.zeros((len(X), C))
        for (mdl, st), r_i, k in zip(models_states, rho, keys):
            logits = _single_predict(mdl, st, X, k)  # (n,C)
            probs += r_i * jax.nn.softmax(logits, axis=-1)
        return np.array(probs)  # (n,C)

    # ---------- aggregation of labels/values ------------------------
    def _aggregate_label_idx(
        self,
        X: jnp.ndarray,
        models_states: List[Tuple[eqx.Module, object]],
        rho: np.ndarray,
        key: jr.PRNGKey,
    ):
        """Return *internal* label indices (or regression values)."""
        if self.task == "regression":
            preds = np.zeros(len(X))
            keys = jr.split(key, len(models_states))
            for (mdl, st), r_i, k in zip(models_states, rho, keys):
                preds += r_i * np.array(_single_predict(mdl, st, X, k)).ravel()
            return preds

        P = self._aggregate_proba(X, models_states, rho, key)
        if self.task == "binary":
            return (P[:, 1] >= 0.5).astype(int)
        return P.argmax(axis=1).astype(int)

    # -----------------------------------------------------------------
    #                                fit
    # -----------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        m_values: Sequence[int],
        n_runs: int = 3,
        batch_size: int = 64,
        num_epochs: int = 200,
        patience: int = 20,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ):
        # ---------- label preprocessing -----------------------------
        if self.task == "multiclass":
            self.classes_ = np.unique(y)
            idx_of = {lbl: i for i, lbl in enumerate(self.classes_)}
            y_idx = np.vectorize(idx_of.get)(y)
            if y_test is not None:
                y_test_idx = np.vectorize(idx_of.get)(y_test)
        else:
            self.classes_ = None
            y_idx = y
            y_test_idx = y_test

        # convert once to JAX
        Xj = jnp.asarray(X, dtype=jnp.float32)
        yj = jnp.asarray(y_idx)
        y_np = np.array(y_idx).ravel()

        if X_test is not None and y_test is not None:
            Xtest_j = jnp.asarray(X_test, dtype=jnp.float32)
            ytest_np = np.array(y_test_idx).ravel()
        else:
            Xtest_j = ytest_np = None

        # ---------- general set-up ----------------------------------
        n, d = X.shape
        r = self.r_user if self.r_user is not None else d + 1
        if not (1 <= r < n):
            raise ValueError(f"need 1 ≤ r < n; got r={r}, n={n}")
        n_r = n - r

        criterion = _CRITERIA[self.bound_type]()
        uses_lambda = getattr(criterion, "uses_lambda", False)

        rng_master = jr.PRNGKey(self.seed)
        py_rng = check_random_state(self.seed)

        self.results_.clear()
        self.best_bound_ = float("inf")

        # ---------- outer loops -------------------------------------
        for run in range(n_runs):
            rng_run, rng_master = jr.split(rng_master)
            for m in m_values:
                t0 = time.time()

                # 1) train weak learners ----------------------------
                models_states: List[Tuple[eqx.Module, object]] = []
                val_masks: List[np.ndarray] = []
                losses = np.zeros(m)

                for i in range(m):
                    rng_i, rng_run = jr.split(rng_run)
                    idx = py_rng.choice(n, size=r, replace=False)
                    mask_val = np.ones(n, dtype=bool)
                    mask_val[idx] = False

                    raw = self.constructors[i % len(self.constructors)](rng_i)
                    mdl, state0 = raw if isinstance(raw, tuple) else (raw, None)
                    opt_state = self.opt.init(eqx.filter(mdl, eqx.is_inexact_array))

                    mdl_tr, state_tr, *_ = train(  # helper
                        mdl,
                        state0,
                        opt_state,
                        self.opt,
                        self.loss_fn,
                        Xj[idx],
                        yj[idx],
                        Xj[mask_val],
                        yj[mask_val],
                        batch_size,
                        num_epochs,
                        patience,
                        rng_i,
                    )
                    models_states.append((mdl_tr, state_tr))
                    val_masks.append(mask_val)

                # 2) hold-out and pair‐losses ------------------------
                rng_pred, rng_run = jr.split(rng_run)
                keys_pred = jr.split(rng_pred, m)

                for i, ((mdl, st), k) in enumerate(zip(models_states, keys_pred)):
                    losses[i] = self._compute_hold_loss(
                        mdl, st, Xj[val_masks[i]], y_np[val_masks[i]], k
                    )

                if self.bound_type == "tandem":
                    pair_losses = np.zeros((m, m))
                    min_inter = n
                    preds_cache = [
                        self._aggregate_label_idx(Xj, [(mdl, st)], np.array([1.0]), k)
                        for (mdl, st), k in zip(models_states, keys_pred)
                    ]
                    for i in range(m):
                        for j in range(i, m):
                            mask = val_masks[i] & val_masks[j]
                            sz = int(mask.sum())
                            min_inter = min(min_inter, sz)
                            cij = (
                                0.0
                                if sz == 0
                                else float(
                                    np.mean(
                                        (preds_cache[i][mask] != y_np[mask])
                                        & (preds_cache[j][mask] != y_np[mask])
                                    )
                                )
                            )
                            pair_losses[i, j] = pair_losses[j, i] = cij
                else:
                    pair_losses = min_inter = None

                # 3) alternating minimisation -----------------------
                rho = np.full(m, 1.0 / m)
                lam = 0.5 if uses_lambda else None
                log_const = math.log(2.0 * math.sqrt(n_r) / self.delta)
                prev_bound = float("inf")

                for _ in range(200):
                    kl_rho = float((rho * np.log(rho * m)).sum())

                    if self.bound_type == "tandem":
                        stat, bound = criterion.compute(
                            pair_losses, rho, kl_rho, min_inter, self.delta, lam
                        )
                    else:
                        stat, bound = criterion.compute(
                            losses, rho, kl_rho, n_r, self.delta, lam
                        )

                    if abs(prev_bound - bound) < 1e-6:
                        break
                    prev_bound = bound

                    if uses_lambda:
                        lam = 2.0 / (
                            math.sqrt(1.0 + 2.0 * n_r * stat / (kl_rho + log_const))
                            + 1.0
                        )

                    shift = losses.min()
                    rho = np.exp(-(lam or 1.0) * n_r * (losses - shift))
                    rho /= rho.sum()

                # 4) diagnostics ------------------------------------
                if Xtest_j is not None:
                    y_pred_idx = self._aggregate_label_idx(
                        Xtest_j, models_states, rho, rng_run
                    )
                    y_pred = (
                        self.classes_[y_pred_idx]
                        if self.task == "multiclass"
                        else y_pred_idx
                    )
                    test_loss = (
                        zero_one_loss(y_test, y_pred)
                        if self.task in {"binary", "multiclass"}
                        else mean_squared_error(ytest_np, y_pred)
                    )
                else:
                    test_loss = None

                hold_loss = float(rho @ losses)
                self.results_.append(
                    dict(
                        m=m,
                        hold_loss=hold_loss,
                        bound=float(bound),
                        test_loss=test_loss,
                        time=time.time() - t0,
                    )
                )

                if bound < self.best_bound_:
                    self.best_bound_ = float(bound)
                    self.best_m_ = m
                    self.best_rho_ = rho.copy()
                    self.best_models_ = [
                        (copy.deepcopy(mdl), copy.deepcopy(st))
                        for mdl, st in models_states
                    ]

        self.is_fitted = True
        return self

    # -----------------------------------------------------------------
    #                         public interface
    # -----------------------------------------------------------------
    def predict(self, X: np.ndarray, key: Optional[jr.PRNGKey] = None):
        if not self.is_fitted:
            raise RuntimeError("call fit() first")
        key = key or jr.PRNGKey(self.seed + 12345)
        Xj = jnp.asarray(X, dtype=jnp.float32)
        idx = self._aggregate_label_idx(Xj, self.best_models_, self.best_rho_, key)
        if self.task == "multiclass":
            return self.classes_[idx]
        return idx

    def predict_proba(
        self, X: np.ndarray, key: Optional[jr.PRNGKey] = None
    ) -> np.ndarray:
        if self.task == "regression":
            raise AttributeError("predict_proba undefined for regression")
        if not self.is_fitted:
            raise RuntimeError("call fit() first")
        key = key or jr.PRNGKey(self.seed + 67890)
        Xj = jnp.asarray(X, dtype=jnp.float32)
        return self._aggregate_proba(Xj, self.best_models_, self.best_rho_, key)

    # -----------------------------------------------------------------
    def summary(self):
        if not self.is_fitted:
            raise RuntimeError("call fit() first")
        hdr = f"{'m':>4s} | {'hold':>6s} | {'bound':>6s} | {'test':>6s} | {'t[s]':>6s}"
        print(hdr)
        print("-" * len(hdr))
        by_m = {}
        for r in self.results_:
            by_m.setdefault(r["m"], []).append(r)
        for m, rows in sorted(by_m.items()):
            hold = np.mean([r["hold_loss"] for r in rows])
            bnd = np.mean([r["bound"] for r in rows])
            tst = (
                np.mean([r["test_loss"] for r in rows if r["test_loss"] is not None])
                if rows[0]["test_loss"] is not None
                else float("nan")
            )
            tavg = np.mean([r["time"] for r in rows])
            print(f"{m:4d} | {hold:6.3f} | {bnd:6.3f} | {tst:6.3f} | {tavg:6.2f}")


# ----------------------------------------------------------------------
#  Tiny demo network
# ----------------------------------------------------------------------
class BinaryStump(eqx.Module):
    lin: eqx.nn.Linear

    def __init__(self, d, key):
        self.lin = eqx.nn.Linear(d, 1, key=key)

    def __call__(self, x, key, state):
        return self.lin(x), state


# ----------------------------------------------------------------------
#  Smoke-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=600, n_features=20, n_informative=10, random_state=0
    )
    X_train, y_train = X[:500], y[:500]
    X_test, y_test = X[500:], y[500:]

    constructors = [
        lambda k: BinaryStump(X.shape[1], k),
        lambda k: BinaryStump(X.shape[1], jr.split(k)[0]),
    ]

    ens = PacBayesEnsemble(
        constructors,
        task="binary",
        loss_fn=binary_loss,  # helper
        optimizer=optax.adam(1e-2),
        bound_type="splitbernstein",
        r=None,  # defaults to d+1
        delta=0.05,
        seed=0,
    )

    ens.fit(
        X_train,
        y_train,
        m_values=[4, 8],
        n_runs=2,
        batch_size=64,
        num_epochs=50,
        patience=10,
        X_test=X_test,
        y_test=y_test,
    )
    ens.summary()

    y_pred = ens.predict(X_test)
    emp_err = zero_one_loss(y_test, y_pred)
    print(
        f"\nEmpirical MV error: {emp_err:.3f}   |  Certified bound: {ens.best_bound_:.3f}"
    )
    assert emp_err <= ens.best_bound_ + 1e-8, "PAC-Bayes bound violated!"
    print("Smoke-test passed ✓")
