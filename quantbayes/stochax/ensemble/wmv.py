# quantbayes/stochax/ensemble/pacbayes_wmv_equinox.py
"""
PAC-Bayes Weighted-Majority-Vote (WMV) for Equinox models

- Trains m weak learners (Equinox modules) on independent random subsets
  of size r (default: d+1), validates on the complement.
- Minimizes a chosen PAC-Bayes bound over posterior ρ (and λ if applicable).
- Final predictions use the *deterministic majority vote* (MV), which is
  what the bound certifies. `predict_proba` is provided for convenience.

Depends on existing utilities in your codebase:
    from quantbayes.stochax.trainer.train import train, predict, binary_loss, multiclass_loss

And PAC-Bayes criteria:
    from quantbayes.pacbayes.wmv import PBLambdaCriterion, PBKLCriterion, PBBernsteinCriterion,
                                     TandemCriterion, SplitKLCriterion, SplitBernsteinCriterion,
                                     UnexpectedBernsteinCriterion

Author: 2025-08-10
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

# --- PAC-Bayes criteria (your implementations) ---
from quantbayes.pacbayes.wmv import (
    PBBernsteinCriterion,
    PBKLCriterion,
    PBLambdaCriterion,
    SplitKLCriterion,
    SplitBernsteinCriterion,
    TandemCriterion,
    UnexpectedBernsteinCriterion,
)

# --- Your training & inference helpers ---
from quantbayes.stochax.trainer.train import (
    train as _train,
    predict as _predict,
    binary_loss,
    multiclass_loss,
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


# ---------------------------------------------------------------------
#  Constant fallback model for degenerate (single-class) subsets
# ---------------------------------------------------------------------
class _ConstantEqx(eqx.Module):
    """Equinox module that outputs a constant logit/logits."""

    logit_or_class: float
    n_classes: Optional[int] = None  # None → binary (single logit)

    def __call__(self, x, key, state):
        n = x.shape[0]
        if self.n_classes is None:
            # binary: one logit column
            out = jnp.full((n, 1), float(self.logit_or_class))
        else:
            # multiclass: peaky logits for a fixed class
            out = jnp.full((n, int(self.n_classes)), -10.0)
            out = out.at[:, int(self.logit_or_class)].set(10.0)
        return out, state


# ---------------------------------------------------------------------
#  Main WMV Ensemble
# ---------------------------------------------------------------------
class PacBayesEnsemble(eqx.Module):
    """
    PAC-Bayes WMV ensemble for Equinox networks (classification only).

    Parameters
    ----------
    constructors : sequence of callables
        Each callable receives a PRNGKey and returns either:
          - `model`, or
          - `(model, state)` if your model keeps an explicit state.
    task : {"binary","multiclass"}
        Problem type.
    optimizer : optax optimizer
        Used for training each weak learner.
    bound_type : one of {"pblambda","pbkl","pbbernstein","tandem","splitkl","splitbernstein","unexpectedbernstein"}
        Which PAC-Bayes bound to minimize.
    delta : float
        Confidence level δ.
    r : Optional[int]
        Subset size for each weak learner. Defaults to `d + 1`.
    seed : int
        RNG seed.
    """

    constructors: Sequence[Callable[[jr.PRNGKey], object]]
    task: str
    optimizer: optax.GradientTransformation
    bound_type: str
    delta: float
    r_user: Optional[int]
    seed: int

    # runtime (filled by fit)
    classes_: Optional[np.ndarray] = None
    best_models_: Optional[List[Tuple[eqx.Module, object]]] = None
    best_rho_: Optional[np.ndarray] = None
    best_m_: Optional[int] = None
    best_bound_: float = float("inf")
    results_: List[Dict] = ()

    def __init__(
        self,
        constructors: Sequence[Callable[[jr.PRNGKey], object]],
        *,
        task: str = "binary",
        optimizer: optax.GradientTransformation = optax.adam(1e-3),
        bound_type: str = "pblambda",
        delta: float = 0.05,
        r: Optional[int] = None,
        seed: int = 0,
    ):
        if task not in {"binary", "multiclass"}:
            raise ValueError(
                "PacBayesEnsemble supports 'binary' or 'multiclass' tasks."
            )
        if bound_type not in _CRITERIA:
            raise ValueError(f"Unknown bound_type '{bound_type}'.")

        object.__setattr__(self, "constructors", list(constructors))
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "bound_type", bound_type)
        object.__setattr__(self, "delta", float(delta))
        object.__setattr__(self, "r_user", r)
        object.__setattr__(self, "seed", int(seed))

        object.__setattr__(self, "classes_", None)
        object.__setattr__(self, "best_models_", None)
        object.__setattr__(self, "best_rho_", None)
        object.__setattr__(self, "best_m_", None)
        object.__setattr__(self, "best_bound_", float("inf"))
        object.__setattr__(self, "results_", [])

    # ------------------------------ internal helpers ------------------------------

    def _mv_predict_idx(
        self,
        Xj: jnp.ndarray,
        models_states: List[Tuple[eqx.Module, object]],
        rho: np.ndarray,
        key: jr.PRNGKey,
    ) -> np.ndarray:
        """Deterministic majority-vote prediction → internal class indices."""
        m = len(models_states)
        keys = jr.split(key, m)

        if self.task == "binary":
            # vote in {-1,+1}; break ties towards +1
            votes = np.zeros(len(Xj), dtype=float)
            for (mdl, st), w, k in zip(models_states, rho, keys):
                logit = _predict(mdl, st, Xj, k).ravel()
                pred = np.where(logit >= 0.0, 1.0, -1.0)
                votes += w * pred
            # map {-1,+1} to class indices (0/1)
            return (votes >= 0).astype(int)

        # multiclass
        C = len(self.classes_)
        votes = np.zeros((len(Xj), C), dtype=float)
        for (mdl, st), w, k in zip(models_states, rho, keys):
            logits = _predict(mdl, st, Xj, k)
            pred = np.asarray(jnp.argmax(logits, axis=-1))
            for c in range(C):
                votes[:, c] += w * (pred == c)
        return votes.argmax(axis=1)

    def _proba_aggregate(
        self,
        Xj: jnp.ndarray,
        models_states: List[Tuple[eqx.Module, object]],
        rho: np.ndarray,
        key: jr.PRNGKey,
    ) -> np.ndarray:
        """Probability aggregation (useful for ranking). Not what the bound certifies."""
        m = len(models_states)
        keys = jr.split(key, m)

        if self.task == "binary":
            p1 = np.zeros(len(Xj), dtype=float)
            for (mdl, st), w, k in zip(models_states, rho, keys):
                logits = _predict(mdl, st, Xj, k).ravel()
                p1 += w * jax.nn.sigmoid(logits)
            p1 = np.clip(p1, 1e-12, 1 - 1e-12)
            return np.stack([1 - p1, p1], axis=1)

        # multiclass
        C = len(self.classes_)
        probs = np.zeros((len(Xj), C), dtype=float)
        for (mdl, st), w, k in zip(models_states, rho, keys):
            logits = _predict(mdl, st, Xj, k)
            probs += w * np.asarray(jax.nn.softmax(logits, axis=-1))
        probs = np.clip(probs, 1e-12, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    # ------------------------------ fit ------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        m_values: Sequence[int],
        n_runs: int = 3,
        batch_size: int = 128,
        num_epochs: int = 200,
        patience: int = 20,
        verbose: bool = False,
    ):
        """
        Train and select the (run, m) combo with the smallest certified MV bound.

        Notes
        -----
        - For 'binary', y can be in {0,1} or {-1,1}. Internally we use {-1,1}.
        - For 'multiclass', y can be arbitrary labels; we map to 0..C-1 internally.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n, d = X.shape

        # normalize labels to internal representation
        if self.task == "binary":
            uniq = np.unique(y)
            if set(uniq) == {0, 1}:
                y_int = np.where(y == 0, -1, 1)
                classes = np.array([0, 1])  # external order retained in predict_proba
            elif set(uniq) == {-1, 1}:
                y_int = y.copy()
                classes = np.array([-1, 1])
            else:
                raise ValueError("Binary task requires labels in {0,1} or {-1,1}.")
        else:
            classes = np.unique(y)
            class_to_idx = {c: i for i, c in enumerate(classes)}
            y_int = np.vectorize(class_to_idx.get)(y)

        object.__setattr__(self, "classes_", classes)

        # choose r
        r = self.r_user if self.r_user is not None else (d + 1)
        if not (1 <= r < n):
            raise ValueError(f"need 1 ≤ r < n; got r={r}, n={n}")
        n_r = n - r

        # choose loss for training each weak learner
        used_loss = binary_loss if self.task == "binary" else multiclass_loss

        Xj = jnp.asarray(X, dtype=jnp.float32)
        yj = jnp.asarray(
            (y_int + 1) // 2 if self.task == "binary" else y_int
        )  # binary expects {0,1}

        rng_master = jr.PRNGKey(self.seed)
        results: List[Dict] = []
        best_bound = float("inf")
        best_models = None
        best_rho = None
        best_m = None

        Crit = _CRITERIA[self.bound_type]
        criterion = Crit()
        uses_lambda = getattr(criterion, "uses_lambda", False)

        for run in range(n_runs):
            rng_run, rng_master = jr.split(rng_master)
            rng_np = np.random.default_rng(int(jr.randint(rng_run, (), 0, 2**31 - 1)))

            for m in m_values:
                t0 = time.time()
                models_states: List[Tuple[eqx.Module, object]] = []
                val_masks: List[np.ndarray] = []
                indiv_losses = np.zeros(m, dtype=float)

                # ---- train m weak learners on independent r-subsets
                for i in range(m):
                    idx = rng_np.choice(n, size=r, replace=False)
                    mask_val = np.ones(n, dtype=bool)
                    mask_val[idx] = False

                    key_i, rng_run = jr.split(rng_run)
                    ctor = self.constructors[i % len(self.constructors)]
                    raw = ctor(key_i)
                    mdl, state0 = raw if isinstance(raw, tuple) else (raw, None)

                    # If subset is single-class, use constant fallback
                    if self.task == "binary" and len(np.unique(y_int[idx])) < 2:
                        const_logit = 10.0 if (y_int[idx][0] == 1) else -10.0
                        mdl = _ConstantEqx(const_logit)
                        state0 = None
                        opt_state = self.optimizer.init(
                            eqx.filter(mdl, eqx.is_inexact_array)
                        )
                        mdl_tr, state_tr = mdl, state0
                    elif self.task == "multiclass" and len(np.unique(y_int[idx])) < 2:
                        const_class = int(np.unique(y_int[idx])[0])
                        mdl = _ConstantEqx(const_class, n_classes=len(classes))
                        state0 = None
                        opt_state = self.optimizer.init(
                            eqx.filter(mdl, eqx.is_inexact_array)
                        )
                        mdl_tr, state_tr = mdl, state0
                    else:
                        opt_state = self.optimizer.init(
                            eqx.filter(mdl, eqx.is_inexact_array)
                        )
                        mdl_tr, state_tr, *_ = _train(
                            mdl,
                            state0,
                            opt_state,
                            self.optimizer,
                            used_loss,
                            Xj[idx],
                            yj[idx],
                            Xj[mask_val],
                            yj[mask_val],
                            batch_size,
                            num_epochs,
                            patience,
                            key_i,
                        )

                    models_states.append((mdl_tr, state_tr))
                    val_masks.append(mask_val)

                # ---- compute individual 0-1 holdout losses
                pred_key, rng_run = jr.split(rng_run)
                keys = jr.split(pred_key, m)

                if self.task == "binary":
                    y_pm1 = y_int  # {-1,+1}
                    for i, ((mdl, st), k) in enumerate(zip(models_states, keys)):
                        mask = val_masks[i]
                        logit = _predict(mdl, st, Xj[mask], k).ravel()
                        pred_pm1 = np.where(np.asarray(logit) >= 0.0, 1, -1)
                        indiv_losses[i] = np.mean(pred_pm1 != y_pm1[mask])
                else:
                    for i, ((mdl, st), k) in enumerate(zip(models_states, keys)):
                        mask = val_masks[i]
                        logits = _predict(mdl, st, Xj[mask], k)
                        pred_idx = np.asarray(jnp.argmax(logits, axis=-1))
                        indiv_losses[i] = np.mean(pred_idx != y_int[mask])

                # ---- tandem co-error on intersections if requested
                if self.bound_type == "tandem":
                    pair_losses = np.zeros((m, m), dtype=float)
                    min_inter = n
                    # cache each model's *label* predictions on full X for speed
                    full_keys = jr.split(rng_run, m)
                    cached_idx = []
                    for (mdl, st), k in zip(models_states, full_keys):
                        if self.task == "binary":
                            logit = _predict(mdl, st, Xj, k).ravel()
                            cached_idx.append((logit >= 0.0).astype(int))
                        else:
                            logits = _predict(mdl, st, Xj, k)
                            cached_idx.append(np.asarray(jnp.argmax(logits, axis=-1)))
                    for i in range(m):
                        for j in range(i, m):
                            mask = val_masks[i] & val_masks[j]
                            sz = int(mask.sum())
                            min_inter = min(min_inter, sz)
                            if sz == 0:
                                cij = 0.0
                            else:
                                err_i = cached_idx[i][mask] != (
                                    y_pm1[mask] > 0
                                    if self.task == "binary"
                                    else y_int[mask]
                                )
                                err_j = cached_idx[j][mask] != (
                                    y_pm1[mask] > 0
                                    if self.task == "binary"
                                    else y_int[mask]
                                )
                                cij = float(np.mean(err_i & err_j))
                            pair_losses[i, j] = pair_losses[j, i] = cij
                else:
                    pair_losses, min_inter = None, None

                # ---- alternate: update ρ (and λ when applicable)
                rho = np.full(m, 1.0 / m, dtype=float)
                lam = 0.5 if getattr(criterion, "uses_lambda", False) else None
                log_const = math.log(2.0 * math.sqrt(n_r) / self.delta)
                prev_bound = float("inf")

                for _ in range(200):
                    # KL(ρ || π=uniform) with zero-skip handled by shift in rho updates
                    nz = rho > 0
                    kl_rho = float((rho[nz] * np.log(rho[nz] * m)).sum())

                    if self.bound_type == "tandem":
                        stat, bound = criterion.compute(
                            pair_losses, rho, kl_rho, min_inter, self.delta, lam
                        )
                    else:
                        stat, bound = criterion.compute(
                            indiv_losses, rho, kl_rho, n_r, self.delta, lam
                        )

                    if abs(prev_bound - bound) < 1e-6:
                        break
                    prev_bound = bound

                    # update λ if needed
                    if getattr(criterion, "uses_lambda", False):
                        lam = 2.0 / (
                            math.sqrt(1.0 + 2.0 * n_r * stat / (kl_rho + log_const))
                            + 1.0
                        )

                    # update ρ to Gibbs-like posterior (monotone for λ-bounds; heuristic otherwise)
                    shift = float(indiv_losses.min())
                    rho = np.exp(-(lam or 1.0) * n_r * (indiv_losses - shift))
                    s = rho.sum()
                    rho = rho / s if s > 0 else np.full(m, 1.0 / m)

                # ---- empirical MV error on *full* training set (diagnostic)
                mv_idx = self._mv_predict_idx(Xj, models_states, rho, rng_run)
                emp_mv = (
                    np.mean((mv_idx != (y_int > 0).astype(int)))
                    if self.task == "binary"
                    else np.mean(mv_idx != y_int)
                )

                results.append(
                    dict(
                        m=m,
                        bound=float(bound),
                        emp_mv=float(emp_mv),
                        time=time.time() - t0,
                    )
                )

                if verbose:
                    print(
                        f"[run {run+1}/{n_runs} | m={m:3d}]  bound={float(bound):.6f}  empMV={emp_mv:.4f}"
                    )

                if float(bound) < best_bound:
                    best_bound = float(bound)
                    best_models = [
                        (copy.deepcopy(mdl), copy.deepcopy(st))
                        for (mdl, st) in models_states
                    ]
                    best_rho = rho.copy()
                    best_m = m

        # commit best
        object.__setattr__(self, "best_models_", best_models)
        object.__setattr__(self, "best_rho_", best_rho)
        object.__setattr__(self, "best_m_", best_m)
        object.__setattr__(self, "best_bound_", best_bound)
        object.__setattr__(self, "results_", results)
        return self

    # ------------------------------ inference ------------------------------

    def predict(self, X: np.ndarray, *, seed: Optional[int] = None) -> np.ndarray:
        if self.best_models_ is None:
            raise RuntimeError("Call fit() before predict().")
        Xj = jnp.asarray(X, dtype=jnp.float32)
        key = jr.PRNGKey(self.seed if seed is None else seed)
        idx = self._mv_predict_idx(Xj, self.best_models_, self.best_rho_, key)

        if self.task == "binary":
            # map internal 0/1 indices back to external labels order in self.classes_
            neg, pos = self.classes_[0], self.classes_[1]
            return np.where(idx == 1, pos, neg)
        # multiclass
        return self.classes_[idx]

    def predict_proba(self, X: np.ndarray, *, seed: Optional[int] = None) -> np.ndarray:
        if self.best_models_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        Xj = jnp.asarray(X, dtype=jnp.float32)
        key = jr.PRNGKey(self.seed if seed is None else seed)
        P = self._proba_aggregate(Xj, self.best_models_, self.best_rho_, key)

        if self.task == "binary":
            # columns must align to self.classes_ order
            if set(self.classes_) == {0, 1}:
                # P computed as [neg,pos] in 0/1 order already
                return P
            # if classes are {-1,1}, reorder columns accordingly
            out = np.zeros_like(P)
            neg_idx = int(np.where(self.classes_ == -1)[0][0])
            pos_idx = int(np.where(self.classes_ == 1)[0][0])
            out[:, neg_idx] = P[:, 0]
            out[:, pos_idx] = P[:, 1]
            return out

        return P

    # ------------------------------ reporting ------------------------------

    def summary(self):
        if not self.results_:
            print("No results. Did you call fit()?")
            return
        header = f"{'m':>4s} | {'bound':>10s} | {'empMV':>8s} | {'time[s]':>8s}"
        print(header)
        print("-" * len(header))
        # aggregate by m
        by_m: Dict[int, List[Dict]] = {}
        for r in self.results_:
            by_m.setdefault(r["m"], []).append(r)
        for m, rows in sorted(by_m.items()):
            b = np.array([r["bound"] for r in rows])
            e = np.array([r["emp_mv"] for r in rows])
            t = np.array([r["time"] for r in rows])
            print(
                f"{m:4d} | {b.mean():10.6f}±{b.std():.6f} | {e.mean():8.4f}±{e.std():.4f} | {t.mean():8.3f}±{t.std():.3f}"
            )
        print(
            f"\nBest m = {self.best_m_}  |  Best certified bound = {self.best_bound_:.6f}"
        )
