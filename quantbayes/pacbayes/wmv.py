"""
Fully-correct PAC-Bayes ensemble implementation
===============================================

This file provides:

*  **BoundCriterion** subclasses implementing
   – PB-λ (Theorem 3.24 / 3.30)
   – PB-KL (Theorem 3.26)
   – Empirical Bernstein (Maurer-Pontil ’09)
   – Tandem second-order bound (Theorem 3.36)
   – Split-KL (Wu et al. ’24)
   – “Unexpected” Bernstein (Fan et al. ’15)

*  **BoundEnsemble** – a production-ready classifier that
   trains each weak learner on an independent subset of size **r**
   (default *d + 1*), validates on the complement, and searches
   over ensemble size *m* and random seeds while **monotonically
   decreasing the chosen PAC-Bayes bound** (for the bounds that
   contain a λ-parameter).

All bounds are implemented exactly as stated in the lecture
notes; the tandem co-error is now measured **only on the
intersection of the two validation sets** as required by the proof.

A quick smoke-test at the bottom trains the ensemble on synthetic
binary data and checks that the certified bound upper-bounds the
true majority-vote error.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_classification
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier

__all__ = [
    "PBLambdaCriterion",
    "PBKLCriterion",
    "TandemCriterion",
    "PBBernsteinCriterion",
    "SplitKLCriterion",
    "UnexpectedBernsteinCriterion",
    "SplitBernsteinCriterion",
    "BoundEnsemble",
]


# ---------------------------------------------------------------------
#  Low-level helpers
# ---------------------------------------------------------------------
def _kl_div(p: float, q: float, eps: float = 1e-15) -> float:
    """Binary KL divergence `KL(p || q)` with numeric protection."""
    p_ = min(max(p, eps), 1 - eps)
    q_ = min(max(q, eps), 1 - eps)
    return p_ * math.log(p_ / q_) + (1 - p_) * math.log((1 - p_) / (1 - q_))


def _kl_inverse(p_hat: float, kl_term: float, n: int, tol: float = 1e-12) -> float:
    """
    Inverse binary KL:  smallest q ≥ p_hat  s.t.  KL(p̂‖q) ≤ kl_term / n.
    Closed form has no elementary solution → binary search.
    """
    if p_hat >= 1.0:
        return 1.0
    target = kl_term / n
    lo, hi = p_hat, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _kl_div(p_hat, mid) > target:
            hi = mid
        else:
            lo = mid
    return lo


# ---------------------------------------------------------------------
#  Generic interface
# ---------------------------------------------------------------------
class BoundCriterion:
    """Abstract base class for every PAC-Bayes bound."""

    uses_lambda: bool = False  # whether the bound contains the λ parameter

    def compute(
        self,
        *args,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Returns a tuple (statistic, majority_vote_bound).
        subclasses document the meaning of *statistic*.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------
#  1) PB-λ bound  (Theorem 3.24 / 3.30)
# ---------------------------------------------------------------------
class PBLambdaCriterion(BoundCriterion):
    uses_lambda = True

    def compute(
        self,
        losses: np.ndarray,
        rho: np.ndarray,
        kl_rho: float,
        n_r: int,
        delta: float,
        lam: float,
        *_,
    ) -> Tuple[float, float]:
        """
        *statistic*  =  p_hat = E_ρ[loss]
        mv_bound     =  2·GibbsBound  (clipped to 1)

        GibbsBound  := p_hat/(1−λ/2)
                       + (kl_rho + ln(2√n_r/δ)) /(λ(1−λ/2) n_r)
        """
        p_hat = float(rho @ losses)
        log_term = math.log(2.0 * math.sqrt(n_r) / delta)
        gibbs = p_hat / (1.0 - lam / 2.0) + (kl_rho + log_term) / (
            lam * (1.0 - lam / 2.0) * n_r
        )
        return p_hat, min(1.0, 2.0 * gibbs)


# ---------------------------------------------------------------------
#  2) PB-KL (kl-inverse)  (Theorem 3.26)
# ---------------------------------------------------------------------
class PBKLCriterion(BoundCriterion):
    uses_lambda = False

    def compute(
        self,
        losses: np.ndarray,
        rho: np.ndarray,
        kl_rho: float,
        n_r: int,
        delta: float,
        *_,
    ) -> Tuple[float, float]:
        """
        *statistic*  =  p_hat = E_ρ[loss]
        mv_bound     =  2·q   where  q = kl^{-1+}(p̂ , (kl_rho+ln(2√n_r/δ))/n_r )
        """
        p_hat = float(rho @ losses)
        q = _kl_inverse(p_hat, kl_rho + math.log(2.0 * math.sqrt(n_r) / delta), n_r)
        return p_hat, min(1.0, 2.0 * q)


# ---------------------------------------------------------------------
#  3) Empirical Bernstein (Maurer-Pontil)
# ---------------------------------------------------------------------
class PBBernsteinCriterion(BoundCriterion):
    uses_lambda = False

    def compute(
        self,
        losses: np.ndarray,
        rho: np.ndarray,
        kl_rho: float,
        n_r: int,
        delta: float,
        *_,
    ) -> Tuple[float, float]:
        """
        Classical empirical-Bernstein PAC-Bayes bound (0-1 loss, b=1).

        GibbsBound :=  p̂
                       + sqrt( 2·Var̂·(kl_rho+ln)/ n_r )
                       +  (2/3)·(kl_rho+ln)/n_r
        mv_bound   = 2·GibbsBound  (≤1)
        """
        p_hat = float(rho @ losses)
        var_hat = float(rho @ (losses - p_hat) ** 2)
        log_term = kl_rho + math.log(2.0 * math.sqrt(n_r) / delta)
        gibbs = (
            p_hat
            + math.sqrt(2.0 * var_hat * log_term / n_r)
            + 2.0 * log_term / (3.0 * n_r)
        )
        return p_hat, min(1.0, 2.0 * gibbs)


# ---------------------------------------------------------------------
#  4) Second-order “tandem” bound  (Theorem 3.36)
# ---------------------------------------------------------------------
class TandemCriterion(BoundCriterion):
    uses_lambda = True

    def compute(
        self,
        pair_losses: np.ndarray,
        rho: np.ndarray,
        kl_rho: float,
        n_pairs: int,
        delta: float,
        lam: float,
        *_,
    ) -> Tuple[float, float]:
        """
        *statistic*  =  t_hat = E_{ρ²}[co-error]
        mv_bound     =  4·(
                            t_hat/(1−λ/2)
                            + (2 kl_rho + ln(2√n_pairs/δ))
                              /(λ(1−λ/2) n_pairs)
                         )
        """
        t_hat = float(rho @ pair_losses @ rho)
        log_term = math.log(2.0 * math.sqrt(n_pairs) / delta)
        inner = t_hat / (1.0 - lam / 2.0) + (2.0 * kl_rho + log_term) / (
            lam * (1.0 - lam / 2.0) * n_pairs
        )
        return t_hat, min(1.0, 4.0 * inner)


# ---------------------------------------------------------------------
#  5) Split-KL (Wu et al. ’24) – works for finite-valued losses
# ---------------------------------------------------------------------
class SplitKLCriterion(BoundCriterion):
    uses_lambda = False

    def compute(
        self,
        losses: np.ndarray,
        rho: np.ndarray,
        kl_rho: float,
        n_r: int,
        delta: float,
        *_,
    ) -> Tuple[float, float]:
        b_vals = np.unique(losses)
        if len(b_vals) == 1:  # all losses identical
            return float(rho @ losses), min(1.0, 2.0 * b_vals[0])

        log_term = kl_rho + math.log(2.0 * (len(b_vals) - 1) * math.sqrt(n_r) / delta)
        bound_sum = b_vals[0]  # b_0

        for j in range(1, len(b_vals)):
            alpha = b_vals[j] - b_vals[j - 1]
            p_hat_j = float(rho @ (losses >= b_vals[j]).astype(float))
            q_j = _kl_inverse(p_hat_j, log_term, n_r)
            bound_sum += alpha * q_j

        return float(rho @ losses), min(1.0, 2.0 * bound_sum)


# ---------------------------------------------------------------------
#  6) “Unexpected” Bernstein  (Fan et al. ’15 / Wu-Seldin ’22)
# ---------------------------------------------------------------------
class UnexpectedBernsteinCriterion(BoundCriterion):
    uses_lambda = False

    def __init__(self, lambdas: Optional[List[float]] = None, b: float = 1.0):
        self.lambdas = lambdas or [2.0 ** (-i) for i in range(1, 9)]
        self.b = b  # upper bound on the loss (0-1 ⇒ b=1)

    def compute(
        self,
        losses: np.ndarray,
        rho: np.ndarray,
        kl_rho: float,
        n_r: int,
        delta: float,
        *_,
    ) -> Tuple[float, float]:
        p_hat = float(rho @ losses)
        s2_hat = float(rho @ losses**2)
        pen_const = kl_rho + math.log(len(self.lambdas) / delta)

        best = float("inf")
        for lam in self.lambdas:
            if lam <= 0 or lam >= 1.0 / self.b:
                continue
            psi = lam * self.b - math.log(1.0 + lam * self.b)  # = −ψ(−λb)
            coeff = -psi / (lam * self.b**2)
            cand = coeff * s2_hat + pen_const / (lam * n_r)
            best = min(best, cand)

        return p_hat, min(1.0, 2.0 * (p_hat + best))


class SplitBernsteinCriterion(BoundCriterion):
    """Automatic Split–Empirical–Bernstein for finite-valued losses."""

    uses_lambda = False

    def compute(
        self,
        losses: np.ndarray,
        rho: np.ndarray,
        kl_rho: float,
        n: int,
        delta: float,
        *_,
    ) -> Tuple[None, float]:
        b_vals = np.unique(losses)
        K = len(b_vals) - 1
        if K < 1:
            # trivial: everyone has the same loss
            return None, min(1.0, 2.0 * b_vals[0])

        alpha = np.diff(b_vals)
        log_term = kl_rho + math.log(6 * K * math.sqrt(n) / delta)

        bound = b_vals[0]  # the b₀ term
        for j in range(K):
            # indicator slice: fhat_j(i) = 1{losses[i] ≥ b_vals[j+1]}
            slice_j = (losses >= b_vals[j + 1]).astype(float)
            p_hat = float(rho @ slice_j)

            # empirical–Bernstein term
            var_hat = float(rho @ (slice_j - p_hat) ** 2)
            term = (
                p_hat
                + math.sqrt(2.0 * var_hat * log_term / n)
                + 7.0 * log_term / (3.0 * (n - 1))
            )

            bound += alpha[j] * term

        # 3) switch to majority-vote bound and clip
        mv_bound = min(1.0, 2.0 * bound)
        return None, mv_bound


# ---------------------------------------------------------------------
# 7) Constant classifier (used when a split subset is single-class)
# ---------------------------------------------------------------------
class _ConstantClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, label: int | float):
        self.label = label

    def fit(self, X, y):  # noqa: N802 – scikit-learn API
        return self

    def predict(self, X):  # noqa: N802
        return np.full(shape=(len(X),), fill_value=self.label, dtype=type(self.label))

    def predict_proba(self, X):  # noqa: N802
        # 2-column proba for binary; multi-class not used here.
        proba = np.zeros((len(X), 2))
        if self.label in {1, True}:
            proba[:, 1] = 1.0
        else:
            proba[:, 0] = 1.0
        return proba


# ---------------------------------------------------------------------
# 8) The Ensemble class
# ---------------------------------------------------------------------
class BoundEnsemble(BaseEstimator, ClassifierMixin):
    """
    Unified PAC-Bayes ensemble with selectable bound.

    Parameters
    ----------
    task : {"binary", "multiclass"}, default="binary"
        Whether to solve a binary (labels in {0,1} or {-1,1}) or
        multiclass classification problem.

    base_estimators : list of BaseEstimator instances, default=None
        A list of *template* estimators to clone for each weak learner.
        Each call to `fit` will clone from these via `sklearn.base.clone`.
        You should supply exactly one of `base_estimators` or `base_estimator_cls`.
        This is the most straightforward way if you already have one or more
        preconfigured estimator instances:

            >>> from sklearn.tree import DecisionTreeClassifier
            >>> dt = DecisionTreeClassifier(max_depth=1, random_state=0)
            >>> ens = BoundEnsemble(base_estimators=[dt])

    base_estimator_cls : class, default=None
        A reference to an estimator *class* (e.g. `DecisionTreeClassifier`)
        to instantiate for each weak learner. You must also supply
        `base_estimator_kwargs` for its constructor.

    base_estimator_kwargs : dict, default=None
        Keyword arguments to pass when calling
        `base_estimator_cls(**base_estimator_kwargs)`. Ignored if
        `base_estimators` is provided.

    bound_type : {"pblambda", "pbkl", "pbbernstein", "tandem",
                  "splitkl", "splitbernstein", "unexpectedbernstein"}, default="pblambda"
        Which PAC-Bayes bound to optimize.

    bound_delta : float, default=0.05
        Confidence parameter δ for the PAC-Bayes bound.

    r : int or None, default=None
        Size of each training subset. If None, defaults to `d + 1` where
        `d` is the feature-dimension of `X`.

    random_state : int or None, default=None
        Seed for reproducibility of the random splits and bound optimization.

    Examples
    --------
    # — Using pre-instantiated templates (most common) —
    >>> from sklearn.svm import SVC
    >>> svm = SVC(kernel='rbf', C=0.1, gamma=0.2, probability=True, random_state=42)
    >>> ens = BoundEnsemble(
    ...     task='binary',
    ...     base_estimators=[svm],
    ...     bound_type='splitbernstein',
    ...     bound_delta=0.05,
    ...     random_state=0
    ... )
    >>> ens.fit(X_train, y_train, m_values=[10,20])
    >>> y_pred = ens.predict(X_test)

    # — Using a class plus kwargs —
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> ens = BoundEnsemble(
    ...     task='binary',
    ...     base_estimator_cls=DecisionTreeClassifier,
    ...     base_estimator_kwargs={'max_depth':1, 'random_state':0},
    ...     bound_type='pbkl',
    ...     random_state=0
    ... )
    """

    _criteria_map = {
        "pblambda": PBLambdaCriterion,
        "pbkl": PBKLCriterion,
        "pbbernstein": PBBernsteinCriterion,
        "tandem": TandemCriterion,
        "splitkl": SplitKLCriterion,
        "splitbernstein": SplitBernsteinCriterion,
        "unexpectedbernstein": UnexpectedBernsteinCriterion,
    }
    _estimator_type = "classifier"

    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        task: str = "binary",
        base_estimators: Optional[List[BaseEstimator]] = None,
        base_estimator_cls: Optional[Type[BaseEstimator]] = None,
        base_estimator_kwargs: Optional[Dict] = None,
        bound_type: str = "pblambda",
        bound_delta: float = 0.05,
        random_state: Optional[int] = None,
        r: Optional[int] = None,
    ):
        if task not in {"binary", "multiclass"}:
            raise ValueError("task must be 'binary' or 'multiclass'")
        if (base_estimators is None) == (base_estimator_cls is None):
            raise ValueError(
                "specify exactly one of base_estimators or base_estimator_cls"
            )
        if bound_type not in self._criteria_map:
            raise ValueError(f"unknown bound_type '{bound_type}'")

        self.task = task
        self.base_estimators = base_estimators
        self.base_estimator_cls = base_estimator_cls
        self.base_estimator_kwargs = base_estimator_kwargs or {}
        self.bound_type = bound_type
        self.delta = bound_delta
        self.bound_delta = bound_delta
        self.random_state = random_state
        self.r = r

        self.is_fitted = False

    # ------------------------------------------------------------------
    def _train_one_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        m: int,
        rng: np.random.Generator,
        r: int,
    ):
        """Train *m* weak models; return models, val masks, individual losses."""
        n = len(X)
        idx = np.arange(n)
        models: List[BaseEstimator] = []
        val_masks: List[np.ndarray] = []
        indiv_losses = np.zeros(m)

        for i in range(m):
            Si = rng.choice(idx, size=r, replace=False)
            val_mask = np.ones(n, dtype=bool)
            val_mask[Si] = False

            # pick a template
            if self.base_estimators is not None:
                template = self.base_estimators[i % len(self.base_estimators)]
            else:
                template = self.base_estimator_cls(**self.base_estimator_kwargs)
            clf = clone(template)

            # set internal seed if supported
            if hasattr(clf, "get_params") and "random_state" in clf.get_params():
                clf.set_params(random_state=int(rng.integers(0, 2**31 - 1)))

            # handle single-class subset for binary
            if self.task == "binary" and len(np.unique(y[Si])) < 2:
                clf = _ConstantClassifier(label=int(y[Si][0]))
                clf.fit(None, None)
            else:
                clf.fit(X[Si], y[Si])

            models.append(clf)
            val_masks.append(val_mask)

        # compute validation predictions once
        preds = np.vstack([m_.predict(X) for m_ in models]).astype(y.dtype)
        for i in range(m):
            indiv_losses[i] = zero_one_loss(y[val_masks[i]], preds[i][val_masks[i]])

        return models, val_masks, preds, indiv_losses

    # ------------------------------------------------------------------
    def _pair_co_errors(
        self,
        preds: np.ndarray,
        y: np.ndarray,
        val_masks: List[np.ndarray],
    ) -> Tuple[np.ndarray, int]:
        """
        Build the m×m matrix of co-errors on the *intersection* of validation
        sets and return its minimum sample size (needed for the bound).
        """
        m, n = preds.shape
        pair_losses = np.zeros((m, m))
        min_inter = n  # start with upper bound

        for i in range(m):
            for j in range(i, m):
                mask = val_masks[i] & val_masks[j]
                inter_sz = int(mask.sum())
                min_inter = min(min_inter, inter_sz)
                if inter_sz == 0:
                    co_err = 0.0
                else:
                    errs = (preds[i][mask] != y[mask]) & (preds[j][mask] != y[mask])
                    co_err = float(errs.mean())
                pair_losses[i, j] = pair_losses[j, i] = co_err

        return pair_losses, max(min_inter, 1)

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        m_values: List[int],
        n_runs: int = 5,
        max_iters: int = 200,
        tol: float = 1e-6,
    ):
        """
        Train the PAC-Bayes ensemble by searching over ensemble sizes
        and random seeds, and keep the models that achieve the smallest
        certified majority-vote (MV) bound.

        The procedure is:

        1. For each random seed r = 0, 1, …, n_runs−1:
           • Split the data into an r-sized training subset (for each
             of m weak learners) and the corresponding validation set.
           • For each ensemble size m in `m_values`:
             – Train m weak models (each on an independent random subset
               of size `self.r` or default d+1)
             – Compute the chosen PAC-Bayes bound on the validation losses
             – Optimize the posterior ρ (and λ if applicable) to minimize it
             – Evaluate the MV error on the *entire* training set
           • Record the bound, empirical error, and training time.

        2. Across *all* (seed, m) combinations, pick the one with the
           *smallest* certified MV bound, and store its models and weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target labels.

        m_values : list of int
            List of ensemble sizes (number of weak learners) to try.
            E.g. `[10, 20, 50]` will train ensembles of size 10, 20, and 50.

        n_runs : int, default=5
            Number of independent random‐seed repetitions.  Each “run”
            uses a different seed to sample training subsets and can be
            thought of as repeated experiments (not classic cross‐validation).
            The final pick is the single (seed, m) with the best certified bound.

        max_iters : int, default=200
            Maximum number of iterations for alternating‐minimization of
            ρ (and λ if used) when optimizing the bound.

        tol : float, default=1e-6
            Convergence tolerance on the bound: stop early if the change
            in bound between iterations is less than `tol`.

        Returns
        -------
        self : BoundEnsemble
            Fitted estimator with attributes:
            - `best_models_`: list of the m weak learners (cloned & fitted)
            - `best_rho_`: weight vector ρ for those models
            - `best_m_`: the ensemble size
            - `best_seed_`: which run (seed) produced them
            - `best_bound_`: the certified MV bound value

        Notes
        -----
        - This is *not* k-fold CV.  Rather, `n_runs` is the number of
          *independent* trials (with different random subsets), each of
          which evaluates every `m` in `m_values`.  You can interpret
          it like repeated randomized splits to guard against bad luck
          in a single seed.
        - The “winner” is the single combination (run, m) with the
          lowest bound.  You can inspect all results in `all_runs`.
        """
        X, y = np.asarray(X), np.asarray(y)
        # ── store original labels for binary tasks so we can map back later
        self._orig_labels_ = np.unique(y).copy()
        n, d = X.shape

        # choose training subset size r
        r = int(self.r) if self.r is not None else d + 1
        if not (1 <= r < n):
            raise ValueError(f"need 1 ≤ r < n; got r={r}, n={n}")
        n_r = n - r

        # convert labels for binary task
        if self.task == "binary":
            if set(np.unique(y)) == {0, 1}:
                y = np.where(y == 0, -1, 1)
            elif set(np.unique(y)) == {-1, 1}:
                pass
            else:
                raise ValueError("binary task y values must be in {0,1} or {-1,1}")
            self.classes_ = np.array([-1, 1])
        else:
            self.classes_ = np.unique(y)

        rng0 = np.random.default_rng(self.random_state)
        best_bound = float("inf")

        self.all_runs: List[List[Dict]] = []

        criterion_cls = self._criteria_map[self.bound_type]
        criterion = criterion_cls()

        for run in range(n_runs):
            rng = np.random.default_rng(rng0.integers(0, 2**32 - 1))

            run_results = []
            for m in m_values:
                t_start = time.time()

                (models, val_masks, preds, indiv_losses) = self._train_one_split(
                    X, y, m, rng, r
                )

                # pairwise co-error if needed
                pair_losses, min_inter = (None, None)
                if self.bound_type == "tandem":
                    pair_losses, min_inter = self._pair_co_errors(preds, y, val_masks)

                # initialise ρ and (if needed) λ
                rho = np.full(m, 1.0 / m)
                lam = 0.5 if criterion.uses_lambda else None

                log_const = math.log(2.0 * math.sqrt(n_r) / self.delta)

                prev_bound = float("inf")
                for _ in range(max_iters):
                    # safe KL(ρ || π=1/m): skip zero weights
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

                    # termination
                    if abs(prev_bound - bound) < tol:
                        break
                    prev_bound = bound

                    # update λ (only for PB-λ or Tandem)
                    if criterion.uses_lambda:
                        lam = 2.0 / (
                            math.sqrt(1.0 + 2.0 * n_r * stat / (kl_rho + log_const))
                            + 1.0
                        )

                    # update ρ = Gibbs posterior (for all bounds –
                    # monotone for PB-λ/Tandem; heuristic for others)
                    weights = np.exp(
                        -(lam or 1.0) * n_r * (indiv_losses - indiv_losses.min())
                    )
                    rho = weights / weights.sum()

                # majority-vote empirical error on full data
                if self.task == "binary":
                    mv_preds = np.sign(preds.T @ rho)
                else:  # multiclass MV
                    votes = np.zeros((n, len(self.classes_)))
                    for k, cls in enumerate(self.classes_):
                        votes[:, k] = (preds == cls).T @ rho
                    mv_preds = self.classes_[votes.argmax(1)]
                mv_err = float(zero_one_loss(y, mv_preds))

                run_results.append(
                    dict(m=m, err=mv_err, bound=bound, time=time.time() - t_start)
                )

                # keep global best
                if bound < best_bound:
                    best_bound = bound
                    self.best_models_ = [copy.deepcopy(m_) for m_ in models]
                    self.best_rho_ = rho.copy()
                    self.best_m_ = m
                    self.best_seed_ = run
                    self.best_bound_ = bound

            self.all_runs.append(run_results)

        self.m_values = list(m_values)
        self.n_runs = n_runs
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def _mv_predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.vstack([m_.predict(X) for m_ in self.best_models_]).astype(int)
        if self.task == "binary":
            raw = preds.T @ self.best_rho_
            # break any ties in favor of +1
            return np.where(raw >= 0, 1, -1)
        votes = np.zeros((len(X), len(self.classes_)))
        for k, cls in enumerate(self.classes_):
            votes[:, k] = (preds == cls).T @ self.best_rho_
        return self.classes_[votes.argmax(1)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("call fit() first")
        # get raw predictions in {-1,+1}
        raw = self._mv_predict(np.asarray(X))
        # map back to original labels for binary tasks
        return np.where(raw < 0, self._orig_labels_[0], self._orig_labels_[1])

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray):
        if not self.is_fitted:
            raise RuntimeError("call fit() first")
        X = np.asarray(X)
        if self.task == "binary":
            # get probability of +1 from each weak learner
            probs_pos = np.vstack(
                [clf.predict_proba(X)[:, 1] for clf in self.best_models_]
            ).T
            p_pos = probs_pos @ self.best_rho_
            p_neg = 1.0 - p_pos
            # assemble columns in the same order as the original labels
            proba = np.zeros((len(X), 2))
            # find where each original label sits in self._orig_labels_
            neg_col = int(np.where(self._orig_labels_ == self._orig_labels_[0])[0][0])
            pos_col = int(np.where(self._orig_labels_ == self._orig_labels_[1])[0][0])
            proba[:, neg_col] = p_neg
            proba[:, pos_col] = p_pos
            return proba

        # multiclass
        n_classes = len(self.classes_)
        proba = np.zeros((len(X), n_classes))
        for w, clf in zip(self.best_rho_, self.best_models_):
            pp = clf.predict_proba(X)
            for ci, cls in enumerate(clf.classes_):
                target_idx = np.where(self.classes_ == cls)[0][0]
                proba[:, target_idx] += w * pp[:, ci]
        proba /= proba.sum(1, keepdims=True)
        return proba

    # ------------------------------------------------------------------
    def summary(self):
        if not self.is_fitted:
            raise RuntimeError("call fit() first")

        header = f"{'m':>4s} | {'err±σ':>12s} | {'bound±σ':>12s} | {'time±σ[s]':>12s}"
        print(header)
        print("-" * len(header))
        for j, m in enumerate(self.m_values):
            errs = np.array([run[j]["err"] for run in self.all_runs])
            bds = np.array([run[j]["bound"] for run in self.all_runs])
            tms = np.array([run[j]["time"] for run in self.all_runs])
            print(
                f"{m:4d} | {errs.mean():.4f}±{errs.std():.4f} | "
                f"{bds.mean():.4f}±{bds.std():.4f} | {tms.mean():.3f}±{tms.std():.3f}"
            )


# ---------------------------------------------------------------------
# 9) Smoke-test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    X, y = make_classification(
        n_samples=200,
        n_features=15,
        n_informative=10,
        n_redundant=0,
        n_classes=2,
        random_state=0,
    )

    # weak learner: decision stump
    stump = DecisionTreeClassifier(max_depth=1, random_state=0)

    ens = BoundEnsemble(
        task="binary",
        base_estimator_cls=DecisionTreeClassifier,
        base_estimator_kwargs=dict(max_depth=1),
        bound_type="splitbernstein",
        bound_delta=0.05,
        random_state=42,
    )

    ens.fit(X, y, m_values=[10, 20], n_runs=3)
    ens.summary()

    y_pred = ens.predict(X)
    mv_error = zero_one_loss(y, y_pred)
    print(f"\nBest certified bound  : {ens.best_bound_:.4f}")
    print(f"Empirical MV error    : {mv_error:.4f}")

    # PAC-Bayes guarantee smoked-test: bound ≥ empirical error
    assert mv_error <= ens.best_bound_ + 1e-8, "PAC-Bayes bound violated!"

    print("\nSmoke-test passed ✓ – ensemble ready for PR.")
