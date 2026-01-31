"""
recursive_pac_bayes_ensemble.py
--------------------------------
Finite-ℋ Recursive PAC-Bayes (Wu et al., NeurIPS 2024) with two
theoretically-justified single-stage bounds:

    bound_type ∈ {
        "splitkl",         # paper-exact (default)
        "splitbernstein",  # variance-adaptive extension (this file)
    }

Stage 1 (t = 1) always uses the classical plain-KL inequality.
Stage t ≥ 2 uses the bound chosen via `bound_type`.

Author : Joseph Margaryan — updated 2025-06-29
"""

from __future__ import annotations

import math
from math import log, sqrt
from typing import List, Tuple, Sequence, Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state, check_X_y, check_array

# ---------------------------------------------------------------------
# 1)  low-level helpers
# ---------------------------------------------------------------------
_EPS = 1e-12


def _kl_binary(p: float, q: float, eps=_EPS) -> float:
    p, q = np.clip(p, eps, 1 - eps), np.clip(q, eps, 1 - eps)
    return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))


def _kl_inv_plus(p_hat: float, c: float, tol=1e-10) -> float:
    """Upper KL-inverse  kl⁻¹⁺(p̂ , c)."""
    if c <= 0:
        return p_hat
    lo, hi = p_hat, 1 - 1e-15
    for _ in range(60):
        mid = (lo + hi) * 0.5
        (_kl_binary(p_hat, mid) <= c and (lo := mid)) or (hi := mid)
        if hi - lo < tol:
            break
    return lo


# ---------------------------------------------------------------------
# 2)  single-stage criteria
# ---------------------------------------------------------------------
class PlainKLCriterion:
    """Seeger/Maurer kl-inverse (stage 1 only)."""

    uses_lambda = False

    def compute(self, losses, rho, kl, n, delta, *_):
        p_hat = float(rho @ losses)
        b = _kl_inv_plus(p_hat, (kl + log(2 * sqrt(n) / delta)) / n)
        return p_hat, b


class SplitKLCriterion:
    """Split-KL bound (Wu et al. §3)."""

    uses_lambda = False

    def __init__(self, gamma: float):
        self.b = (
            np.array([-gamma, 0.0, 1 - gamma, 1.0])
            if abs(gamma - 1.0) > 1e-12
            else np.array([-1.0, 0.0, 1.0])
        )
        self.alpha = np.diff(self.b)
        self.k = len(self.alpha)

    def compute(self, fhat_slices, rho, kl, n, delta, *_):
        c = (kl + log(2 * self.k * sqrt(n) / delta)) / n
        bound = self.b[0]
        for j, alpha_j in enumerate(self.alpha):
            p_hat = float(rho @ fhat_slices[j])
            bound += alpha_j * _kl_inv_plus(p_hat, c)
        return None, bound


class SplitBernsteinCriterion:
    """Split–Empirical–Bernstein (variance-adaptive)."""

    uses_lambda = False

    def __init__(self, gamma: float):
        # Four‐point support {b0,b1,b2,b3} = {−γ,0,1−γ,1}
        self.b = (
            np.array([-gamma, 0.0, 1.0 - gamma, 1.0])
            if abs(gamma - 1) > 1e-12
            else np.array([-1.0, 0.0, 1.0])
        )
        self.alpha = np.diff(self.b)  # length 3
        self.k = len(self.alpha)  # =3

    def compute(self, fhat_slices, rho, kl, n, delta, *_):
        # The PAC-Bayes “split-EB” penalty uses log(6 K √n / δ)
        log_term = kl + math.log(6 * self.k * math.sqrt(n) / delta)

        bound = float(self.b[0])  # start with b₀
        for j, α_j in enumerate(self.alpha):
            p_hat = float(rho @ fhat_slices[j])
            var_hat = float(rho @ (fhat_slices[j] - p_hat) ** 2)
            # empirical-Bernstein term (Maurer–Pontil):
            term = (
                p_hat
                + math.sqrt(2 * var_hat * log_term / n)
                + 7 * log_term / (3 * (n - 1))
            )
            bound += α_j * term

        # convert to majority-vote bound and clip to [0,1]
        mv_bound = max(0.0, min(1.0, 2 * bound))
        return None, mv_bound


# ---------------------------------------------------------------------
# 3)  top-level estimator
# ---------------------------------------------------------------------
class RecursivePACBayesEnsemble(BaseEstimator, ClassifierMixin):
    """
    Finite-ℋ Recursive PAC-Bayes with either split-KL or split-Bernstein.

    Parameters
    ----------
    base_learners : list[(str, estimator)]
        Unfitted scikit-learn estimators forming ℋ (trained on S₁).
    n_steps : int               Number of recursive steps T (≥ 2 recommended).
    bound_type : {"splitkl","splitbernstein"}
    delta : float               Overall confidence level.
    gamma_grid : sequence[float] Search grid for γ (default 0.1…0.9).
    prior_weights : ndarray|None Uniform if None.
    """

    _CRIT_MAP = {
        "splitkl": SplitKLCriterion,
        "splitbernstein": SplitBernsteinCriterion,
    }

    # -------------------------------- constructor -----------------
    def __init__(
        self,
        base_learners: List[Tuple[str, BaseEstimator]],
        *,
        n_steps: int = 2,
        bound_type: str = "splitkl",
        delta: float = 0.05,
        gamma_grid: Optional[Sequence[float]] = None,
        prior_weights: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        if bound_type not in self._CRIT_MAP:
            raise ValueError("bound_type must be 'splitkl' or 'splitbernstein'")
        self.base_learners = base_learners
        self.n_steps = int(n_steps)
        self.bound_type = bound_type
        self.delta = float(delta)
        self.gamma_grid = (
            np.asarray(gamma_grid, float)
            if gamma_grid is not None
            else np.linspace(0.1, 0.9, 9, dtype=float)
        )
        self.prior_weights = prior_weights
        self.random_state = random_state
        self.verbose = verbose

    # ----------------------------- helpers ------------------------
    def _log(self, *msg):
        if self.verbose:
            print(*msg)

    def _check_fitted(self):
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("Call fit() before using this estimator.")

    # ----------------------------- fit ----------------------------
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        self.classes_, y = np.unique(y, return_inverse=True)
        n = len(y)

        # split data
        perm = rng.permutation(n)
        chunk_sz = np.full(self.n_steps, n // self.n_steps, int)
        chunk_sz[: n % self.n_steps] += 1
        self._chunks_ = np.split(perm, np.cumsum(chunk_sz))[:-1]

        # train ℋ on S₁
        S1 = self._chunks_[0]
        self._hypotheses_ = []
        for name, proto in self.base_learners:
            est = clone(proto)
            if hasattr(est, "random_state"):
                est.random_state = rng.randint(1, 2**31 - 1)
            est.fit(X[S1], y[S1])
            self._hypotheses_.append((name, est))
        K = len(self._hypotheses_)

        # loss matrix
        L = np.vstack(
            [(clf.predict(X) != y).astype(float) for _, clf in self._hypotheses_]
        )

        # prior π₀
        prior = (
            np.full(K, 1 / K)
            if self.prior_weights is None
            else np.asarray(self.prior_weights, float)
        )
        prior /= prior.sum()

        # containers
        self._pi_list_, self._B_list_, self._gamma_list_ = [], [], []

        δ_step = self.delta / self.n_steps
        δ_grid = δ_step / max(1, len(self.gamma_grid))

        # ----------------  stage 1 (plain-KL) ---------------------
        crit1 = PlainKLCriterion()
        ℓ1 = L[:, S1].mean(1)

        def obj1(v):
            w = np.exp(v - v.max())
            w /= w.sum()
            kl = np.sum(w * np.log((w + _EPS) / prior))
            _, b = crit1.compute(ℓ1, w, kl, len(S1), δ_grid)
            return b

        res = minimize(obj1, np.zeros(K), method="Nelder-Mead")
        w_prev = np.exp(res.x - res.x.max())
        w_prev /= w_prev.sum()
        kl1 = np.sum(w_prev * np.log((w_prev + _EPS) / prior))
        _, B_prev = crit1.compute(L.mean(1), w_prev, kl1, n, δ_step)

        self._pi_list_.append(w_prev)
        self._B_list_.append(float(B_prev))
        self._gamma_list_.append(0.0)
        self._log(f"Stage 1  B₁ = {B_prev:.5f}")

        # ----------------  stages t ≥ 2 ---------------------------
        SplitCriterion = self._CRIT_MAP[self.bound_type]

        for t in range(2, self.n_steps + 1):
            St = self._chunks_[t - 1]
            Uval = np.concatenate(self._chunks_[t - 1 :])
            n_tr, n_val = len(St), len(Uval)

            best = (np.inf, None, None)  # (B_t, γ, w_t)

            for γ in self.gamma_grid:
                crit = SplitCriterion(γ)

                # binary slices
                b = crit.b
                α = crit.alpha

                def make_fhat(indices):
                    loss_prev = w_prev @ L[:, indices]
                    out = np.empty((len(α), K))
                    for j, thr in enumerate(b[1:]):
                        out[j] = (L[:, indices] - γ * loss_prev >= thr - 1e-12).mean(1)
                    return out

                F_tr, F_val = make_fhat(St), make_fhat(Uval)

                # inner optimisation
                def obj(v):
                    w = np.exp(v - v.max())
                    w /= w.sum()
                    kl = np.sum(w * np.log((w + _EPS) / (w_prev + _EPS)))
                    _, eps = crit.compute(F_tr, w, kl, n_tr, δ_grid)
                    return eps + γ * B_prev

                res = minimize(obj, np.log(w_prev + _EPS), method="Nelder-Mead")
                w_t = np.exp(res.x - res.x.max())
                w_t /= w_t.sum()

                kl_t = np.sum(w_t * np.log((w_t + _EPS) / (w_prev + _EPS)))
                _, eps_val = crit.compute(F_val, w_t, kl_t, n_val, δ_step)
                B_t = float(eps_val + γ * B_prev)

                if B_t < best[0]:
                    best = (B_t, γ, w_t)

            B_prev, γ_star, w_prev = best
            self._B_list_.append(B_prev)
            self._gamma_list_.append(γ_star)
            self._pi_list_.append(w_prev)
            self._log(f"Stage {t}  γ={γ_star:.3f}  B_{t}={B_prev:.5f}")

        self._is_fitted = True
        return self

    # ------------------------- inference --------------------------
    def _posterior_proba(self, X):
        self._check_fitted()
        X = check_array(X)
        w = self._pi_list_[-1]
        proba = np.zeros((len(X), len(self.classes_)))
        for wt, (_, clf) in zip(w, self._hypotheses_):
            p = clf.predict_proba(X)
            # align cols
            if not np.array_equal(clf.classes_, self.classes_):
                tmp = np.zeros_like(proba)
                for i, cls in enumerate(clf.classes_):
                    tmp[:, np.searchsorted(self.classes_, cls)] = p[:, i]
                p = tmp
            proba += wt * p
        proba = np.clip(proba, 1e-12, 1.0)
        proba /= proba.sum(1, keepdims=True)
        return proba

    def predict_proba(self, X):
        return self._posterior_proba(X)

    def predict(self, X):
        return self.classes_.take(self._posterior_proba(X).argmax(1))

    # ------------------------- public attrs -----------------------
    @property
    def posterior_weights_(self):
        self._check_fitted()
        return self._pi_list_[-1].copy()

    @property
    def risk_bounds_(self):
        self._check_fitted()
        return self._B_list_.copy()

    @property
    def gammas_(self):
        self._check_fitted()
        return self._gamma_list_.copy()


# ---------------------------------------------------------------------
# smoke-test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score

    X, y = make_classification(
        2000, 20, n_informative=15, n_redundant=2, random_state=0
    )

    base = [
        ("lr", LogisticRegression(max_iter=3000, solver="liblinear")),
        ("dt", DecisionTreeClassifier(max_depth=6)),
        ("nb", GaussianNB()),
    ]

    rpb = RecursivePACBayesEnsemble(
        base_learners=base,
        n_steps=3,
        bound_type="splitkl",  # try "splitkl" or "splitbernstein"
        delta=0.05,
        random_state=0,
        verbose=True,
    ).fit(X, y)

    acc = accuracy_score(y, rpb.predict(X))
    print("\n===== smoke-test =====")
    print("π_T =", np.round(rpb.posterior_weights_, 4))
    print("B_t =", [round(b, 4) for b in rpb.risk_bounds_])
    print("γ_t =", rpb.gammas_)
    print("train accuracy =", acc)
