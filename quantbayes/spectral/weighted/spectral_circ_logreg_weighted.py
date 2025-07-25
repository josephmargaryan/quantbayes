#!/usr/bin/env python3
"""
SpectralCirculantWeightedLogisticRegression
-------------------------------------------
Circulant‑parameterised logistic regression with per‑frequency ℓ₂ weights β_j.
"""

import time
import logging
import warnings
from typing import Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning


def _setup_logger(name: str, verbose: bool) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.INFO if verbose else logging.WARNING)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(
            logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
        )
        log.addHandler(h)
    return log


class SpectralCirculantWeightedLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Circulant‑parameterised logistic regression with per‑frequency ℓ₂ weights β_j.

    Parameters
    ----------
    padded_dim : int or None
        FFT length (>= original feature dim). If None, uses the original dim.
    K : int or None
        Number of low frequencies to keep. If None, keeps all.
    beta : array-like of shape (K,) or (2*k_half,), or None
        Per-frequency ℓ₂ weight(s). If None, all weights are 1.
    solver : str
        Passed to underlying `LogisticRegression`.
    C : float
        Inverse regularization strength for `LogisticRegression`.
    max_iter : int
        Maximum iterations for solver.
    tol : float
        Tolerance for solver convergence.
    random_state : int, RandomState instance or None
        Random seed.
    verbose : bool
        If True, prints solver progress.
    """

    def __init__(
        self,
        padded_dim: Optional[int] = None,
        K: Optional[int] = None,
        beta: Optional[Sequence[float]] = None,
        solver: str = "lbfgs",
        C: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        # Only store parameters here
        self.padded_dim = padded_dim
        self.K = K
        self.beta = beta
        self.solver = solver
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        # set up logger once
        self._log = _setup_logger(self.__class__.__name__, verbose)

    def _make_fft_features(self, X: np.ndarray):
        n, D = X.shape

        # determine padded dimension
        self.padded_dim_ = D if self.padded_dim is None else int(self.padded_dim)
        if self.padded_dim_ < D:
            raise ValueError("padded_dim must be >= original feature dim")

        # compute half-spectrum size
        self.k_half_ = self.padded_dim_ // 2 + 1

        # decide how many freqs to keep
        self.K_ = (
            self.k_half_ if self.K is None or self.K > self.k_half_ else int(self.K)
        )
        mask = np.arange(self.k_half_) < self.K_

        # zero-pad, FFT, and mask high freqs
        X_pad = np.zeros((n, self.padded_dim_), dtype=float)
        X_pad[:, :D] = X
        Xf = np.fft.rfft(X_pad, axis=1)
        Xf[:, ~mask] = 0.0

        # stack real & imag parts
        Phi = np.hstack([Xf.real, Xf.imag])

        # build per-feature sqrt(beta)
        M = 2 * self.k_half_
        if self.beta is None:
            sqrt_beta = np.ones(M, dtype=float)
        else:
            beta_arr = np.asarray(self.beta, dtype=float)
            beta_full = np.ones(M, dtype=float)
            if beta_arr.shape[0] == self.K_:
                beta_full[: self.K_] = beta_arr
                beta_full[self.k_half_ : self.k_half_ + self.K_] = beta_arr
            elif beta_arr.shape[0] == M:
                beta_full = beta_arr.copy()
            else:
                raise ValueError(
                    f"beta must have length {self.K_} or {M}, got {beta_arr.shape}"
                )
            sqrt_beta = np.sqrt(beta_full)

        # scale features
        return Phi / sqrt_beta[np.newaxis, :], sqrt_beta

    def fit(self, X, y):
        # validate and convert
        X, y = check_X_y(X, y, dtype=float)
        self.n_features_in_ = X.shape[1]

        # build FFT features & weights
        Phi_scaled, sqrt_beta = self._make_fft_features(X)

        # fit underlying logistic model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model_ = LogisticRegression(
                penalty="l2",
                C=self.C,
                solver=self.solver,
                max_iter=self.max_iter,
                tol=self.tol,
                fit_intercept=True,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            t0 = time.time()
            self.model_.fit(Phi_scaled, y)
            self._log.info(f"Solver converged in {time.time() - t0:.3f}s")

        # recover Fourier‑space coefficients
        coef = self.model_.coef_.ravel() / sqrt_beta
        self.F_real_ = coef[: self.k_half_].copy()
        self.F_imag_ = coef[self.k_half_ :].copy()
        self.intercept_ = float(self.model_.intercept_[0])
        self.classes_ = self.model_.classes_

        return self

    def _transform(self, X: np.ndarray):
        check_is_fitted(self, ["F_real_"])
        X = check_array(X, dtype=float)
        n, D = X.shape
        if D != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {D}")

        # zero-pad, FFT, and mask
        X_pad = np.zeros((n, self.padded_dim_), dtype=float)
        X_pad[:, :D] = X
        Xf = np.fft.rfft(X_pad, axis=1)
        mask = np.arange(self.k_half_) < self.K_
        Xf[:, ~mask] = 0.0

        Phi = np.hstack([Xf.real, Xf.imag])

        # apply beta scaling if provided
        if self.beta is not None:
            M = 2 * self.k_half_
            beta_arr = np.asarray(self.beta, dtype=float)
            beta_full = np.ones(M, dtype=float)
            if beta_arr.shape[0] == self.K_:
                beta_full[: self.K_] = beta_arr
                beta_full[self.k_half_ : self.k_half_ + self.K_] = beta_arr
            elif beta_arr.shape[0] == M:
                beta_full = beta_arr
            Phi = Phi / np.sqrt(beta_full)[np.newaxis, :]

        return Phi

    def decision_function(self, X):
        return self.model_.decision_function(self._transform(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self._transform(X))

    def predict(self, X):
        return self.model_.predict(self._transform(X))

    def score(self, X, y) -> float:
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X))


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score

    # Synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=5,
        n_redundant=2,
        flip_y=0.01,
        class_sep=1.0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Vanilla logistic
    from sklearn.linear_model import LogisticRegression as _LR

    vanilla = _LR(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    vanilla.fit(X_train, y_train)
    y_pred_v = vanilla.predict(X_test)
    y_score_v = vanilla.predict_proba(X_test)[:, 1]

    # Circulant weighted logistic
    K = 10
    beta = 1.0 + (np.linspace(0, 1, K) ** 2)
    circ_wt = SpectralCirculantWeightedLogisticRegression(
        padded_dim=None,
        K=None,
        beta=None,
        C=1.0,
        max_iter=1000,
        random_state=42,
        verbose=False,
    )
    circ_wt.fit(X_train, y_train)
    y_pred_c = circ_wt.predict(X_test)
    y_score_c = circ_wt.predict_proba(X_test)[:, 1]

    acc_v = accuracy_score(y_test, y_pred_v)
    auc_v = roc_auc_score(y_test, y_score_v)
    acc_c = accuracy_score(y_test, y_pred_c)
    auc_c = roc_auc_score(y_test, y_score_c)

    print("Vanilla LogisticRegression")
    print(f"  Accuracy: {acc_v:.3f},  ROC AUC: {auc_v:.3f}")
    print("SpectralCirculantWeightedLogisticRegression")
    print(f"  Accuracy: {acc_c:.3f},  ROC AUC: {auc_c:.3f}")
