#!/usr/bin/env python3
"""
SpectralCirculantLogisticRegression
-----------------------------------

A strongly‑convex logistic regression whose weight vector w∈R^d is
constrained to be circulant (i.e. specified by its D‑point RFFT
coefficients).  We learn only the Fourier‑domain parameters, and
delegate the affine‐in‑those‐parameters logistic + ℓ₂‐penalty
to sklearn's solvers for production‑quality performance.
"""

import logging
import warnings
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression as SKLogReg
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score, roc_auc_score


class SpectralCirculantLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic regression with a circulant weight structure:
      w = irfft(F),   F ∈ C^{⌊D/2⌋+1}
    We parameterize F by its real and imaginary parts on the
    lower frequencies, and solve
      min_{F_real, F_imag, b}
        (1/n)∑_i ℓ( x_i · w + b, y_i )
        + (λ/2) ⋅ (‖F_real‖² + ‖F_imag‖²)
    which is _strongly convex_ in the Fourier parameters.
    """

    def __init__(
        self,
        padded_dim: Optional[int] = None,
        K: Optional[int] = None,
        *,
        solver: str = "lbfgs",
        dual: bool = False,
        C: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        # store parameters
        self.padded_dim = padded_dim
        self.K = K
        self.solver = solver
        self.dual = dual
        self.C = C
        self.max_iter = max_iter  # now fully user‑exposed
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        # set up logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
        )
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "SpectralCirculantLogisticRegression":
        X, y = check_X_y(X, y, dtype=np.float64)
        n_samples, D = X.shape

        # padded dimension: None ⇒ use D exactly (no padding)
        if self.padded_dim is None:
            self.padded_dim_ = D
        else:
            self.padded_dim_ = int(self.padded_dim)
            if self.padded_dim_ < D:
                raise ValueError("padded_dim must be ≥ original feature dim")

        # how many rfft bins
        self.k_half_ = self.padded_dim_ // 2 + 1
        self.K_ = (
            self.k_half_ if self.K is None or self.K > self.k_half_ else int(self.K)
        )
        freq_idx = np.arange(self.k_half_)
        self._mask = freq_idx < self.K_

        # FFT feature map
        X_pad = np.zeros((n_samples, self.padded_dim_), dtype=np.float64)
        X_pad[:, :D] = X
        Xf = np.fft.rfft(X_pad, axis=1)
        Xf[:, ~self._mask] = 0.0
        Xf_real, Xf_imag = Xf.real, Xf.imag
        Phi = np.hstack([Xf_real, Xf_imag])  # shape (n_samples, 2*k_half)

        # delegate to sklearn
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.model_ = SKLogReg(
                penalty="l2",
                C=self.C,
                solver=self.solver,
                dual=self.dual,
                max_iter=self.max_iter,  # uses user’s max_iter
                tol=self.tol,
                fit_intercept=True,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            self.model_.fit(Phi, y)

        # recover learned Fourier‐domain params
        coef = self.model_.coef_.ravel()
        self.F_real_ = coef[: self.k_half_].copy()
        self.F_imag_ = coef[self.k_half_ :].copy()
        self.intercept_ = float(self.model_.intercept_[0])
        self.classes_ = self.model_.classes_
        self.n_features_in_ = D
        return self

    def _make_Phi(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, dtype=np.float64)
        n_samples, D = X.shape
        if D != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_}, got {D}")
        X_pad = np.zeros((n_samples, self.padded_dim_), dtype=np.float64)
        X_pad[:, :D] = X
        Xf = np.fft.rfft(X_pad, axis=1)
        Xf[:, ~self._mask] = 0.0
        return np.hstack([Xf.real, Xf.imag])

    def decision_function(self, X):
        check_is_fitted(self, ["model_", "F_real_"])
        return self.model_.decision_function(self._make_Phi(X))

    def predict_proba(self, X):
        check_is_fitted(self, ["model_", "F_real_"])
        return self.model_.predict_proba(self._make_Phi(X))

    def predict(self, X):
        check_is_fitted(self, ["model_"])
        return self.model_.predict(self._make_Phi(X))

    @property
    def weight_(self) -> np.ndarray:
        check_is_fitted(self, ["F_real_"])
        Ff = self.F_real_ + 1j * self.F_imag_
        w_full = np.fft.irfft(Ff, n=self.padded_dim_)
        return w_full[: self.n_features_in_]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


if __name__ == "__main__":
    import time
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.linear_model import LogisticRegression as VanillaLogReg

    # generate synthetic data
    X, y = make_classification(
        n_samples=5000,
        n_features=100,
        n_informative=20,
        n_redundant=10,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Spectral circulant logistic ---
    print("\n=== SpectralCirculantLogisticRegression ===")
    spec = SpectralCirculantLogisticRegression(
        padded_dim=None,  # next‐power‑of‑two → 128
        K=None,  # retain 50 frequencies
        solver="lbfgs",
        C=1.0,
        max_iter=500,  # bumped up to avoid warnings
        tol=1e-6,
        random_state=0,
        verbose=False,
    )
    t0 = time.time()
    spec.fit(X_train, y_train)
    t_spec = time.time() - t0
    y_spec = spec.predict(X_test)
    p_spec = spec.predict_proba(X_test)[:, 1]
    print(f"Time:    {t_spec:.3f}s")
    print(f"Accuracy:{accuracy_score(y_test, y_spec):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, p_spec):.4f}")

    # --- Vanilla logistic regression ---
    print("\n=== Vanilla LogisticRegression ===")
    van = VanillaLogReg(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=500,  # likewise increased
        tol=1e-6,
        random_state=0,
    )
    t1 = time.time()
    van.fit(X_train, y_train)
    t_van = time.time() - t1
    y_van = van.predict(X_test)
    p_van = van.predict_proba(X_test)[:, 1]
    print(f"Time:    {t_van:.3f}s")
    print(f"Accuracy:{accuracy_score(y_test, y_van):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, p_van):.4f}")
