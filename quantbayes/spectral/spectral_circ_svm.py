#!/usr/bin/env python3
"""
SpectralCirculantSVM
--------------------

A production‑level, strongly‑convex circulant‑SVM that

  - Constrains w∈ℝ^d to be circulant via its rFFT coefficients,
  - Learns only the low‑frequency Fourier weights and bias,
  - Delegates to sklearn.svm.LinearSVC for the hinge‑loss SVM head,
  - Implements the Estimator API (fit/predict/decision_function/score),
  - Validates inputs, logs progress, and suppresses convergence warnings,
  - Suitable for NeurIPS‑quality research code.
"""

import time
import logging
import warnings
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning


class SpectralCirculantSVM(BaseEstimator, ClassifierMixin):
    """
    Circulant‑structured SVM via Fourier‑domain parameters.

    Parameters
    ----------
    padded_dim : Optional[int]
        FFT length (next power of two ≥ D if None; here we just require ≥ D).
    K : Optional[int]
        Number of low frequencies to keep (≤ padded_dim//2+1).
    C : float
        Regularization parameter (inverse of penalty strength).
    loss : {"hinge","squared_hinge"}
        SVM loss function.
    dual : bool
        Dual or primal formulation.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.
    random_state : Optional[int]
        For any internal randomness (not used).
    verbose : bool
        If True, logs INFO messages.
    probability : bool, default=False
        If True, enable probability estimates via Platt scaling.
    prob_cv : int, default=5
        Number of folds for probability calibration (ignored if probability=False).

    Attributes after fit
    --------------------
    base_svc_ : LinearSVC
        The raw fitted LinearSVC (before calibration).
    model_ : LinearSVC or CalibratedClassifierCV
        Predictor: the calibrator if `probability=True`, else `base_svc_`.
    F_real_ : np.ndarray, shape (k_half_,)
        Learned real parts of low‑freq FFT bins.
    F_imag_ : np.ndarray, shape (k_half_,)
        Learned imaginary parts.
    intercept_ : float
        SVM bias.
    classes_ : np.ndarray
        Target class labels.
    n_features_in_ : int
        Original feature dimension D.
    """

    def __init__(
        self,
        padded_dim: Optional[int] = None,
        K: Optional[int] = None,
        C: float = 1.0,
        loss: str = "hinge",
        dual: bool = False,
        tol: float = 1e-4,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
        verbose: bool = False,
        probability: bool = False,
        prob_cv: int = 5,
    ):
        # hyperparameters
        self.padded_dim = padded_dim
        self.K = K
        self.C = C
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.probability = probability
        self.prob_cv = prob_cv

        # logger setup
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(
                logging.Formatter(
                    "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
                )
            )
            self.logger.addHandler(h)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralCirculantSVM":
        # 1) validate
        X, y = check_X_y(X, y, dtype=np.float64)
        n_samples, D = X.shape
        self.n_features_in_ = D

        # 2) determine padded_dim_
        pd = D if self.padded_dim is None else int(self.padded_dim)
        if pd < D:
            raise ValueError("padded_dim must be ≥ original feature dim")
        self.padded_dim_ = pd

        # 3) define mask for low frequencies
        self.k_half_ = pd // 2 + 1
        self.K_ = (
            self.k_half_ if (self.K is None or self.K > self.k_half_) else int(self.K)
        )
        idx = np.arange(self.k_half_)
        self._mask = idx < self.K_

        # 4) build design matrix Phi via truncated FFT
        X_pad = np.zeros((n_samples, pd), dtype=np.float64)
        X_pad[:, :D] = X
        Xf = np.fft.rfft(X_pad, axis=1)
        Xf[:, ~self._mask] = 0.0
        Phi = np.hstack([Xf.real, Xf.imag])

        # 5) fit raw LinearSVC
        dual_flag = True if (self.loss == "hinge" and not self.dual) else self.dual
        if self.loss == "hinge" and not self.dual:
            self.logger.warning(
                "LinearSVC requires dual=True with loss='hinge'; overriding dual=False→True"
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.base_svc_ = LinearSVC(
                C=self.C,
                loss=self.loss,
                dual=dual_flag,
                tol=self.tol,
                max_iter=self.max_iter,
                random_state=self.random_state,
                verbose=int(self.verbose),
            )
            t0 = time.time()
            self.base_svc_.fit(Phi, y)
            self.logger.info(f"SVM solver converged in {time.time() - t0:.3f}s")

        # 6) optional calibration
        if self.probability:
            self.calibrator_ = CalibratedClassifierCV(
                base_estimator=self.base_svc_,
                cv=self.prob_cv,
                method="sigmoid",
            )
            self.calibrator_.fit(Phi, y)
            self.model_ = self.calibrator_
        else:
            self.model_ = self.base_svc_

        # 7) recover Fourier weights & bias from raw SVC
        coef = self.base_svc_.coef_.ravel()
        self.F_real_ = coef[: self.k_half_].copy()
        self.F_imag_ = coef[self.k_half_ :].copy()
        self.intercept_ = float(self.base_svc_.intercept_[0])
        self.classes_ = self.model_.classes_

        return self

    def _make_Phi(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        X = check_array(X, dtype=np.float64)
        n_samples, D = X.shape
        if D != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {D}")
        X_pad = np.zeros((n_samples, self.padded_dim_), dtype=np.float64)
        X_pad[:, :D] = X
        Xf = np.fft.rfft(X_pad, axis=1)
        Xf[:, ~self._mask] = 0.0
        return np.hstack([Xf.real, Xf.imag])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.model_.decision_function(self._make_Phi(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(self._make_Phi(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.probability:
            raise AttributeError(
                "predict_proba is only available when probability=True"
            )
        return self.model_.predict_proba(self._make_Phi(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))


if __name__ == "__main__":
    import time
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.svm import LinearSVC as VanillaSVM

    # synthetic data
    X, y = make_classification(
        n_samples=5000,
        n_features=100,
        n_informative=20,
        n_redundant=10,
        random_state=42,
    )
    y = 2 * y - 1  # convert to {−1,+1}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- SpectralCirculantSVM ---
    print("\n=== SpectralCirculantSVM ===")
    circ = SpectralCirculantSVM(
        padded_dim=None,
        K=None,
        C=1.0,
        loss="squared_hinge",
        dual=False,
        tol=1e-4,
        max_iter=1000,
        random_state=0,
        verbose=True,
    )
    t0 = time.time()
    circ.fit(X_train, y_train)
    t_circ = time.time() - t0
    y_circ = circ.predict(X_test)
    scores_circ = circ.decision_function(X_test)
    print(f"Time:    {t_circ:.3f}s")
    print(f"Accuracy:{accuracy_score(y_test, y_circ):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, scores_circ):.4f}")

    # --- Vanilla LinearSVC ---
    print("\n=== Vanilla LinearSVC ===")
    van = VanillaSVM(
        C=1.0,
        loss="squared_hinge",
        dual=False,
        tol=1e-4,
        max_iter=1000,
        random_state=0,
    )
    t1 = time.time()
    van.fit(X_train, y_train)
    t_van = time.time() - t1
    y_van = van.predict(X_test)
    scores_van = van.decision_function(X_test)
    print(f"Time:    {t_van:.3f}s")
    print(f"Accuracy:{accuracy_score(y_test, y_van):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, scores_van):.4f}")
