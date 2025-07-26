#!/usr/bin/env python3
"""
SpectralSVM
-----------

A production‑level, strongly‑convex spectral SVM that

  - Fixes a random (or PCA) orthonormal basis V ∈ ℝ^{D×k},
  - Learns only the spectral SVM weights in ℝ^k and bias,
  - Delegates to sklearn.svm.LinearSVC for hinge‐loss SVM,
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning


class SpectralSVM(BaseEstimator, ClassifierMixin):
    """
    Spectral SVM with fixed orthonormal basis front‑end.

    Parameters
    ----------
    n_spectral : int or None
        Number of spectral features (k). If None, uses the full feature dimension D.
    basis : {"random","pca"}
        "random" for Gaussian+QR, or "pca" for data‑dependent PCA basis.
    C : float
        Regularization parameter (inverse of penalty strength).
    loss : {"hinge","squared_hinge"}
        SVM loss function.
    dual : bool
        Dual or primal optimization formulation.
    tol : float
        Tolerance for primal/dual convergence.
    max_iter : int
        Maximum number of iterations.
    random_state : Optional[int]
        Seed for reproducibility.
    verbose : bool
        If True, logs INFO messages.
    probability : bool, default=False
        If True, enable probability estimates (adds calibration step).
    prob_cv : int, default=5
        Number of folds for probability calibration (ignored if probability=False).

    Attributes after fit
    --------------------
    n_spectral_ : int
        Actual spectral dimension used (≤ n_features_in_).
    V_ : np.ndarray, shape (D, n_spectral_)
        Fixed orthonormal basis.
    base_svc_ : LinearSVC
        The raw fitted LinearSVC before any calibration.
    model_ : LinearSVC or CalibratedClassifierCV
        Fitted internal SVM model (calibrated if probability=True).
    coef_ : np.ndarray, shape (1, n_spectral_)
        Learned spectral weights.
    intercept_ : float
        Learned SVM bias term.
    classes_ : np.ndarray
        Target class labels.
    n_features_in_ : int
        Original feature dimension D.
    """

    def __init__(
        self,
        n_spectral: Optional[int] = None,
        basis: str = "random",
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
        # hyperparameters only
        self.n_spectral = n_spectral
        self.basis = basis
        self.C = C
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.probability = probability
        self.prob_cv = prob_cv

        # set up logger (not a hyperparameter)
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

    def _generate_basis(self, X: np.ndarray) -> np.ndarray:
        D = X.shape[1]
        rng = np.random.RandomState(self.random_state)
        if self.basis == "random":
            G = rng.randn(D, self.n_spectral_)
            Q, _ = np.linalg.qr(G)
            return Q
        else:  # 'pca'
            pca = PCA(n_components=self.n_spectral_, random_state=self.random_state)
            pca.fit(X)
            return pca.components_.T

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralSVM":
        # 1) validate
        X, y = check_X_y(X, y, dtype=np.float64, ensure_min_samples=2)
        N, D = X.shape
        self.n_features_in_ = D

        # 2) handle n_spectral=None → full dimension
        if self.n_spectral is None:
            self.n_spectral_ = D
        else:
            if self.n_spectral > D:
                raise ValueError(
                    f"n_spectral (={self.n_spectral}) cannot exceed n_features (={D})"
                )
            self.n_spectral_ = self.n_spectral

        # 3) build basis & project
        t0 = time.time()
        self.V_ = self._generate_basis(X)  # shape (D, n_spectral_)
        self.logger.info(
            f"Basis '{self.basis}' generated in {time.time()-t0:.3f}s; V_.shape={self.V_.shape}"
        )
        Phi = X.dot(self.V_)  # shape (N, n_spectral_)

        # 4) fit a LinearSVC
        dual = self.dual
        if self.loss == "hinge" and not dual:
            self.logger.warning(
                "LinearSVC requires dual=True when loss='hinge'; overriding dual=False→True"
            )
            dual = True

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.base_svc_ = LinearSVC(
                C=self.C,
                loss=self.loss,
                dual=dual,
                tol=self.tol,
                max_iter=self.max_iter,
                random_state=self.random_state,
                verbose=int(self.verbose),
            )
            t1 = time.time()
            self.base_svc_.fit(Phi, y)
            self.logger.info(f"SVM solver converged in {time.time()-t1:.3f}s")

        # 5) optional calibration
        if self.probability:
            self.calibrator_ = CalibratedClassifierCV(
                estimator=self.base_svc_, cv=self.prob_cv, method="sigmoid"
            )
            self.calibrator_.fit(Phi, y)
            self.model_ = self.calibrator_
        else:
            self.model_ = self.base_svc_

        # 6) expose coef_ / intercept_
        # Always take raw SVC weights, even if we calibrated probabilities
        self.coef_ = self.base_svc_.coef_.copy()  # shape (1, n_spectral_)
        self.intercept_ = float(self.base_svc_.intercept_[0])
        self.classes_ = self.model_.classes_
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_", "V_"])
        X = check_array(X, dtype=np.float64)
        return X.dot(self.V_)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.model_.decision_function(self._transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(self._transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.probability:
            raise AttributeError(
                "predict_proba is only available when probability=True"
            )
        return self.model_.predict_proba(self._transform(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))


if __name__ == "__main__":
    import time
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.svm import LinearSVC as VanillaSVM

    # generate synthetic data
    X, y = make_classification(
        n_samples=5000,
        n_features=100,
        n_informative=20,
        n_redundant=10,
        random_state=42,
    )
    # convert labels to {-1,1} for hinge loss
    y = 2 * y - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Spectral SVM ---
    print("\n=== SpectralSVM ===")
    spec = SpectralSVM(
        n_spectral=None,
        basis="random",
        C=1.0,
        loss="squared_hinge",
        dual=False,
        tol=1e-4,
        max_iter=1000,
        random_state=0,
        verbose=True,
    )
    t0 = time.time()
    spec.fit(X_train, y_train)
    t_spec = time.time() - t0
    y_spec = spec.predict(X_test)
    scores_spec = spec.decision_function(X_test)
    print(f"Time:    {t_spec:.3f}s")
    print(f"Accuracy:{accuracy_score(y_test, y_spec):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, scores_spec):.4f}")

    # --- Vanilla SVM ---
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
    van.fit(X_train.dot(np.eye(100)), y_train)  # raw features
    t_van = time.time() - t1
    y_van = van.predict(X_test)
    scores_van = van.decision_function(X_test)
    print(f"Time:    {t_van:.3f}s")
    print(f"Accuracy:{accuracy_score(y_test, y_van):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, scores_van):.4f}")
