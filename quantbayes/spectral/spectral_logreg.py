#!/usr/bin/env python3
"""
SpectralLogisticRegression
--------------------------

A production‐level, strongly‐convex “spectral” logistic regression that

  - Fixes a random (or PCA) orthonormal basis V ∈ ℝ^{D×k},
  - Learns only the spectral weights s ∈ ℝ^k and bias b ∈ ℝ,
  - Wraps sklearn.linear_model.LogisticRegression for full solver support,
  - Implements the Estimator API (fit/predict/predict_proba/decision_function),
  - Validates inputs, logs progress, and suppresses convergence warnings,
  - Suitable for NeurIPS‐quality research code.
"""

import time
import logging
import warnings
from typing import Optional, Union

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression as SKLogReg
from sklearn.decomposition import PCA
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)
from sklearn.metrics import accuracy_score, roc_auc_score


class SpectralLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Spectral logistic regression with fixed orthonormal basis.

    Parameters
    ----------
    n_spectral : int or None
        Number of spectral features (k). If None, uses the full feature dimension D.
    basis : str
        How to generate the basis V:
        - "random": random Gaussian + QR orthonormalization,
        - "pca":   data‐dependent PCA basis.
    solver : str
        Solver for the internal sklearn LogisticRegression
        (e.g. "liblinear","lbfgs","sag","saga","newton-cg").
    dual : bool
        Dual formulation flag (only valid for some solvers).
    C : float
        Inverse regularization strength (1/λ) for sklearn.
    max_iter : int
        Maximum iterations for the internal solver.
    tol : float
        Tolerance for stopping criterion.
    random_state : Optional[int]
        Seed for reproducibility.
    verbose : bool
        If True, logs INFO messages.

    Attributes
    ----------
    n_spectral_ : int
        Actual spectral dimension used (≤ n_features_in_).
    V_ : np.ndarray, shape (D, n_spectral_)
        Fixed orthonormal basis.
    model_ : sklearn.linear_model.LogisticRegression
        Fitted internal model on spectral features.
    coef_ : np.ndarray, shape (1, n_spectral_)
        Learned spectral weights.
    intercept_ : float
        Learned bias.
    n_features_in_ : int
        Number of original features D.
    classes_ : np.ndarray
        Array of class labels.
    """

    def __init__(
        self,
        n_spectral: Optional[int] = None,
        basis: str = "random",
        solver: str = "lbfgs",
        dual: bool = False,
        C: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        # hyperparameters only
        self.n_spectral = n_spectral
        self.basis = basis
        self.solver = solver
        self.dual = dual
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        # setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
        )
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def _generate_basis(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        D = X.shape[1]
        # now use self.n_spectral_, which is guaranteed int
        if self.basis == "random":
            G = rng.randn(D, self.n_spectral_)
            Q, _ = np.linalg.qr(G)
            return Q
        elif self.basis == "pca":
            pca = PCA(n_components=self.n_spectral_, random_state=self.random_state)
            pca.fit(X)
            return pca.components_.T
        else:
            raise ValueError(f"Unknown basis '{self.basis}'")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralLogisticRegression":
        """
        Fit the spectral logistic regression.

        X : array-like, shape (N, D)
        y : array-like, shape (N,), binary labels {0,1}
        """
        X, y = check_X_y(X, y, dtype=np.float64, ensure_min_samples=2)
        N, D = X.shape
        self.n_features_in_ = D

        # decide actual spectral dimension
        if self.n_spectral is None:
            self.n_spectral_ = D
        else:
            if self.n_spectral > D:
                raise ValueError(
                    f"n_spectral (={self.n_spectral}) cannot exceed n_features (={D})"
                )
            self.n_spectral_ = self.n_spectral

        # generate basis
        t0 = time.time()
        self.V_ = self._generate_basis(X)
        self.logger.info(
            f"Basis '{self.basis}' generated in {time.time()-t0:.3f}s; "
            f"V_.shape = {self.V_.shape}"
        )

        # spectral features
        Phi = X.dot(self.V_)  # shape (N, n_spectral_)

        # delegate to sklearn, suppress convergence warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model_ = SKLogReg(
                penalty="l2",
                C=self.C,
                solver=self.solver,
                dual=self.dual,
                max_iter=self.max_iter,
                tol=self.tol,
                fit_intercept=True,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            t1 = time.time()
            self.model_.fit(Phi, y)
        self.logger.info(f"Solver converged in {time.time()-t1:.3f}s")

        # expose parameters
        self.coef_ = self.model_.coef_.copy()  # shape (1, n_spectral_)
        self.intercept_ = float(self.model_.intercept_[0])
        self.classes_ = self.model_.classes_
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_", "V_"])
        X = check_array(X, dtype=np.float64)
        Phi = X.dot(self.V_)
        return self.model_.decision_function(Phi)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_", "V_"])
        X = check_array(X, dtype=np.float64)
        Phi = X.dot(self.V_)
        return self.model_.predict_proba(Phi)

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_", "V_"])
        X = check_array(X, dtype=np.float64)
        Phi = X.dot(self.V_)
        return self.model_.predict(Phi)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        check_is_fitted(self, ["model_", "V_"])
        preds = self.predict(X)
        return accuracy_score(y, preds)


if __name__ == "__main__":
    import time
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.linear_model import LogisticRegression as VanillaLogReg

    # generate data
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

    # --- SpectralLogisticRegression ---
    print("\n=== SpectralLogisticRegression ===")
    spec_start = time.time()
    spec_model = SpectralLogisticRegression(
        n_spectral=None,
        basis="random",
        solver="lbfgs",
        C=1.0,
        max_iter=500,  # user can increase to avoid warnings
        tol=1e-6,
        random_state=0,
        verbose=False,
    )
    spec_model.fit(X_train, y_train)
    y_spec_pred = spec_model.predict(X_test)
    y_spec_proba = spec_model.predict_proba(X_test)[:, 1]
    spec_time = time.time() - spec_start
    print(f"Time elapsed:    {spec_time:.3f}s")
    print(f"Test accuracy:   {accuracy_score(y_test, y_spec_pred):.4f}")
    print(f"Test ROC AUC:    {roc_auc_score(y_test, y_spec_proba):.4f}")

    # --- Vanilla LogisticRegression ---
    print("\n=== Vanilla LogisticRegression (sklearn) ===")
    van_start = time.time()
    van_model = VanillaLogReg(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=500,  # match iteration budget
        tol=1e-6,
        random_state=0,
    )
    van_model.fit(X_train, y_train)
    y_van_pred = van_model.predict(X_test)
    y_van_proba = van_model.predict_proba(X_test)[:, 1]
    van_time = time.time() - van_start
    print(f"Time elapsed:    {van_time:.3f}s")
    print(f"Test accuracy:   {accuracy_score(y_test, y_van_pred):.4f}")
    print(f"Test ROC AUC:    {roc_auc_score(y_test, y_van_proba):.4f}")
