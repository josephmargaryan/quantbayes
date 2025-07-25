#!/usr/bin/env python3
"""
SpectralWeightedLogisticRegression
----------------------------------
Spectral logistic regression with per‑frequency ℓ₂ regularization weights β_j.
"""

import logging
import warnings
from typing import Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression as SKLogReg
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.exceptions import ConvergenceWarning


class SpectralWeightedLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Parameters
    ----------
    n_spectral : int
        Number of spectral features k.
    basis : {'random','pca'}
        How to generate the fixed orthonormal basis.
    beta : array-like of shape (k,), optional
        Per-frequency regularization weights β_j > 0. If None, defaults to all ones.
    solver : str
        Solver for the internal logistic regression.
    C : float
        Inverse regularization strength (1/λ).
    max_iter : int
        Maximum iterations for the solver.
    tol : float
        Solver tolerance.
    random_state : Optional[int]
        Random seed.
    verbose : bool
        If True, enable INFO logging.
    """

    def __init__(
        self,
        n_spectral: int = 50,
        basis: str = "random",
        beta: Optional[Sequence[float]] = None,
        solver: str = "lbfgs",
        C: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_spectral = n_spectral
        self.basis = basis
        self.beta = None if beta is None else np.asarray(beta, dtype=float)
        self.solver = solver
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

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

    def _validate_beta(self, k: int) -> np.ndarray:
        if self.beta is None:
            return np.ones(k, dtype=float)
        if self.beta.shape != (k,):
            raise ValueError(f"beta must have shape ({k},), got {self.beta.shape}")
        if np.any(self.beta <= 0):
            raise ValueError("All entries of beta must be > 0")
        return self.beta.copy()

    def _generate_basis(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        D = X.shape[1]
        if self.basis == "random":
            G = rng.randn(D, self.n_spectral)
            Q, _ = np.linalg.qr(G)
            return Q
        elif self.basis == "pca":
            pca = PCA(n_components=self.n_spectral, random_state=self.random_state)
            pca.fit(X)
            return pca.components_.T
        else:
            raise ValueError(f"Unknown basis '{self.basis}'")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralWeightedLogisticRegression":
        X, y = check_X_y(X, y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        self.V_ = self._generate_basis(X)
        self.logger.info(f"Basis generated with shape {self.V_.shape}")

        Phi = X.dot(self.V_)
        k = Phi.shape[1]

        beta = self._validate_beta(k)
        sqrt_beta = np.sqrt(beta)
        Phi_scaled = Phi / sqrt_beta

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model_ = SKLogReg(
                penalty="l2",
                C=self.C,
                solver=self.solver,
                max_iter=self.max_iter,
                tol=self.tol,
                fit_intercept=True,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            self.model_.fit(Phi_scaled, y)

        coef_scaled = self.model_.coef_.ravel()
        self.coef_ = coef_scaled / sqrt_beta
        self.intercept_ = float(self.model_.intercept_[0])
        self.classes_ = self.model_.classes_
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_", "V_"])
        X = check_array(X, dtype=np.float64)
        Phi = X.dot(self.V_)
        beta = self._validate_beta(Phi.shape[1])
        Phi_scaled = Phi / np.sqrt(beta)
        return self.model_.decision_function(Phi_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_", "V_"])
        X = check_array(X, dtype=np.float64)
        Phi = X.dot(self.V_)
        beta = self._validate_beta(Phi.shape[1])
        return self.model_.predict_proba(Phi / np.sqrt(beta))

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_", "V_"])
        X = check_array(X, dtype=np.float64)
        Phi = X.dot(self.V_)
        return self.model_.predict(Phi / np.sqrt(self._validate_beta(Phi.shape[1])))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X))


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
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
    vanilla = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    vanilla.fit(X_train, y_train)
    y_pred_v = vanilla.predict(X_test)
    y_score_v = vanilla.predict_proba(X_test)[:, 1]

    # Spectral weighted logistic
    k = 10
    beta = 1.0 + (np.linspace(0, 1, k) ** 2)
    spectral_wt = SpectralWeightedLogisticRegression(
        n_spectral=None,
        basis="pca",
        beta=None,
        C=1.0,
        max_iter=1000,
        random_state=42,
        verbose=False,
    )
    spectral_wt.fit(X_train, y_train)
    y_pred_s = spectral_wt.predict(X_test)
    y_score_s = spectral_wt.predict_proba(X_test)[:, 1]

    acc_v = accuracy_score(y_test, y_pred_v)
    auc_v = roc_auc_score(y_test, y_score_v)
    acc_s = accuracy_score(y_test, y_pred_s)
    auc_s = roc_auc_score(y_test, y_score_s)

    print("Vanilla LogisticRegression")
    print(f"  Accuracy: {acc_v:.3f},  ROC AUC: {auc_v:.3f}")
    print("SpectralWeightedLogisticRegression")
    print(f"  Accuracy: {acc_s:.3f},  ROC AUC: {auc_s:.3f}")
