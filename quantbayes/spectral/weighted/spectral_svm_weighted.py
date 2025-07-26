#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spectral_svm_weighted.py

SpectralWeightedSVM
-------------------
Spectral SVM with per-frequency ℓ₂ regularization weights β_j.
Trains in a low-dimensional spectral basis, then projects weights
back to the original feature space for prediction.
"""

import logging
import time
import warnings
from typing import Optional, Sequence, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def _setup_logger(name: str, verbose: bool) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.INFO if verbose else logging.WARNING)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
        )
        log.addHandler(handler)
    return log


def _validate_beta(beta: Sequence[float], n_spectral: int) -> np.ndarray:
    arr = np.asarray(beta, dtype=float)
    if arr.shape != (n_spectral,):
        raise ValueError(f"beta must have length {n_spectral}, got {arr.shape}")
    if np.any(arr <= 0):
        raise ValueError("All entries of beta must be > 0")
    return arr


class SpectralWeightedSVM(BaseEstimator, ClassifierMixin):
    """
    SpectralWeightedSVM wraps a LinearSVC trained on spectral features.

    Parameters
    ----------
    n_spectral : int or None, default=None
        Number of spectral basis vectors (must be ≤ n_features).
        If None, uses n_spectral = n_features (i.e. full feature basis).
    basis : {'random', 'pca'}, default='random'
        How to compute the basis from the data.
    beta : array-like of shape (n_spectral,), optional
        Per-frequency ℓ₂ regularization weights. If None, all weights = 1.
    C : float, default=1.0
        Inverse of regularization strength for LinearSVC.
    loss : {'hinge', 'squared_hinge'}, default='hinge'
    dual : bool, default=False
        Dual formulation flag for LinearSVC (overridden to True if loss='hinge').
    tol : float, default=1e-4
        Tolerance for stopping criterion.
    max_iter : int, default=1000
        Maximum iterations for LinearSVC.
    random_state : Optional[int], default=None
        Seed for reproducibility.
    verbose : bool, default=False
        If True, prints convergence info.
    probability : bool, default=False
        If True, enable probability estimates (adds calibration step).
    prob_cv : int, default=5
        Number of folds for probability calibration (ignored if probability=False).
    """

    def __init__(
        self,
        n_spectral: Optional[int] = None,
        *,
        basis: Literal["random", "pca"] = "random",
        beta: Optional[Sequence[float]] = None,
        C: float = 1.0,
        loss: Literal["hinge", "squared_hinge"] = "hinge",
        dual: bool = False,
        tol: float = 1e-4,
        max_iter: int = 1_000,
        random_state: Optional[int] = None,
        verbose: bool = False,
        probability: bool = False,
        prob_cv: int = 5,
    ):
        # hyperparameters only
        self.n_spectral = n_spectral
        self.basis = basis
        self.beta = beta
        self.C = C
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.probability = probability
        self.prob_cv = prob_cv

        # logger (not a hyperparameter)
        self._log = _setup_logger(self.__class__.__name__, verbose)

    def _generate_basis(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        D = X.shape[1]
        if self.basis == "random":
            G = rng.randn(D, self.n_spectral)
            Q, _ = np.linalg.qr(G)
            return Q[:, : self.n_spectral]
        else:  # 'pca'
            pca = PCA(n_components=self.n_spectral, random_state=self.random_state)
            pca.fit(X)
            return pca.components_.T

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralWeightedSVM":
        # 1) Validate inputs
        X, y = check_X_y(X, y, dtype=np.float64)
        N, D = X.shape

        # If n_spectral not set, use full feature count
        if self.n_spectral is None:
            self.n_spectral = D

        if self.n_spectral > D:
            raise ValueError(
                f"n_spectral (={self.n_spectral}) cannot exceed n_features (={D})"
            )
        self.n_features_in_ = D

        # 2) Build basis & spectral features
        self.V_ = self._generate_basis(X)  # shape (D, n_spectral)
        Phi = X.dot(self.V_)  # shape (N, n_spectral)

        # 3) Validate / scale by sqrt(beta)
        if self.beta is None:
            beta_sqrt = np.ones(self.n_spectral, dtype=float)
        else:
            beta_sqrt = np.sqrt(_validate_beta(self.beta, self.n_spectral))
        Phi_scaled = Phi / beta_sqrt[np.newaxis, :]

        # 4) Possibly override dual if hinge loss
        dual_flag = self.dual
        if self.loss == "hinge" and not dual_flag:
            self._log.warning(
                "LinearSVC requires dual=True when loss='hinge'; overriding dual=False→True"
            )
            dual_flag = True

        # 5) Fit raw LinearSVC
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            t0 = time.time()
            self.base_svc_ = LinearSVC(
                C=self.C,
                loss=self.loss,
                dual=dual_flag,
                tol=self.tol,
                max_iter=self.max_iter,
                random_state=self.random_state,
                verbose=int(self.verbose),
            )
            self.base_svc_.fit(Phi_scaled, y)
            self._log.info(f"LinearSVC converged in {time.time() - t0:.3f}s")

        # 6) Optional calibration for probabilities
        if self.probability:
            self.calibrator_ = CalibratedClassifierCV(
                estimator=self.base_svc_,
                cv=self.prob_cv,
                method="sigmoid",
            )
            self.calibrator_.fit(Phi_scaled, y)
            self.model_ = self.calibrator_
        else:
            self.model_ = self.base_svc_

        # 7) Project back to original space using raw SVM weights
        w_spec = self.base_svc_.coef_ / beta_sqrt[np.newaxis, :]
        self.coef_ = w_spec.dot(self.V_.T)  # shape (n_classes, D)
        self.intercept_ = self.base_svc_.intercept_.copy()
        self.classes_ = self.base_svc_.classes_

        return self

    def _spectral_transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_", "V_"])
        X = check_array(X, dtype=np.float64)
        Phi = X.dot(self.V_)
        if self.beta is not None:
            Phi = (
                Phi / np.sqrt(_validate_beta(self.beta, self.n_spectral))[np.newaxis, :]
            )
        return Phi

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.model_.decision_function(self._spectral_transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(self._spectral_transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.probability:
            raise AttributeError(
                "predict_proba is only available when probability=True"
            )
        return self.model_.predict_proba(self._spectral_transform(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # 1) Synthetic data
    X, y = make_classification(
        n_samples=1_000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        flip_y=0.03,
        class_sep=1.0,
        random_state=0,
    )

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    # 3a) SpectralWeightedSVM
    spec_clf = SpectralWeightedSVM(
        n_spectral=None,  # ≤ n_features=20
        basis="random",
        C=1.0,
        loss="hinge",
        dual=True,
        tol=1e-4,
        max_iter=1_000,
        random_state=0,
        verbose=False,
    )
    t0 = time.time()
    spec_clf.fit(X_train, y_train)
    t_spec = time.time() - t0

    # 3b) Vanilla LinearSVC
    vanilla_clf = LinearSVC(
        C=1.0,
        loss="hinge",
        dual=True,
        tol=1e-4,
        max_iter=1_000,
        random_state=0,
    )
    t0 = time.time()
    vanilla_clf.fit(X_train, y_train)
    t_vanilla = time.time() - t0

    # 4) Evaluate
    y_pred_spec = spec_clf.predict(X_test)
    y_pred_vanilla = vanilla_clf.predict(X_test)
    print("\n=== SpectralWeightedSVM ===")
    print(f"Time: {t_spec:.3f}s   Accuracy: {spec_clf.score(X_test, y_test):.4f}")
    print(classification_report(y_test, y_pred_spec))
    print("=== Vanilla LinearSVC ===")
    print(f"Time: {t_vanilla:.3f}s   Accuracy: {vanilla_clf.score(X_test, y_test):.4f}")
    print(classification_report(y_test, y_pred_vanilla))
