#!/usr/bin/env python3
"""
SpectralWeightedRidge
---------------------
Strongly-convex ridge regression on fixed spectral features
with per-frequency regularization weights β_j.
"""

import logging
import time
from typing import Optional, Sequence, Literal

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class SpectralWeightedRidge(BaseEstimator, RegressorMixin):
    """
    Parameters
    ----------
    n_spectral : int
        Number of spectral features k.
    basis : {'random','pca'}
        How to generate the fixed orthonormal basis.
    beta : array-like of shape (k,), optional
        Per-frequency regularization weights β_j. Must be > 0.
        If None, defaults to ones.
    alpha : float
        Regularization strength λ for Ridge.
    fit_intercept : bool
        Whether to fit an intercept.
    solver : str
        Solver for the Ridge ('auto', 'svd', etc.).
    max_iter : Optional[int]
        Maximum iterations (for solvers that respect max_iter).
    tol : float
        Solver tolerance.
    random_state : Optional[int]
        Seed for basis generation and (if applicable) solver.
    verbose : bool
        If True, log timing and basis info.
    """

    def __init__(
        self,
        n_spectral: int = 128,
        basis: Literal["random", "pca"] = "random",
        beta: Optional[Sequence[float]] = None,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: str = "auto",
        max_iter: Optional[int] = None,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_spectral = n_spectral
        self.basis = basis
        self.beta = None if beta is None else np.asarray(beta, dtype=float)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        # set up logger
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
        """Ensure beta has length k and is strictly positive."""
        if self.beta is None:
            return np.ones(k, dtype=float)
        if self.beta.shape != (k,):
            raise ValueError(f"beta must have shape ({k},), got {self.beta.shape}")
        if np.any(self.beta <= 0):
            raise ValueError("All entries of beta must be > 0")
        return self.beta.copy()

    def _generate_basis(self, X: np.ndarray) -> np.ndarray:
        """Build a D×k orthonormal basis, random or PCA."""
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralWeightedRidge":
        X, y = check_X_y(X, y, multi_output=True, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # 1) Build basis & spectral features
        self.V_ = self._generate_basis(X)
        self.logger.info(f"Generated basis of shape {self.V_.shape}")
        Phi = X.dot(self.V_)  # (n_samples, k)

        # 2) Scale features by 1/√β so that a standard Ridge α‖θ‖²
        #    becomes ∑_j α β_j θ_j² in the original coords
        k = Phi.shape[1]
        beta = self._validate_beta(k)
        sqrt_beta = np.sqrt(beta)
        Phi_scaled = Phi / sqrt_beta

        # 3) Fit Ridge on scaled features
        self.model_ = Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            solver=self.solver,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        t0 = time.time()
        self.model_.fit(Phi_scaled, y)
        self.logger.info(f"Ridge converged in {time.time() - t0:.3f}s")

        # 4) Recover original θ = θ_scaled / √β, then back-project to x-space
        coef_scaled = self.model_.coef_  # (n_targets, k) or (k,)
        theta_orig = coef_scaled / sqrt_beta
        if theta_orig.ndim == 1:
            self.coef_ = theta_orig.dot(self.V_.T)  # (D,)
        else:
            self.coef_ = theta_orig.dot(self.V_.T)  # (n_targets, D)
        self.intercept_ = self.model_.intercept_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_", "V_"])
        X = check_array(X, dtype=np.float64)
        Phi = X.dot(self.V_)
        beta = self._validate_beta(Phi.shape[1])
        Phi_scaled = Phi / np.sqrt(beta)
        return self.model_.predict(Phi_scaled)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(X))


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, root_mean_squared_error

    # 1) Generate synthetic regression data
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=5,
        noise=5.0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 2) Vanilla Ridge baseline
    vanilla = Ridge(alpha=1.0, fit_intercept=True, random_state=42)
    vanilla.fit(X_train, y_train)
    preds_v = vanilla.predict(X_test)
    r2_v = r2_score(y_test, preds_v)
    rmse_v = root_mean_squared_error(y_test, preds_v)

    # 3) Spectral weighted Ridge
    # k = 10
    # beta = 1.0 + (np.linspace(0, 1, k) ** 2)
    Ks = list(range(1, 21))
    for k in Ks:
        print("*" * 50, "\n")
        print(f"Testing with K: {k}\n")
        model = SpectralWeightedRidge(
            n_spectral=k,
            basis="pca",
            beta=None,
            alpha=1.0,
            fit_intercept=True,
            random_state=42,
            verbose=False,
        )
        model.fit(X_train, y_train)
        preds_s = model.predict(X_test)
        r2_s = r2_score(y_test, preds_s)
        rmse_s = root_mean_squared_error(y_test, preds_s)

        print(
            "Vanilla Ridge              - R²: {:.3f}, RMSE: {:.3f}".format(r2_v, rmse_v)
        )
        print(
            "SpectralWeightedRidge      - R²: {:.3f}, RMSE: {:.3f}".format(r2_s, rmse_s)
        )
