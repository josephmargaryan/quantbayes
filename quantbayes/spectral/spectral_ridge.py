#!/usr/bin/env python3
"""
SpectralRidge
-------------
Strongly‑convex ridge regression on fixed spectral features.
"""

import logging
import time
from typing import Optional, Literal

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class SpectralRidge(BaseEstimator, RegressorMixin):
    """
    Parameters
    ----------
    n_spectral : int
        Number of spectral features k.
    basis : {'random','pca'}
        How to generate the fixed orthonormal basis.
    alpha : float
        Regularization strength λ in ½‖θ‖².
    fit_intercept : bool
        Whether to fit an intercept.
    max_iter : Optional[int]
        Maximum iterations for the solver.
    tol : float
        Solver tolerance.
    random_state : Optional[int]
        Random seed.
    verbose : bool
        If True, log timing info.
    """

    def __init__(
        self,
        n_spectral: int = 128,
        basis: Literal["random", "pca"] = "random",
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: Optional[int] = None,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_spectral = n_spectral
        self.basis = basis
        self.alpha = alpha
        self.fit_intercept = fit_intercept
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralRidge":
        X, y = check_X_y(X, y, multi_output=True, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        self.V_ = self._generate_basis(X)
        Phi = X.dot(self.V_)  # shape (n_samples, k)

        self.model_ = Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            solver="auto",
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        t0 = time.time()
        self.model_.fit(Phi, y)
        self.logger.info(f"Ridge converged in {time.time()-t0:.3f}s")

        # back‑project coefficients to original feature space
        coef_spec = self.model_.coef_  # shape (n_targets, k) or (k,)
        # handle single‑target
        if coef_spec.ndim == 1:
            self.coef_ = coef_spec @ self.V_.T  # shape (D,)
        else:
            self.coef_ = coef_spec.dot(self.V_.T)  # shape (n_targets, D)
        self.intercept_ = self.model_.intercept_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_", "V_"])
        X = check_array(X, dtype=np.float64)
        return self.model_.predict(X.dot(self.V_))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(X))


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_squared_error

    # 1) Synthetic regression data
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

    # 2) Vanilla Ridge
    vanilla = Ridge(alpha=1.0, fit_intercept=True, random_state=42)
    vanilla.fit(X_train, y_train)
    y_pred_v = vanilla.predict(X_test)

    # 3) Spectral Ridge
    spectral = SpectralRidge(
        n_spectral=None,
        basis="pca",
        alpha=1.0,
        fit_intercept=True,
        random_state=42,
        verbose=False,
    )
    spectral.fit(X_train, y_train)
    y_pred_s = spectral.predict(X_test)

    # 4) Metrics
    r2_v = r2_score(y_test, y_pred_v)
    rmse_v = np.sqrt(mean_squared_error(y_test, y_pred_v))
    r2_s = r2_score(y_test, y_pred_s)
    rmse_s = np.sqrt(mean_squared_error(y_test, y_pred_s))

    print("Vanilla Ridge")
    print(f"  R²:   {r2_v:.3f}")
    print(f"  RMSE: {rmse_v:.3f}\n")

    print("SpectralRidge")
    print(f"  R²:   {r2_s:.3f}")
    print(f"  RMSE: {rmse_s:.3f}")
