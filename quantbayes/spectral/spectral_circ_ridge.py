#!/usr/bin/env python3
"""
SpectralCirculantRidge
----------------------
Strongly‑convex circulant ridge regression via low‑frequency Fourier parameters.
"""

import logging
import time
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class SpectralCirculantRidge(BaseEstimator, RegressorMixin):
    """
    Parameters
    ----------
    padded_dim : Optional[int]
        FFT length (if None, uses original feature dim).
    K : Optional[int]
        Number of low frequencies to keep (≤ padded_dim//2+1).
    alpha : float
        Regularization strength λ in ½‖θ‖².
    fit_intercept : bool
        Whether to fit an intercept.
    solver : str
        Solver for Ridge ('auto', 'svd', 'cholesky', etc.).
    max_iter : Optional[int]
        Maximum iterations for iterative solvers.
    tol : float
        Solver tolerance.
    random_state : Optional[int]
        Ignored by FFT but passed to Ridge if needed.
    verbose : bool
        If True, log timing info.
    """

    def __init__(
        self,
        padded_dim: Optional[int] = None,
        K: Optional[int] = None,
        *,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: str = "auto",
        max_iter: Optional[int] = None,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.padded_dim = padded_dim
        self.K = K
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
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

    def _make_fft_features(self, X: np.ndarray):
        n, D = X.shape

        # determine padded length
        if self.padded_dim is None:
            self.padded_dim_ = D
        else:
            self.padded_dim_ = int(self.padded_dim)
            if self.padded_dim_ < D:
                raise ValueError("padded_dim must be ≥ original feature dim")

        # number of unique FFT bins
        self.k_half_ = self.padded_dim_ // 2 + 1
        # how many to keep
        self.K_ = (
            self.k_half_ if self.K is None or self.K > self.k_half_ else int(self.K)
        )
        mask = np.arange(self.k_half_) < self.K_

        # pad and FFT
        X_pad = np.zeros((n, self.padded_dim_), dtype=np.float64)
        X_pad[:, :D] = X
        Xf = np.fft.rfft(X_pad, axis=1)
        Xf[:, ~mask] = 0.0

        # real+imag features
        return np.hstack([Xf.real, Xf.imag])  # shape (n, 2*K_)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralCirculantRidge":
        X, y = check_X_y(X, y, multi_output=True, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        Phi = self._make_fft_features(X)

        self.model_ = Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            solver=self.solver,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        t0 = time.time()
        self.model_.fit(Phi, y)
        self.logger.info(f"Ridge converged in {time.time()-t0:.3f}s")

        # recover Fourier‐domain parameters
        coef_spec = self.model_.coef_  # shape (n_targets, 2*K_) or (2*K_,)
        if coef_spec.ndim == 1:
            spec = coef_spec
        else:
            spec = coef_spec[0]
        self.F_real_ = spec[: self.k_half_].copy()
        self.F_imag_ = spec[self.k_half_ :].copy()

        # back‑project to original space
        Ff = self.F_real_ + 1j * self.F_imag_
        w_full = np.fft.irfft(Ff, n=self.padded_dim_)
        self.coef_ = w_full[: self.n_features_in_]  # shape (D,)
        self.intercept_ = self.model_.intercept_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        X = check_array(X, dtype=np.float64)
        Phi = self._make_fft_features(X)
        return self.model_.predict(Phi)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(X))


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_squared_error

    # 1) Synthetic data
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

    # 3) Circulant Ridge
    circ = SpectralCirculantRidge(
        padded_dim=None,
        K=None,
        alpha=1.0,
        fit_intercept=True,
        solver="auto",
        random_state=42,
        verbose=False,
    )
    circ.fit(X_train, y_train)
    y_pred_c = circ.predict(X_test)

    # 4) Metrics
    r2_v = r2_score(y_test, y_pred_v)
    rmse_v = np.sqrt(mean_squared_error(y_test, y_pred_v))
    r2_c = r2_score(y_test, y_pred_c)
    rmse_c = np.sqrt(mean_squared_error(y_test, y_pred_c))

    print("Vanilla Ridge")
    print(f"  R²:   {r2_v:.3f}")
    print(f"  RMSE: {rmse_v:.3f}\n")

    print("SpectralCirculantRidge")
    print(f"  R²:   {r2_c:.3f}")
    print(f"  RMSE: {rmse_c:.3f}")
