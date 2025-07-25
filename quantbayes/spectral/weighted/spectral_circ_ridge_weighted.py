#!/usr/bin/env python3
"""
SpectralCirculantWeightedRidge
-----------------------------
Strongly-convex circulant ridge regression via low-frequency Fourier parameters
with per-frequency regularization weights β_j.
"""

import logging
import time
from typing import Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


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


class SpectralCirculantWeightedRidge(BaseEstimator, RegressorMixin):
    """
    Circulant-parameterised ridge regression in the FFT domain, with per-frequency ℓ₂ weights β_j.

    Parameters
    ----------
    padded_dim : Optional[int]
        FFT length ≥ D if given, else uses D.
    K : Optional[int]
        Number of low frequencies to keep.
    beta : array-like, shape (K,) or (2*k_half_,), optional
        If length=K, weight per frequency (applied to both real & imag).
        If length=2*k_half_, weight per feature column.
    alpha : float
        Regularization strength λ for Ridge.
    fit_intercept : bool
        Whether to fit an intercept.
    solver : str
        Solver for Ridge ('auto', 'svd', etc.).
    max_iter : Optional[int]
        Maximum iterations.
    tol : float
        Solver tolerance.
    random_state : Optional[int]
        Passed to Ridge if solver uses RNG.
    verbose : bool
        If True, log timing info.
    """

    def __init__(
        self,
        padded_dim: Optional[int] = None,
        K: Optional[int] = None,
        *,
        beta: Optional[Sequence[float]] = None,
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
        self.beta = None if beta is None else np.asarray(beta, dtype=float)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.logger = _setup_logger(self.__class__.__name__, verbose)

    def _make_fft_features(self, X: np.ndarray):
        n, D = X.shape
        # determine padded dim
        self.padded_dim_ = D if self.padded_dim is None else int(self.padded_dim)
        if self.padded_dim_ < D:
            raise ValueError("padded_dim must be ≥ original feature dim")

        # frequency bins
        self.k_half_ = self.padded_dim_ // 2 + 1
        self.K_ = (
            self.k_half_ if (self.K is None or self.K > self.k_half_) else int(self.K)
        )
        mask = np.arange(self.k_half_) < self.K_

        # pad & FFT
        X_pad = np.zeros((n, self.padded_dim_), dtype=float)
        X_pad[:, :D] = X
        Xf = np.fft.rfft(X_pad, axis=1)
        Xf[:, ~mask] = 0.0

        # stack real + imag → shape (n, 2*k_half_)
        Phi = np.hstack([Xf.real, Xf.imag])
        M = Phi.shape[1]  # = 2*k_half_

        # build full-length β of size M
        if self.beta is None:
            sqrt_beta = np.ones(M, dtype=float)
        else:
            beta_full = np.ones(M, dtype=float)
            b = self.beta
            if b.shape[0] == self.K_:
                # fill low-frequency real and imag
                beta_full[: self.K_] = b
                beta_full[self.k_half_ : self.k_half_ + self.K_] = b
            elif b.shape[0] == M:
                beta_full = b.copy()
            else:
                raise ValueError(
                    f"beta must have length {self.K_} or {M}, got {b.shape}"
                )
            sqrt_beta = np.sqrt(beta_full)

        # scale features
        return Phi / sqrt_beta[np.newaxis, :], sqrt_beta

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralCirculantWeightedRidge":
        X, y = check_X_y(X, y, dtype=np.float64, multi_output=True)
        self.n_features_in_ = X.shape[1]

        # build & scale FFT features
        Phi_scaled, sqrt_beta = self._make_fft_features(X)

        # fit Ridge in Fourier domain
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

        # recover Fourier‐space coefficients and back‑scale
        coef_scaled = self.model_.coef_
        if coef_scaled.ndim == 1:
            theta = coef_scaled / sqrt_beta
        else:
            theta = coef_scaled[0] / sqrt_beta

        # split into real & imag parts
        self.F_real_ = theta[: self.k_half_].copy()
        self.F_imag_ = theta[self.k_half_ :].copy()

        # inverse FFT to get input‑space weights
        F_complex = self.F_real_ + 1j * self.F_imag_
        w_full = np.fft.irfft(F_complex, n=self.padded_dim_)
        self.coef_ = w_full[: self.n_features_in_]
        self.intercept_ = self.model_.intercept_
        return self

    def _transform(self, X: np.ndarray):
        check_is_fitted(self, ["model_"])
        # reuse make_fft to rebuild scaled features
        Phi_scaled, _ = self._make_fft_features(X)
        return Phi_scaled

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(self._transform(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(X))


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, root_mean_squared_error

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

    # 2) Vanilla Ridge baseline
    vanilla = Ridge(alpha=1.0, fit_intercept=True, random_state=42)
    vanilla.fit(X_train, y_train)
    preds_v = vanilla.predict(X_test)
    r2_v = r2_score(y_test, preds_v)
    rmse_v = root_mean_squared_error(y_test, preds_v)

    # 3) Spectral circulant weighted Ridge
    K = 10
    beta = 1.0 + (np.linspace(0, 1, K) ** 2)
    circ_ridge = SpectralCirculantWeightedRidge(
        padded_dim=None,
        K=None,
        beta=None,
        alpha=1.0,
        random_state=42,
        verbose=False,
    )
    circ_ridge.fit(X_train, y_train)
    preds_c = circ_ridge.predict(X_test)
    r2_c = r2_score(y_test, preds_c)
    rmse_c = root_mean_squared_error(y_test, preds_c)

    print(f"Vanilla Ridge                   – R²: {r2_v:.3f}, RMSE: {rmse_v:.3f}")
    print(f"SpectralCirculantWeightedRidge – R²: {r2_c:.3f}, RMSE: {rmse_c:.3f}")
