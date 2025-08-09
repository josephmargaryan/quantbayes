# =============================================================================
# spectral_regression.py
# Spectral/circulant + fixed-basis linear REGRESSION models
# - Orthonormal FFT with DC/Nyquist handling
# - Optional Sobolev weighting in frequency (Fourier) or custom weights (fixed basis)
# - Ridge and LinearSVR variants
# - Multi-output regression supported (Ridge natively, SVR via MultiOutputRegressor)
# - Clean mapping back to original-space (W, b)
# =============================================================================

import time
import logging
import warnings
from typing import Optional, Literal, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.exceptions import ConvergenceWarning


# ------------------------------ utilities ------------------------------------


def _make_logger(name: str, verbose: bool) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(
            logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
        )
        logger.addHandler(h)
    return logger


# ---------------------- Fourier feature map (1D) ------------------------------


class _FourierFeatureMap1D:
    """
    Build a real design matrix from 1D RFFT with:
      - orthonormal FFT (norm='ortho'),
      - low-pass truncation to K bins,
      - dropping imag columns for DC and (if even) Nyquist,
      - optional feature scaling: 'none' | 'sobolev' | 'standardize'.

    Scaling: z = (phi - mean) / scale
    Intercept mapping to raw-phi:
        b_raw = b_trained - (mean/scale) @ coef_trained
    """

    def __init__(
        self,
        padded_dim: Optional[int] = None,
        K: Optional[int] = None,
        *,
        feature_scaling: Literal["none", "sobolev", "standardize"] = "none",
        sobolev_s: float = 1.0,
        sobolev_mu: float = 0.0,
        verbose: bool = False,
    ):
        self.padded_dim = padded_dim
        self.K = K
        self.feature_scaling = feature_scaling
        self.sobolev_s = float(sobolev_s)
        self.sobolev_mu = float(sobolev_mu)
        self.verbose = verbose
        self.logger = _make_logger(self.__class__.__name__, verbose)

        # fitted later
        self.D_ = None
        self.padded_dim_ = None
        self.k_half_ = None
        self.K_ = None
        self.col_k_ = None
        self.col_is_real_ = None
        self.n_cols_ = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> "_FourierFeatureMap1D":
        X = check_array(X, dtype=np.float64)
        n, D = X.shape
        self.D_ = D

        pd = D if self.padded_dim is None else int(self.padded_dim)
        if pd < D:
            raise ValueError("padded_dim must be ≥ original feature dim")
        self.padded_dim_ = pd

        self.k_half_ = pd // 2 + 1
        self.K_ = min(self.k_half_, self.k_half_ if self.K is None else int(self.K))
        if self.K_ <= 0:
            raise ValueError("K must be ≥ 1")

        nyquist_bin = self.k_half_ - 1 if (pd % 2 == 0) else None

        # Layout columns
        col_k = []
        col_is_real = []
        for k in range(self.k_half_):
            if k >= self.K_:
                break
            col_k.append(k)
            col_is_real.append(True)  # Re
            if k != 0 and not (nyquist_bin is not None and k == nyquist_bin):
                col_k.append(k)
                col_is_real.append(False)  # Im
        self.col_k_ = np.asarray(col_k, dtype=int)
        self.col_is_real_ = np.asarray(col_is_real, dtype=bool)
        self.n_cols_ = self.col_k_.shape[0]

        t0 = time.time()
        if self.feature_scaling == "sobolev":
            omega = (
                2.0 * np.pi * self.col_k_.astype(np.float64) / float(self.padded_dim_)
            )
            alpha = (self.sobolev_mu + omega**2) ** self.sobolev_s
            self.mean_ = np.zeros(self.n_cols_, dtype=np.float64)
            self.scale_ = np.sqrt(alpha)
        elif self.feature_scaling == "standardize":
            phi = self._phi_raw(X)
            self.mean_ = phi.mean(axis=0)
            sd = phi.std(axis=0)
            self.scale_ = np.where(sd > 1e-12, sd, 1.0)
        else:
            self.mean_ = np.zeros(self.n_cols_, dtype=np.float64)
            self.scale_ = np.ones(self.n_cols_, dtype=np.float64)

        self.logger.info(
            f"Fourier map: pd={self.padded_dim_}, K={self.K_}, Phi_cols={self.n_cols_}, "
            f"built in {time.time()-t0:.3f}s"
        )
        return self

    def _phi_raw(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, dtype=np.float64)
        n, D = X.shape
        if D != self.D_:
            raise ValueError(f"Expected {self.D_} features, got {D}")
        X_pad = np.zeros((n, self.padded_dim_), dtype=np.float64)
        X_pad[:, :D] = X
        Xf = np.fft.rfft(X_pad, axis=1, norm="ortho")
        Phi = np.empty((n, self.n_cols_), dtype=np.float64)
        for j, (k, is_real) in enumerate(zip(self.col_k_, self.col_is_real_)):
            Phi[:, j] = Xf[:, k].real if is_real else Xf[:, k].imag
        return Phi

    def transform(self, X: np.ndarray) -> np.ndarray:
        phi = self._phi_raw(X)
        return (phi - self.mean_) / self.scale_

    # mappings
    def unscale_coef(self, coef_mat: np.ndarray) -> np.ndarray:
        return coef_mat / self.scale_[None, :]

    def intercept_in_raw_phi(
        self, coef_mat: np.ndarray, intercept_vec: np.ndarray
    ) -> np.ndarray:
        shift = (self.mean_ / self.scale_) @ coef_mat.T
        return intercept_vec - shift

    def coef_to_weight(self, coef_mat: np.ndarray) -> np.ndarray:
        coef_raw = self.unscale_coef(coef_mat)  # (T, n_cols)
        T = coef_raw.shape[0]
        F_real = np.zeros((T, self.k_half_), dtype=np.float64)
        F_imag = np.zeros((T, self.k_half_), dtype=np.float64)
        for j, (k, is_real) in enumerate(zip(self.col_k_, self.col_is_real_)):
            if is_real:
                F_real[:, k] = coef_raw[:, j]
            else:
                F_imag[:, k] = coef_raw[:, j]
        Ff = F_real + 1j * F_imag
        w_full = np.fft.irfft(Ff, n=self.padded_dim_, norm="ortho")  # (T, pd)
        return w_full[:, : self.D_]


# --------------------- Fixed-basis (orthonormal) map --------------------------


class _FixedBasisFeatureMap:
    """
    Phi = X @ V  with V orthonormal columns (random+QR or PCA).
    feature_scaling: 'none' | 'standardize' | 'weights' (per-coordinate penalties).
    """

    def __init__(
        self,
        n_spectral: Optional[int] = None,
        basis: Literal["random", "pca"] = "random",
        *,
        feature_scaling: Literal["none", "standardize", "weights"] = "none",
        penalty_weights: Optional[Sequence[float]] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_spectral = n_spectral
        self.basis = basis
        self.feature_scaling = feature_scaling
        self.penalty_weights = (
            None
            if penalty_weights is None
            else np.asarray(penalty_weights, dtype=np.float64)
        )
        self.random_state = random_state
        self.verbose = verbose
        self.logger = _make_logger(self.__class__.__name__, verbose)

        # fitted
        self.D_ = None
        self.n_spectral_ = None
        self.V_ = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> "_FixedBasisFeatureMap":
        X = check_array(X, dtype=np.float64)
        N, D = X.shape
        self.D_ = D

        k = D if self.n_spectral is None else int(self.n_spectral)
        if not (1 <= k <= D):
            raise ValueError(f"n_spectral must be in [1, {D}]")
        self.n_spectral_ = k

        t0 = time.time()
        if self.basis == "random":
            rng = np.random.RandomState(self.random_state)
            G = rng.randn(D, k)
            Q, _ = np.linalg.qr(G)
            self.V_ = Q[:, :k]
        elif self.basis == "pca":
            from sklearn.decomposition import PCA

            pca = PCA(n_components=k, random_state=self.random_state)
            pca.fit(X)
            self.V_ = pca.components_.T
        else:
            raise ValueError(f"Unknown basis='{self.basis}'")
        self.logger.info(
            f"Built basis '{self.basis}' V.shape={self.V_.shape} in {time.time()-t0:.3f}s"
        )

        Phi = X @ self.V_

        if self.feature_scaling == "weights":
            if self.penalty_weights is None:
                raise ValueError("feature_scaling='weights' requires penalty_weights.")
            pw = np.asarray(self.penalty_weights, dtype=np.float64)
            if pw.shape[0] != k:
                raise ValueError(
                    f"penalty_weights length must be {k}, got {pw.shape[0]}."
                )
            self.mean_ = np.zeros(k, dtype=np.float64)
            self.scale_ = np.sqrt(pw)
        elif self.feature_scaling == "standardize":
            self.mean_ = Phi.mean(axis=0)
            sd = Phi.std(axis=0)
            self.scale_ = np.where(sd > 1e-12, sd, 1.0)
        else:
            self.mean_ = np.zeros(k, dtype=np.float64)
            self.scale_ = np.ones(k, dtype=np.float64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, dtype=np.float64)
        return (X @ self.V_ - self.mean_) / self.scale_

    def unscale_coef(self, coef_mat: np.ndarray) -> np.ndarray:
        return coef_mat / self.scale_[None, :]

    def intercept_in_raw_phi(
        self, coef_mat: np.ndarray, intercept_vec: np.ndarray
    ) -> np.ndarray:
        shift = (self.mean_ / self.scale_) @ coef_mat.T
        return intercept_vec - shift

    def coef_to_weight(self, coef_mat: np.ndarray) -> np.ndarray:
        beta = self.unscale_coef(coef_mat)
        W = beta @ self.V_.T
        return W


# =============================== RIDGE (L2) ===================================


class SpectralCirculantRidge(BaseEstimator, RegressorMixin):
    """
    Ridge regression on truncated RFFT features (orthonormal), with optional
    Sobolev scaling: alpha_k = (mu + omega_k^2)^s.
    """

    def __init__(
        self,
        padded_dim: Optional[int] = None,
        K: Optional[int] = None,
        *,
        feature_scaling: Literal["none", "sobolev", "standardize"] = "none",
        sobolev_s: float = 1.0,
        sobolev_mu: float = 0.0,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.padded_dim = padded_dim
        self.K = K
        self.feature_scaling = feature_scaling
        self.sobolev_s = sobolev_s
        self.sobolev_mu = sobolev_mu
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose
        self.logger = _make_logger(self.__class__.__name__, verbose)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralCirculantRidge":
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True, multi_output=True)
        self._y_was_1d_ = y.ndim == 1
        if self._y_was_1d_:
            y = y.reshape(-1, 1)

        self._map = _FourierFeatureMap1D(
            padded_dim=self.padded_dim,
            K=self.K,
            feature_scaling=self.feature_scaling,
            sobolev_s=self.sobolev_s,
            sobolev_mu=self.sobolev_mu,
            verbose=self.verbose,
        ).fit(X)

        Z = self._map.transform(X)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model_ = Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                random_state=self.random_state,
            ).fit(Z, y)

        self.n_features_in_ = X.shape[1]
        self.coef_spec_ = np.atleast_2d(self.model_.coef_)  # (T, n_cols)
        self.intercept_spec_ = np.atleast_1d(self.model_.intercept_)  # (T,)
        self.W_, self.b_ = self.linear_params_()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        Z = self._map.transform(check_array(X, dtype=np.float64))
        yhat = self.model_.predict(Z)
        if self._y_was_1d_:
            return yhat.ravel()
        return yhat

    @property
    def weight_(self) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        W = self._map.coef_to_weight(self.coef_spec_)
        if self._y_was_1d_:
            return W[0]
        return W

    def linear_params_(self) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["model_"])
        W = self._map.coef_to_weight(self.coef_spec_)
        b = self._map.intercept_in_raw_phi(self.coef_spec_, self.intercept_spec_)
        if self._y_was_1d_:
            return W[0], float(b[0])
        return W, b


class SpectralRidge(BaseEstimator, RegressorMixin):
    """
    Ridge regression on a fixed orthonormal basis Phi = X @ V.
    'weights' scaling allows arbitrary diagonal penalties α_j via z = Phi / sqrt(α_j).
    """

    def __init__(
        self,
        n_spectral: Optional[int] = None,
        basis: Literal["random", "pca"] = "random",
        *,
        feature_scaling: Literal["none", "standardize", "weights"] = "none",
        penalty_weights: Optional[Sequence[float]] = None,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_spectral = n_spectral
        self.basis = basis
        self.feature_scaling = feature_scaling
        self.penalty_weights = penalty_weights
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose
        self.logger = _make_logger(self.__class__.__name__, verbose)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralRidge":
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True, multi_output=True)
        self._y_was_1d_ = y.ndim == 1
        if self._y_was_1d_:
            y = y.reshape(-1, 1)

        self._map = _FixedBasisFeatureMap(
            n_spectral=self.n_spectral,
            basis=self.basis,
            feature_scaling=self.feature_scaling,
            penalty_weights=self.penalty_weights,
            random_state=self.random_state,
            verbose=self.verbose,
        ).fit(X)

        Z = self._map.transform(X)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model_ = Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                random_state=self.random_state,
            ).fit(Z, y)

        self.n_features_in_ = X.shape[1]
        self.coef_spec_ = np.atleast_2d(self.model_.coef_)  # (T, k)
        self.intercept_spec_ = np.atleast_1d(self.model_.intercept_)  # (T,)
        self.W_, self.b_ = self.linear_params_()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        Z = self._map.transform(check_array(X, dtype=np.float64))
        yhat = self.model_.predict(Z)
        if self._y_was_1d_:
            return yhat.ravel()
        return yhat

    @property
    def weight_(self) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        W = self._map.coef_to_weight(self.coef_spec_)
        if self._y_was_1d_:
            return W[0]
        return W

    def linear_params_(self) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["model_"])
        W = self._map.coef_to_weight(self.coef_spec_)
        b = self._map.intercept_in_raw_phi(self.coef_spec_, self.intercept_spec_)
        if self._y_was_1d_:
            return W[0], float(b[0])
        return W, b


# ================================ SVR (ε) =====================================


class SpectralCirculantSVR(BaseEstimator, RegressorMixin):
    """
    LinearSVR on Fourier features with optional Sobolev scaling.
    Multi-output handled via MultiOutputRegressor.
    """

    def __init__(
        self,
        padded_dim: Optional[int] = None,
        K: Optional[int] = None,
        *,
        feature_scaling: Literal["none", "sobolev", "standardize"] = "none",
        sobolev_s: float = 1.0,
        sobolev_mu: float = 0.0,
        C: float = 1.0,
        epsilon: float = 0.0,
        tol: float = 1e-4,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.padded_dim = padded_dim
        self.K = K
        self.feature_scaling = feature_scaling
        self.sobolev_s = sobolev_s
        self.sobolev_mu = sobolev_mu
        self.C = C
        self.epsilon = epsilon
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.logger = _make_logger(self.__class__.__name__, verbose)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralCirculantSVR":
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True, multi_output=True)
        self._y_was_1d_ = y.ndim == 1

        self._map = _FourierFeatureMap1D(
            padded_dim=self.padded_dim,
            K=self.K,
            feature_scaling=self.feature_scaling,
            sobolev_s=self.sobolev_s,
            sobolev_mu=self.sobolev_mu,
            verbose=self.verbose,
        ).fit(X)

        Z = self._map.transform(X)

        base = LinearSVR(
            C=self.C,
            epsilon=self.epsilon,
            tol=self.tol,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            if self._y_was_1d_:
                self.model_ = base.fit(Z, y)
            else:
                self.model_ = MultiOutputRegressor(base).fit(Z, y)

        # Collect spec coefs/intercepts
        if self._y_was_1d_:
            coef = np.atleast_2d(self.model_.coef_)
            intercept = np.atleast_1d(self.model_.intercept_)
        else:
            ests = self.model_.estimators_
            coef = np.vstack([np.atleast_1d(e.coef_) for e in ests])
            intercept = np.array([float(e.intercept_) for e in ests])

        self.coef_spec_ = coef
        self.intercept_spec_ = intercept
        self.n_features_in_ = X.shape[1]
        self.W_, self.b_ = self.linear_params_()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        Z = self._map.transform(check_array(X, dtype=np.float64))
        yhat = self.model_.predict(Z)
        if self._y_was_1d_:
            return np.asarray(yhat).ravel()
        return np.asarray(yhat)

    @property
    def weight_(self) -> np.ndarray:
        check_is_fitted(self, ["coef_spec_"])
        W = self._map.coef_to_weight(self.coef_spec_)
        if self._y_was_1d_:
            return W[0]
        return W

    def linear_params_(self) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["coef_spec_"])
        W = self._map.coef_to_weight(self.coef_spec_)
        b = self._map.intercept_in_raw_phi(self.coef_spec_, self.intercept_spec_)
        if self._y_was_1d_:
            return W[0], float(b[0])
        return W, b


class SpectralSVR(BaseEstimator, RegressorMixin):
    """
    LinearSVR on fixed orthonormal basis features Phi = X @ V.
    'weights' scaling allows arbitrary diagonal penalties.
    """

    def __init__(
        self,
        n_spectral: Optional[int] = None,
        basis: Literal["random", "pca"] = "random",
        *,
        feature_scaling: Literal["none", "standardize", "weights"] = "none",
        penalty_weights: Optional[Sequence[float]] = None,
        C: float = 1.0,
        epsilon: float = 0.0,
        tol: float = 1e-4,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_spectral = n_spectral
        self.basis = basis
        self.feature_scaling = feature_scaling
        self.penalty_weights = penalty_weights
        self.C = C
        self.epsilon = epsilon
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.logger = _make_logger(self.__class__.__name__, verbose)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralSVR":
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True, multi_output=True)
        self._y_was_1d_ = y.ndim == 1

        self._map = _FixedBasisFeatureMap(
            n_spectral=self.n_spectral,
            basis=self.basis,
            feature_scaling=self.feature_scaling,
            penalty_weights=self.penalty_weights,
            random_state=self.random_state,
            verbose=self.verbose,
        ).fit(X)

        Z = self._map.transform(X)

        base = LinearSVR(
            C=self.C,
            epsilon=self.epsilon,
            tol=self.tol,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            if self._y_was_1d_:
                self.model_ = base.fit(Z, y)
            else:
                self.model_ = MultiOutputRegressor(base).fit(Z, y)

        if self._y_was_1d_:
            coef = np.atleast_2d(self.model_.coef_)
            intercept = np.atleast_1d(self.model_.intercept_)
        else:
            ests = self.model_.estimators_
            coef = np.vstack([np.atleast_1d(e.coef_) for e in ests])
            intercept = np.array([float(e.intercept_) for e in ests])

        self.coef_spec_ = coef
        self.intercept_spec_ = intercept
        self.n_features_in_ = X.shape[1]
        self.W_, self.b_ = self.linear_params_()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        Z = self._map.transform(check_array(X, dtype=np.float64))
        yhat = self.model_.predict(Z)
        if self._y_was_1d_:
            return np.asarray(yhat).ravel()
        return np.asarray(yhat)

    @property
    def weight_(self) -> np.ndarray:
        check_is_fitted(self, ["coef_spec_"])
        W = self._map.coef_to_weight(self.coef_spec_)
        if self._y_was_1d_:
            return W[0]
        return W

    def linear_params_(self) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["coef_spec_"])
        W = self._map.coef_to_weight(self.coef_spec_)
        b = self._map.intercept_in_raw_phi(self.coef_spec_, self.intercept_spec_)
        if self._y_was_1d_:
            return W[0], float(b[0])
        return W, b
