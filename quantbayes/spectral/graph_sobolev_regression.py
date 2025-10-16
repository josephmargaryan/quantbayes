# =============================================================================
# graph_sobolev_regression.py
# Graph/Laplacian-Sobolev REGRESSION models (Ridge & SVR)
# - Structured quadratic prior w^T L w expressed via spectral scaling
# - Dense or sparse Laplacians; optional adjacency->L (unnormalized or normalized)
# - Multi-output support (Ridge native, SVR via MultiOutputRegressor)
# - Clean mapping back to original-space (W, b)
# =============================================================================

import time
import logging
import warnings
from typing import Optional, Literal, Sequence, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.exceptions import ConvergenceWarning

# SciPy optional for sparse graphs / large D
try:
    from scipy.sparse import issparse, spmatrix, csr_matrix, diags
    from scipy.sparse.linalg import eigsh

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    spmatrix = None  # type: ignore
    SCIPY_AVAILABLE = False


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


def _as_csr(M: Union[np.ndarray, "spmatrix"]) -> "csr_matrix":
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "SciPy is required for sparse graph Laplacians. Install scipy>=1.8."
        )
    if isinstance(M, np.ndarray):
        return csr_matrix(M)
    if issparse(M):
        return M.tocsr()
    raise TypeError("L or A must be a numpy array or a scipy.sparse matrix.")


def _build_chain_laplacian(D: int) -> "csr_matrix":
    if not SCIPY_AVAILABLE:
        L = np.zeros((D, D), dtype=float)
        for i in range(D):
            if i > 0:
                L[i, i] += 1
                L[i, i - 1] -= 1
            if i < D - 1:
                L[i, i] += 1
                L[i, i + 1] -= 1
        return _as_csr(L)
    data = []
    rows = []
    cols = []
    for i in range(D):
        deg = 0
        if i > 0:
            rows += [i]
            cols += [i - 1]
            data += [-1.0]
            deg += 1
        if i < D - 1:
            rows += [i]
            cols += [i + 1]
            data += [-1.0]
            deg += 1
        rows += [i]
        cols += [i]
        data += [float(deg)]
    return csr_matrix((data, (rows, cols)), shape=(D, D))


def _normalized_laplacian(A: "csr_matrix") -> "csr_matrix":
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy is required to build normalized Laplacians.")
    deg = np.asarray(A.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-32))
    D_inv_sqrt = diags(d_inv_sqrt)
    I = diags(np.ones_like(deg))
    return I - D_inv_sqrt @ A @ D_inv_sqrt


def _dense_eigh_smallest(L: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(L)
    idx = np.argsort(evals)[:k]
    return evals[idx], evecs[:, idx]


def _sparse_eigsh_smallest(
    L: "csr_matrix", k: int, tol: float = 1e-6, maxiter: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy required for sparse eigensolvers.")
    evals, evecs = eigsh(L, k=k, which="SM", tol=tol, maxiter=maxiter)
    idx = np.argsort(evals)
    return evals[idx], evecs[:, idx]


# --------------------- Graph-spectral feature map -----------------------------


class _GraphSpectralFeatureMap:
    """
    Build Phi = X @ V where V are eigenvectors of an SPD operator L (e.g. a
    graph Laplacian over feature indices). Then scale columns according to:

      - 'sobolev': alpha_j = (mu + lambda_j)^s
      - 'weights': alpha_j = user-given penalties
      - 'standardize': z-score columns of Phi
      - 'none': no scaling

    Training on Z = (Phi - mean)/sqrt(alpha) with uniform L2 on coefficients
    is equivalent to a structured quadratic penalty in original space.
    """

    def __init__(
        self,
        L: Optional[Union[np.ndarray, "spmatrix"]] = None,
        A: Optional[Union[np.ndarray, "spmatrix"]] = None,
        *,
        laplacian_type: Literal["unnormalized", "normalized"] = "unnormalized",
        n_components: Optional[int] = None,
        drop_nullspace: bool = True,
        nullspace_tol: float = 1e-10,
        feature_scaling: Literal[
            "sobolev", "weights", "standardize", "none"
        ] = "sobolev",
        sobolev_s: float = 1.0,
        sobolev_mu: float = 1e-3,
        penalty_weights: Optional[Sequence[float]] = None,
        eig_tol: float = 1e-6,
        eig_maxiter: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.L = L
        self.A = A
        self.laplacian_type = laplacian_type
        self.n_components = n_components
        self.drop_nullspace = drop_nullspace
        self.nullspace_tol = nullspace_tol
        self.feature_scaling = feature_scaling
        self.sobolev_s = float(sobolev_s)
        self.sobolev_mu = float(sobolev_mu)
        self.penalty_weights = (
            None
            if penalty_weights is None
            else np.asarray(penalty_weights, dtype=float)
        )
        self.eig_tol = eig_tol
        self.eig_maxiter = eig_maxiter
        self.random_state = random_state
        self.verbose = verbose
        self.logger = _make_logger(self.__class__.__name__, verbose)

        # fitted
        self.D_ = None
        self.V_ = None
        self.evals_ = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> "_GraphSpectralFeatureMap":
        X = check_array(X, dtype=np.float64)
        N, D = X.shape
        self.D_ = D

        if self.L is None:
            if self.A is not None:
                A = _as_csr(self.A)
                if self.laplacian_type == "normalized":
                    L = _normalized_laplacian(A)
                else:
                    deg = np.asarray(A.sum(axis=1)).ravel()
                    if not SCIPY_AVAILABLE:
                        L = np.diag(deg) - A.toarray()
                    else:
                        L = diags(deg) - A
                L_csr = _as_csr(L)
            else:
                L_csr = _build_chain_laplacian(D)
        else:
            L_csr = _as_csr(self.L)

        ask_k = self.n_components or D
        if self.drop_nullspace:
            ask_k = min(D, ask_k + 4)

        t0 = time.time()
        if SCIPY_AVAILABLE and (L_csr.count_nonzero() / (D * D) < 0.25 or D > 800):
            evals, V = _sparse_eigsh_smallest(
                L_csr,
                k=max(1, min(ask_k, D)),
                tol=self.eig_tol,
                maxiter=self.eig_maxiter,
            )
        else:
            evals, V = _dense_eigh_smallest(
                L_csr.toarray() if SCIPY_AVAILABLE else np.asarray(L_csr),
                k=min(ask_k, D),
            )
        self.logger.info(
            f"Eigendecomposition: D={D}, k={V.shape[1]}, time={time.time()-t0:.3f}s"
        )

        if self.drop_nullspace:
            keep = evals > self.nullspace_tol
            if not np.any(keep):
                keep = np.ones_like(evals, dtype=bool)
                keep[0] = True
            evals = evals[keep]
            V = V[:, keep]

        if self.n_components is not None and V.shape[1] > self.n_components:
            V = V[:, : self.n_components]
            evals = evals[: self.n_components]

        self.V_ = V.astype(np.float64, copy=False)
        self.evals_ = evals.astype(np.float64, copy=False)

        Phi = X @ self.V_
        if self.feature_scaling == "sobolev":
            alpha = (self.sobolev_mu + self.evals_) ** self.sobolev_s
            self.mean_ = np.zeros_like(alpha)
            self.scale_ = np.sqrt(alpha)
        elif self.feature_scaling == "weights":
            if self.penalty_weights is None:
                raise ValueError("feature_scaling='weights' requires penalty_weights.")
            pw = np.asarray(self.penalty_weights, dtype=np.float64)
            if pw.shape[0] != self.V_.shape[1]:
                raise ValueError(
                    f"penalty_weights length must be {self.V_.shape[1]}, got {pw.shape[0]}."
                )
            self.mean_ = np.zeros_like(pw)
            self.scale_ = np.sqrt(pw)
        elif self.feature_scaling == "standardize":
            self.mean_ = Phi.mean(axis=0)
            sd = Phi.std(axis=0)
            self.scale_ = np.where(sd > 1e-12, sd, 1.0)
        else:
            self.mean_ = np.zeros(self.V_.shape[1], dtype=np.float64)
            self.scale_ = np.ones(self.V_.shape[1], dtype=np.float64)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, dtype=np.float64)
        Phi = X @ self.V_
        return (Phi - self.mean_) / self.scale_

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


class GraphSobolevRidge(BaseEstimator, RegressorMixin):
    """
    Ridge regression with a Graph/Laplacian-Sobolev prior implemented via
    spectral feature scaling.
    """

    def __init__(
        self,
        L: Optional[Union[np.ndarray, "spmatrix"]] = None,
        A: Optional[Union[np.ndarray, "spmatrix"]] = None,
        *,
        laplacian_type: Literal["unnormalized", "normalized"] = "unnormalized",
        n_components: Optional[int] = None,
        drop_nullspace: bool = True,
        nullspace_tol: float = 1e-10,
        feature_scaling: Literal[
            "sobolev", "weights", "standardize", "none"
        ] = "sobolev",
        sobolev_s: float = 1.0,
        sobolev_mu: float = 1e-3,
        penalty_weights: Optional[Sequence[float]] = None,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = False,
        eig_tol: float = 1e-6,
        eig_maxiter: Optional[int] = None,
    ):
        self.L = L
        self.A = A
        self.laplacian_type = laplacian_type
        self.n_components = n_components
        self.drop_nullspace = drop_nullspace
        self.nullspace_tol = nullspace_tol
        self.feature_scaling = feature_scaling
        self.sobolev_s = sobolev_s
        self.sobolev_mu = sobolev_mu
        self.penalty_weights = penalty_weights
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose
        self.eig_tol = eig_tol
        self.eig_maxiter = eig_maxiter
        self.logger = _make_logger(self.__class__.__name__, verbose)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GraphSobolevRidge":
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True, multi_output=True)
        self._y_was_1d_ = y.ndim == 1
        if self._y_was_1d_:
            y = y.reshape(-1, 1)

        self.map_ = _GraphSpectralFeatureMap(
            L=self.L,
            A=self.A,
            laplacian_type=self.laplacian_type,
            n_components=self.n_components,
            drop_nullspace=self.drop_nullspace,
            nullspace_tol=self.nullspace_tol,
            feature_scaling=self.feature_scaling,
            sobolev_s=self.sobolev_s,
            sobolev_mu=self.sobolev_mu,
            penalty_weights=self.penalty_weights,
            eig_tol=self.eig_tol,
            eig_maxiter=self.eig_maxiter,
            random_state=self.random_state,
            verbose=self.verbose,
        ).fit(X)

        Z = self.map_.transform(X)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model_ = Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                random_state=self.random_state,
            ).fit(Z, y)

        self.n_features_in_ = X.shape[1]
        self.coef_spec_ = np.atleast_2d(self.model_.coef_)
        self.intercept_spec_ = np.atleast_1d(self.model_.intercept_)
        self.W_, self.b_ = self.linear_params_()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        Z = self.map_.transform(check_array(X, dtype=np.float64))
        yhat = self.model_.predict(Z)
        if self._y_was_1d_:
            return yhat.ravel()
        return yhat

    @property
    def weight_(self) -> np.ndarray:
        check_is_fitted(self, ["coef_spec_"])
        W = self.map_.coef_to_weight(self.coef_spec_)
        if self._y_was_1d_:
            return W[0]
        return W

    def linear_params_(self) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["coef_spec_"])
        W = self.map_.coef_to_weight(self.coef_spec_)
        b = self.map_.intercept_in_raw_phi(self.coef_spec_, self.intercept_spec_)
        if self._y_was_1d_:
            return W[0], float(b[0])
        return W, b


# ================================ SVR (ε) =====================================


class GraphSobolevSVR(BaseEstimator, RegressorMixin):
    """
    LinearSVR with Graph/Laplacian-Sobolev prior via spectral feature scaling.
    Multi-output handled via MultiOutputRegressor.
    """

    def __init__(
        self,
        L: Optional[Union[np.ndarray, "spmatrix"]] = None,
        A: Optional[Union[np.ndarray, "spmatrix"]] = None,
        *,
        laplacian_type: Literal["unnormalized", "normalized"] = "unnormalized",
        n_components: Optional[int] = None,
        drop_nullspace: bool = True,
        nullspace_tol: float = 1e-10,
        feature_scaling: Literal[
            "sobolev", "weights", "standardize", "none"
        ] = "sobolev",
        sobolev_s: float = 1.0,
        sobolev_mu: float = 1e-3,
        penalty_weights: Optional[Sequence[float]] = None,
        C: float = 1.0,
        epsilon: float = 0.0,
        tol: float = 1e-4,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
        verbose: bool = False,
        eig_tol: float = 1e-6,
        eig_maxiter: Optional[int] = None,
    ):
        self.L = L
        self.A = A
        self.laplacian_type = laplacian_type
        self.n_components = n_components
        self.drop_nullspace = drop_nullspace
        self.nullspace_tol = nullspace_tol
        self.feature_scaling = feature_scaling
        self.sobolev_s = sobolev_s
        self.sobolev_mu = sobolev_mu
        self.penalty_weights = penalty_weights
        self.C = C
        self.epsilon = epsilon
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.eig_tol = eig_tol
        self.eig_maxiter = eig_maxiter
        self.logger = _make_logger(self.__class__.__name__, verbose)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GraphSobolevSVR":
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True, multi_output=True)
        self._y_was_1d_ = y.ndim == 1

        self.map_ = _GraphSpectralFeatureMap(
            L=self.L,
            A=self.A,
            laplacian_type=self.laplacian_type,
            n_components=self.n_components,
            drop_nullspace=self.drop_nullspace,
            nullspace_tol=self.nullspace_tol,
            feature_scaling=self.feature_scaling,
            sobolev_s=self.sobolev_s,
            sobolev_mu=self.sobolev_mu,
            penalty_weights=self.penalty_weights,
            eig_tol=self.eig_tol,
            eig_maxiter=self.eig_maxiter,
            random_state=self.random_state,
            verbose=self.verbose,
        ).fit(X)

        Z = self.map_.transform(X)

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
        Z = self.map_.transform(check_array(X, dtype=np.float64))
        yhat = self.model_.predict(Z)
        if self._y_was_1d_:
            return np.asarray(yhat).ravel()
        return np.asarray(yhat)

    @property
    def weight_(self) -> np.ndarray:
        check_is_fitted(self, ["coef_spec_"])
        W = self.map_.coef_to_weight(self.coef_spec_)
        if self._y_was_1d_:
            return W[0]
        return W

    def linear_params_(self) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["coef_spec_"])
        W = self.map_.coef_to_weight(self.coef_spec_)
        b = self.map_.intercept_in_raw_phi(self.coef_spec_, self.intercept_spec_)
        if self._y_was_1d_:
            return W[0], float(b[0])
        return W, b


# Optional shorter aliases:
GraphLaplacianRidge = GraphSobolevRidge
GraphLaplacianSVR = GraphSobolevSVR
