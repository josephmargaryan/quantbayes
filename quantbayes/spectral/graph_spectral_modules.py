# =============================================================================
# graph_spectral_models.py
# Graph/Laplacian-Sobolev linear classifiers (Logistic & SVM)
# =============================================================================
from __future__ import annotations

import time
import logging
import warnings
from typing import Optional, Literal, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

# --- SciPy (optional) ---
try:
    from scipy.sparse import issparse, csr_matrix, diags
    from scipy.sparse.linalg import eigsh

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    issparse = None  # type: ignore[assignment]
    csr_matrix = None  # type: ignore[assignment]
    diags = None  # type: ignore[assignment]
    eigsh = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False

# --- Type aliases only for the type-checker (no runtime dependency on SciPy types) ---
if TYPE_CHECKING:
    from scipy.sparse import spmatrix as SpMatrix
    from scipy.sparse import csr_matrix as CsrMatrix


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


def _as_csr(M: Union[np.ndarray, "SpMatrix"]) -> Union[np.ndarray, "CsrMatrix"]:
    """
    Return a CSR matrix when SciPy is available; otherwise pass through dense arrays.
    """
    if SCIPY_AVAILABLE:
        if isinstance(M, np.ndarray):
            return csr_matrix(M)  # type: ignore[call-arg]
        if issparse is not None and issparse(M):  # type: ignore[operator]
            return M.tocsr()  # type: ignore[union-attr]
        raise TypeError("L or A must be a numpy array or a scipy.sparse matrix.")
    else:
        if isinstance(M, np.ndarray):
            return M  # keep dense when SciPy is unavailable
        raise ImportError("SciPy is required when passing sparse matrices.")


def _build_chain_laplacian(D: int) -> Union[np.ndarray, "CsrMatrix"]:
    """
    Unnormalized 1-D chain Laplacian on D features: L = D - A with path graph.
    """
    if not SCIPY_AVAILABLE:
        # Dense fallback (small/medium D)
        L = np.zeros((D, D), dtype=float)
        for i in range(D):
            if i > 0:
                L[i, i] += 1
                L[i, i - 1] -= 1
            if i < D - 1:
                L[i, i] += 1
                L[i, i + 1] -= 1
        return L
    # Sparse construction
    data = []
    rows = []
    cols = []
    for i in range(D):
        deg = 0
        if i > 0:
            rows.append(i)
            cols.append(i - 1)
            data.append(-1.0)
            deg += 1
        if i < D - 1:
            rows.append(i)
            cols.append(i + 1)
            data.append(-1.0)
            deg += 1
        rows.append(i)
        cols.append(i)
        data.append(float(deg))
    return csr_matrix((data, (rows, cols)), shape=(D, D))  # type: ignore[call-arg]


def _normalized_laplacian(A: "CsrMatrix") -> "CsrMatrix":
    """
    Symmetric normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}.
    A is assumed nonnegative, symmetric, with zero diagonal.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy is required to build normalized Laplacians.")
    deg = np.asarray(A.sum(axis=1)).ravel()  # type: ignore[union-attr]
    with np.errstate(divide="ignore"):
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-32))
    D_inv_sqrt = diags(d_inv_sqrt)  # type: ignore[call-arg]
    I = diags(np.ones_like(deg))  # type: ignore[call-arg]
    return I - D_inv_sqrt @ A @ D_inv_sqrt  # type: ignore[operator]


def _dense_eigh_smallest(L: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dense symmetric eigendecomposition; returns k smallest eigenpairs.
    Only for modest D. Uses np.linalg.eigh then slices.
    """
    evals, evecs = np.linalg.eigh(L)
    idx = np.argsort(evals)[:k]
    return evals[idx], evecs[:, idx]


def _sparse_eigsh_smallest(
    L: "CsrMatrix", k: int, tol: float = 1e-6, maxiter: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smallest-k eigenpairs of symmetric PSD sparse L.
    Robust to ARPACK limitations (k >= n) and solver failures.
    Falls back to dense eigh as needed.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy required for sparse eigensolvers.")

    n = L.shape[0]  # type: ignore[union-attr]

    # ARPACK requires 1 <= k < n
    if k >= n:
        Ld = L.toarray()  # type: ignore[union-attr]
        evals_all, evecs_all = np.linalg.eigh(Ld)
        idx = np.argsort(evals_all)[:n]
        return evals_all[idx], evecs_all[:, idx]

    try:
        evals, evecs = eigsh(L, k=k, which="SM", tol=tol, maxiter=maxiter)  # type: ignore[arg-type]
        idx = np.argsort(evals)
        return evals[idx], evecs[:, idx]
    except Exception:
        # Fallback if ARPACK struggles (e.g., near-singular L)
        Ld = L.toarray()  # type: ignore[union-attr]
        evals_all, evecs_all = np.linalg.eigh(Ld)
        idx = np.argsort(evals_all)[:k]
        return evals_all[idx], evecs_all[:, idx]


# --------------------- Graph-spectral feature map (over features) --------------


class _GraphSpectralFeatureMap:
    """
    Build Phi = X @ V where columns of V are eigenvectors of an SPD operator L
    (e.g., graph Laplacian over feature indices). Then apply one of:
      - 'sobolev': alpha_j = (mu + lambda_j)^s
      - 'weights': alpha_j = user-specified per-coordinate penalties
      - 'standardize': z-score columns of Phi
      - 'none': no scaling

    Training on Z = (Phi - mean)/sqrt(alpha) with standard L2 on coefficients
    is equivalent to a structured quadratic penalty in original space.
    """

    def __init__(
        self,
        L: Optional[Union[np.ndarray, "SpMatrix"]] = None,
        A: Optional[Union[np.ndarray, "SpMatrix"]] = None,
        *,
        laplacian_type: Literal["unnormalized", "normalized"] = "unnormalized",
        n_components: Optional[int] = None,
        drop_nullspace: bool = True,
        nullspace_tol: float = 1e-10,
        feature_scaling: Literal[
            "none", "sobolev", "standardize", "weights"
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
        self.D_: Optional[int] = None
        self.V_: Optional[np.ndarray] = None
        self.evals_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_GraphSpectralFeatureMap":
        X = check_array(X, dtype=np.float64)
        _, D = X.shape
        self.D_ = D

        # Build or accept L
        if self.L is None:
            if self.A is not None:
                A = _as_csr(self.A)
                if self.laplacian_type == "normalized":
                    # Normalized Laplacian requires SciPy
                    L = _normalized_laplacian(A)
                else:
                    # Unnormalized: L = D - A
                    if SCIPY_AVAILABLE and issparse is not None and issparse(A):  # type: ignore[arg-type]
                        deg = np.asarray(A.sum(axis=1)).ravel()  # type: ignore[union-attr]
                        L = diags(deg) - A  # type: ignore[operator]
                    else:
                        A_dense = np.asarray(A)
                        deg = np.asarray(A_dense.sum(axis=1)).ravel()
                        L = np.diag(deg) - A_dense
                L_csr = _as_csr(L)
            else:
                # Default: 1-D chain Laplacian (smoothness over feature order)
                L_csr = _as_csr(_build_chain_laplacian(D))
        else:
            L_csr = _as_csr(self.L)

        # Choose k and compute k smallest eigenpairs (robust selection)
        k_req = self.n_components or D
        if self.drop_nullspace:
            k_req = min(D, k_req + 4)  # small over-ask to skip nullspace
        else:
            k_req = min(D, k_req)

        t0 = time.time()

        # Estimate sparsity safely
        if SCIPY_AVAILABLE and hasattr(L_csr, "count_nonzero"):
            try:
                nnz = L_csr.count_nonzero()  # type: ignore[union-attr]
            except Exception:
                nnz = D * D
        else:
            nnz = int(np.count_nonzero(np.asarray(L_csr)))
        sparsity = nnz / float(D * D)

        # Use sparse solver only if we truly need < D eigenpairs
        use_sparse = (
            SCIPY_AVAILABLE
            and (k_req < D)
            and (issparse is not None)
            and hasattr(L_csr, "tocsr")
            and (sparsity < 0.25 or D > 800)
        )

        if use_sparse:
            evals, V = _sparse_eigsh_smallest(
                L_csr, k=max(1, k_req), tol=self.eig_tol, maxiter=self.eig_maxiter
            )
        else:
            L_dense = L_csr.toarray() if SCIPY_AVAILABLE and hasattr(L_csr, "toarray") else np.asarray(L_csr)  # type: ignore[union-attr]
            evals, V = _dense_eigh_smallest(L_dense, k=k_req)

        self.logger.info(
            f"Eigendecomposition: D={D}, k={V.shape[1]}, time={time.time()-t0:.3f}s"
        )

        # Drop nullspace (near-zero evals) if requested
        if self.drop_nullspace:
            keep = evals > self.nullspace_tol
            if not np.any(keep):
                keep = np.ones_like(evals, dtype=bool)
                keep[0] = True
            evals = evals[keep]
            V = V[:, keep]

        # Truncate to n_components if provided
        if self.n_components is not None and V.shape[1] > self.n_components:
            V = V[:, : self.n_components]
            evals = evals[: self.n_components]

        self.V_ = V.astype(np.float64, copy=False)
        self.evals_ = evals.astype(np.float64, copy=False)

        # Build Phi to compute scaling if needed
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
        else:  # 'none'
            self.mean_ = np.zeros(self.V_.shape[1], dtype=np.float64)
            self.scale_ = np.ones(self.V_.shape[1], dtype=np.float64)

        return self

    # ---------- transform & mappings ----------

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, dtype=np.float64)
        if self.V_ is None:
            raise RuntimeError("Feature map not fitted.")
        Phi = X @ self.V_
        return (Phi - self.mean_) / self.scale_

    def unscale_coef(self, coef_mat: np.ndarray) -> np.ndarray:
        """coef on z -> coef on raw Phi coordinates."""
        return coef_mat / self.scale_[None, :]

    def intercept_in_raw_phi(
        self, coef_mat: np.ndarray, intercept_vec: np.ndarray
    ) -> np.ndarray:
        """Undo centering for intercept."""
        shift = (self.mean_ / self.scale_) @ coef_mat.T
        return intercept_vec - shift

    def coef_to_weight(self, coef_mat: np.ndarray) -> np.ndarray:
        """Map spectral coefficients to original-space weights W (C, D)."""
        beta = self.unscale_coef(coef_mat)  # (C, k)
        W = beta @ self.V_.T  # (C, D)
        return W

    # Convenience for analysis
    @property
    def evals(self) -> np.ndarray:
        if self.evals_ is None:
            raise RuntimeError("Not fitted.")
        return self.evals_.copy()

    @property
    def basis(self) -> np.ndarray:
        if self.V_ is None:
            raise RuntimeError("Not fitted.")
        return self.V_.copy()


# ======================= Estimators: Logistic (Graph-Sobolev) =================


class GraphSobolevLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic regression with a Graph/Laplacian-Sobolev prior:
        penalty in spectral coords: sum_j alpha_j * beta_j^2
      with alpha_j = (mu + lambda_j)^s by default.
    """

    def __init__(
        self,
        L: Optional[Union[np.ndarray, "SpMatrix"]] = None,
        A: Optional[Union[np.ndarray, "SpMatrix"]] = None,
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
        solver: str = "lbfgs",
        dual: bool = False,
        C: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-4,
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
        self.solver = solver
        self.dual = dual
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.eig_tol = eig_tol
        self.eig_maxiter = eig_maxiter

        self.logger = _make_logger(self.__class__.__name__, verbose)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GraphSobolevLogisticRegression":
        X, y = check_X_y(X, y, dtype=np.float64, ensure_min_samples=2)

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

        # dual is only valid with 'liblinear' in scikit
        dual = self.dual
        if dual and self.solver != "liblinear":
            self.logger.warning(
                "dual=True is only valid with solver='liblinear'; overriding to dual=False."
            )
            dual = False

        # Heads-up if user chose liblinear for multi-class
        if self.solver == "liblinear":
            # liblinear doesn't support multinomial; sklearn will fall back to OvR
            # (this is fine; just make it explicit)
            # If y has > 2 classes, note the behavior:
            if np.unique(y).size > 2:
                self.logger.info(
                    "solver='liblinear' with multi-class data: scikit-learn will use One-vs-Rest."
                )

        Z = self.map_.transform(X)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model_ = LogisticRegression(
                penalty="l2",
                C=self.C,
                solver=self.solver,
                dual=dual,
                max_iter=self.max_iter,
                tol=self.tol,
                fit_intercept=True,
                random_state=self.random_state,
                verbose=self.verbose,
            ).fit(Z, y)

        self.classes_ = self.model_.classes_
        self.n_features_in_ = X.shape[1]

        # Original-space parameters
        self.W_, self.b_ = self.linear_params_()
        return self

    # ---------- API ----------

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        return self.model_.decision_function(self.map_.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        return self.model_.predict_proba(self.map_.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        return self.model_.predict(self.map_.transform(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    @property
    def weight_(self) -> np.ndarray:
        """W in original X-space; (D,) for binary, else (C,D)."""
        check_is_fitted(self, ["model_"])
        W = self.map_.coef_to_weight(self.model_.coef_)
        return W[0] if W.shape[0] == 1 else W

    def linear_params_(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (W, b) in original X-space (C,D) and (C,) (or (D,), scalar for binary)."""
        check_is_fitted(self, ["model_"])
        W = self.map_.coef_to_weight(self.model_.coef_)
        b = self.map_.intercept_in_raw_phi(self.model_.coef_, self.model_.intercept_)
        if W.shape[0] == 1:
            return W[0], float(b[0])
        return W, b


# =========================== Estimators: SVM (Graph-Sobolev) ==================


class GraphSobolevSVM(BaseEstimator, ClassifierMixin):
    """
    Linear SVM with a Graph/Laplacian-Sobolev prior implemented via spectral
    feature scaling. For calibrated probabilities, set probability=True.
    """

    def __init__(
        self,
        L: Optional[Union[np.ndarray, "SpMatrix"]] = None,
        A: Optional[Union[np.ndarray, "SpMatrix"]] = None,
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
        loss: Literal["hinge", "squared_hinge"] = "hinge",
        dual: bool = True,
        tol: float = 1e-4,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
        verbose: bool = False,
        probability: bool = False,
        prob_cv: int = 5,
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
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.probability = probability
        self.prob_cv = prob_cv
        self.eig_tol = eig_tol
        self.eig_maxiter = eig_maxiter

        self.logger = _make_logger(self.__class__.__name__, verbose)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GraphSobolevSVM":
        X, y = check_X_y(X, y, dtype=np.float64, ensure_min_samples=2)

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

        # LinearSVC requires dual=True for hinge loss
        dual = self.dual
        if self.loss == "hinge" and not dual:
            self.logger.warning(
                "LinearSVC requires dual=True when loss='hinge'; overriding dual=False→True."
            )
            dual = True

        Z = self.map_.transform(X)

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
            ).fit(Z, y)

        if self.probability:
            self.calibrator_ = CalibratedClassifierCV(
                estimator=self.base_svc_, cv=self.prob_cv, method="sigmoid"
            ).fit(Z, y)
            self.model_ = self.calibrator_
        else:
            self.model_ = self.base_svc_

        self.classes_ = self.base_svc_.classes_
        self.n_features_in_ = X.shape[1]
        self.W_, self.b_ = self.linear_params_()
        return self

    # ----- API -----

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["base_svc_"])
        # Raw margins from base SVM (independent of calibration)
        return self.base_svc_.decision_function(self.map_.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        return self.model_.predict(self.map_.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.probability:
            raise AttributeError(
                "predict_proba is only available when probability=True"
            )
        check_is_fitted(self, ["model_"])
        return self.model_.predict_proba(self.map_.transform(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    @property
    def weight_(self) -> np.ndarray:
        check_is_fitted(self, ["base_svc_"])
        W = self.map_.coef_to_weight(self.base_svc_.coef_)
        return W[0] if W.shape[0] == 1 else W

    def linear_params_(self) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["base_svc_"])
        W = self.map_.coef_to_weight(self.base_svc_.coef_)
        b = self.map_.intercept_in_raw_phi(
            self.base_svc_.coef_, self.base_svc_.intercept_
        )
        if W.shape[0] == 1:
            return W[0], float(b[0])
        return W, b


# ============================== Convenience alias =============================

GraphLaplacianLogistic = GraphSobolevLogisticRegression
GraphLaplacianSVM = GraphSobolevSVM


if __name__ == "__main__":
    """
    Final go/no-go test:
      - High-D (D=600), low-N (N=120) classification.
      - Ground truth depends ONLY on low Laplacian modes of a 1-D chain.
      - Feature variance INCREASES with frequency (hurts PCA/vanilla).
      - We compare:
          1) Baseline Logistic (z-scored), CV over C
          2) PCA+Logistic, CV over n_components & C
          3) GraphSobolevLogistic (chain Laplacian), CV over {n_components, s, mu, C}
      - Also print low-mode energy and cosine alignment to the true weight.
    """
    import numpy as np
    import warnings
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.metrics import accuracy_score

    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(7)

    # ----------------------------- Problem size --------------------------------
    D = 600  # features (large)
    N_train = 120  # samples (small)
    N_test = 2000  # large test set for stable measurement

    # -------------------------- Chain Laplacian eigens --------------------------
    # Use the *same* internal builder so we're testing exactly the estimator's prior.
    L = _build_chain_laplacian(D)
    Ld = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
    evals_all, V_all = np.linalg.eigh(Ld)  # sorted ascending by numpy
    order = np.argsort(evals_all)
    lam = evals_all[order]
    V = V_all[:, order]  # columns are eigenvectors
    # index 0 is the constant (nullspace) eigenvector for the chain Laplacian

    # ------------------------- True linear decision rule ------------------------
    # Build w_true from ONLY the first q smooth modes (skip the constant DC mode).
    q = 15
    idx_low = np.arange(1, q + 1)
    # Slight decay by 1/sqrt(mu+lambda) to bias toward smoother modes
    mu_true = 1e-3
    decay = 1.0 / np.sqrt(mu_true + lam[idx_low])
    coeff = rng.normal(size=q) * decay
    w_true = V[:, idx_low] @ coeff
    w_true /= np.linalg.norm(w_true)

    # ---------------------------- Data generation -------------------------------
    # Create samples by sampling in spectral coords with variance that *increases*
    # with frequency. This makes high-frequency components high-variance noise.
    gamma = 1.25
    sigma = 0.4 + 3.0 * (lam / lam.max()) ** gamma  # D variances (low->high)
    Ztr = rng.normal(size=(N_train, D)) * sigma  # (N,D) in spectral coords
    Zte = rng.normal(size=(N_test, D)) * sigma
    # Back to original feature space (X = Z @ V^T)
    Xtr = Ztr @ V.T
    Xte = Zte @ V.T

    # Logistic labels with scale chosen for moderate Bayes difficulty
    z_tr = Xtr @ w_true
    scale = 2.2 / (np.std(z_tr) + 1e-12)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    p_tr = sigmoid(scale * (Xtr @ w_true))
    p_te = sigmoid(scale * (Xte @ w_true))
    ytr = (rng.random(N_train) < p_tr).astype(int)
    yte = (rng.random(N_test) < p_te).astype(int)

    # ------------------------------ Baseline LR --------------------------------
    base_pipe = Pipeline(
        [
            ("std", StandardScaler(with_mean=True)),
            ("clf", LogisticRegression(max_iter=5000, random_state=0)),
        ]
    )
    base_grid = {"clf__C": [0.1, 0.3, 1, 3, 10]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    gs_base = GridSearchCV(base_pipe, base_grid, cv=cv, n_jobs=-1, scoring="accuracy")
    gs_base.fit(Xtr, ytr)
    base_acc = gs_base.score(Xte, yte)
    print(
        f"[Baseline Logistic]         test acc = {base_acc:.3f} | best C = {gs_base.best_params_['clf__C']}"
    )

    # ---------------------------- PCA(•)+Logistic -------------------------------
    pca_pipe = Pipeline(
        [
            ("std", StandardScaler(with_mean=True)),
            ("pca", PCA(random_state=0)),
            ("clf", LogisticRegression(max_iter=5000, random_state=0)),
        ]
    )
    pca_grid = {
        "pca__n_components": [10, 15, 20, 40],
        "clf__C": [0.1, 0.3, 1, 3, 10],
    }
    gs_pca = GridSearchCV(pca_pipe, pca_grid, cv=cv, n_jobs=-1, scoring="accuracy")
    gs_pca.fit(Xtr, ytr)
    pca_acc = gs_pca.score(Xte, yte)
    print(
        f"[PCA(+Logistic)]            test acc = {pca_acc:.3f} | best = {gs_pca.best_params_}"
    )

    # ----------------------- Sobolev (chain Laplacian) --------------------------
    # No graph A passed -> default 1-D chain Laplacian (over feature order).
    sobo_grid = {
        "n_components": [10, 15, 20, 30],
        "sobolev_s": [1.0, 1.5, 2.0],
        "sobolev_mu": [1e-4, 1e-3],
        "C": [0.3, 1, 3],
    }
    gs_sobo = GridSearchCV(
        GraphSobolevLogisticRegression(max_iter=1500, verbose=False),
        sobo_grid,
        cv=cv,
        n_jobs=-1,
        scoring="accuracy",
    )
    gs_sobo.fit(Xtr, ytr)
    sobo_acc = gs_sobo.score(Xte, yte)
    best_sobo = gs_sobo.best_estimator_
    print(
        f"[Sobolev(chain)]            test acc = {sobo_acc:.3f} | best = {gs_sobo.best_params_}"
    )

    # ------------------------ Diagnostics / interpretability --------------------
    # 1) How concentrated is the learned weight in the first K smooth modes?
    def low_mode_energy(model, K=20):
        beta = model.map_.unscale_coef(model.model_.coef_).ravel()  # spectral coeffs
        num = float(beta[:K] @ beta[:K])
        den = float(beta @ beta) + 1e-12
        return num / den

    e20 = low_mode_energy(best_sobo, K=20)

    # 2) Cosine alignment of learned original-space weight vs ground truth w_true
    W_hat, _ = best_sobo.linear_params_()
    cos = float(
        (W_hat @ w_true) / ((np.linalg.norm(W_hat) * np.linalg.norm(w_true)) + 1e-12)
    )

    print(
        f"[Sobolev(chain)] low-mode energy@20 = {e20:.3f} | cosine(W_hat, W_true) = {cos:.3f}"
    )

    # ------------------------------- Summary ------------------------------------
    print("\n=== FINAL SUMMARY ===")
    print(f"Baseline Logistic      : {base_acc:.3f}")
    print(f"PCA(+Logistic)         : {pca_acc:.3f}")
    print(f"Sobolev (chain)        : {sobo_acc:.3f}")
    print(f"Low-mode energy@20     : {e20:.3f}")
    print(f"Cosine alignment (Ŵ,W*): {cos:.3f}")
    print(
        "\nIf Sobolev ≥ baseline by ~+2–5 pts AND cosine alignment is high (e.g., >0.6),"
    )
    print("the structured prior is doing real work in a sample-starved, high-D regime.")
