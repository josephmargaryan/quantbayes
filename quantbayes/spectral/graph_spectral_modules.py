# =============================================================================
# graph_spectral_models.py
# Graph/Laplacian-Sobolev linear classifiers (Logistic & SVM)
#
# Key ideas:
#   - Build a feature map Phi = X @ V where V are eigenvectors of an SPD
#     operator L (e.g., a graph Laplacian over feature indices).
#   - Impose a diagonal quadratic penalty in this eigenbasis via column scaling:
#       alpha_j = (mu + lambda_j) ** s      [Sobolev weights]
#       z = (Phi - mean) / sqrt(alpha)      ['sobolev' scaling]
#     Training with standard L2 on coefficients becomes a structured L-penalty.
#   - Supports 'standardize' or custom 'weights' scaling instead.
#   - Works with dense or sparse L; falls back to a 1-D chain Laplacian if none
#     is supplied (smoothness over feature order).
#   - Multi-class safe; exports original-space (W, b).
#
# Dependencies: numpy, scikit-learn, (optional) scipy.sparse
# =============================================================================

import time
import logging
import warnings
from typing import Optional, Literal, Sequence, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

# SciPy is optional but strongly recommended for sparse graphs / large D
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
    """
    Unnormalized 1-D chain Laplacian on D features: L = D - A with path graph.
    """
    if not SCIPY_AVAILABLE:
        # Dense fallback (small D)
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
    """
    Symmetric normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}.
    A is assumed nonnegative, symmetric, with zero diagonal.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy is required to build normalized Laplacians.")
    deg = np.asarray(A.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-32))
    D_inv_sqrt = diags(d_inv_sqrt)
    I = diags(np.ones_like(deg))
    return I - D_inv_sqrt @ A @ D_inv_sqrt


def _dense_eigh_smallest(L: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dense symmetric eigendecomposition; returns k smallest eigenpairs.
    Only for modest D. Uses np.linalg.eigh then slices.
    """
    evals, evecs = np.linalg.eigh(L)
    idx = np.argsort(evals)[:k]
    return evals[idx], evecs[:, idx]


def _sparse_eigsh_smallest(
    L: "csr_matrix", k: int, tol: float = 1e-6, maxiter: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sparse symmetric eigendecomposition; returns k smallest eigenpairs.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy required for sparse eigensolvers.")
    # eigsh with which='SM' (smallest magnitude). L should be PSD.
    evals, evecs = eigsh(L, k=k, which="SM", tol=tol, maxiter=maxiter)
    idx = np.argsort(evals)
    return evals[idx], evecs[:, idx]


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

    Parameters
    ----------
    L : array-like (D,D) or sparse matrix, optional
        SPD operator (e.g., Laplacian). If None and A is None, uses a 1-D chain Laplacian.
    A : array-like (D,D) or sparse matrix, optional
        Symmetric adjacency to construct a Laplacian.
    laplacian_type : {'unnormalized','normalized'}, default='unnormalized'
        If A is provided (and L is None), which Laplacian to build.
    n_components : int or None
        Number of eigenvectors to keep (≤ D). If None, keeps all D.
        For Laplacians, the smallest eigenvectors encode smooth variations.
    drop_nullspace : bool, default=True
        Drop eigenvectors with eigenvalue < nullspace_tol (e.g., constant mode).
    nullspace_tol : float, default=1e-10
        Tolerance for detecting zero eigenvalues.
    feature_scaling : {'none','sobolev','standardize','weights'}
        Column scaling strategy.
    sobolev_s : float, default=1.0
        Exponent s in (mu + lambda)^s for 'sobolev'.
    sobolev_mu : float, default=1e-3
        Shift mu to avoid nullspace blow-up in 'sobolev'.
    penalty_weights : array-like, optional
        Per-coordinate α_j for 'weights' scaling (length = n_components after nullspace drop).
    eig_tol : float, default=1e-6
        Tolerance for sparse eigensolver.
    eig_maxiter : int or None
        Max iterations for sparse eigensolver (None lets ARPACK decide).
    random_state : int or None
        Only used for reproducible ordering if degeneracies arise (not critical).
    verbose : bool, default=False
        Log useful timings.

    Fitted attributes
    -----------------
    D_ : int
        Original feature dim.
    V_ : (D, k) ndarray
        Eigenvectors kept (orthonormal columns).
    evals_ : (k,) ndarray
        Corresponding eigenvalues.
    mean_ : (k,) ndarray
        Column means used in scaling.
    scale_ : (k,) ndarray
        Column scales used in scaling (sqrt of α_j for 'sobolev'/'weights', std for 'standardize').
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
        N, D = X.shape
        self.D_ = D

        # Build or accept L
        if self.L is None:
            if self.A is not None:
                A = _as_csr(self.A)
                if self.laplacian_type == "normalized":
                    L = _normalized_laplacian(A)
                else:
                    # L = D - A
                    deg = np.asarray(A.sum(axis=1)).ravel()
                    if not SCIPY_AVAILABLE:
                        L = np.diag(deg) - A.toarray()
                    else:
                        L = diags(deg) - A
                L_csr = _as_csr(L)
            else:
                # Default: 1-D chain Laplacian (smoothness over feature order)
                L_csr = _build_chain_laplacian(D)
        else:
            L_csr = _as_csr(self.L)

        # Choose k and compute k smallest eigenpairs
        k_full = D
        if self.drop_nullspace:
            # We may need a few extra to skip near-zeros; ask for slightly more.
            ask_k = min(D, (self.n_components or D) + 4)
        else:
            ask_k = self.n_components or D

        t0 = time.time()
        if SCIPY_AVAILABLE and (L_csr.count_nonzero() / (D * D) < 0.25 or D > 800):
            # sparse path
            evals, V = _sparse_eigsh_smallest(
                L_csr,
                k=max(1, min(ask_k, D)),
                tol=self.eig_tol,
                maxiter=self.eig_maxiter,
            )
        else:
            # dense path (small D)
            evals, V = _dense_eigh_smallest(
                L_csr.toarray() if SCIPY_AVAILABLE else np.asarray(L_csr),
                k=min(ask_k, D),
            )
        self.logger.info(
            f"Eigendecomposition: D={D}, k={V.shape[1]}, time={time.time()-t0:.3f}s"
        )

        # Drop nullspace (near-zero evals) if requested
        if self.drop_nullspace:
            keep = evals > self.nullspace_tol
            if not np.any(keep):
                # Keep at least the first
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

    Provide either:
      - L: SPD operator (dense or sparse), or
      - A: symmetric adjacency (we'll build L), or
      - nothing: uses a 1-D chain Laplacian over feature order.

    Parameters
    ----------
    L, A, laplacian_type, n_components, drop_nullspace, nullspace_tol :
        See _GraphSpectralFeatureMap.
    feature_scaling : {'sobolev','weights','standardize','none'}
        'sobolev' gives structured smoothness; 'weights' allows custom α_j.
    sobolev_s : float
        Exponent s in (mu + lambda)^s.
    sobolev_mu : float
        Shift mu to avoid nullspace pathology.
    penalty_weights : array-like or None
        α_j when feature_scaling='weights'.
    solver : str
        scikit-learn solver (e.g., 'lbfgs', 'liblinear', 'saga').
    dual : bool
        Only valid with 'liblinear' in scikit.
    C : float
        Inverse L2 strength (1/λ).
    max_iter, tol, multi_class, random_state, verbose :
        Standard logistic args.

    Attributes after fit
    --------------------
    model_ : sklearn.linear_model.LogisticRegression
    classes_ : ndarray
    W_ : (C, D) ndarray
        Original-space weights.
    b_ : (C,) ndarray
        Intercepts in original space.
    map_ : _GraphSpectralFeatureMap
        The fitted spectral map (eigs, basis, scales).
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
        solver: str = "lbfgs",
        dual: bool = False,
        C: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-4,
        multi_class: str = "auto",
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
        self.multi_class = multi_class
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
                multi_class=self.multi_class,
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

    Parameters
    ----------
    L, A, laplacian_type, n_components, drop_nullspace, nullspace_tol :
        See _GraphSpectralFeatureMap.
    feature_scaling : {'sobolev','weights','standardize','none'}
    sobolev_s, sobolev_mu, penalty_weights :
        As above.
    C, loss, dual, tol, max_iter, random_state, verbose :
        Standard LinearSVC args. Note: LinearSVC requires dual=True if loss='hinge'.
    probability : bool, default=False
        If True, wraps with CalibratedClassifierCV (sigmoid).
    prob_cv : int
        Folds for Platt scaling.

    Attributes after fit
    --------------------
    base_svc_ : LinearSVC
    model_ : LinearSVC or CalibratedClassifierCV
    classes_ : ndarray
    W_, b_ : original-space parameters
    map_ : fitted spectral map
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

# Backwards-compatible / shorter names if you like:
GraphLaplacianLogistic = GraphSobolevLogisticRegression
GraphLaplacianSVM = GraphSobolevSVM
