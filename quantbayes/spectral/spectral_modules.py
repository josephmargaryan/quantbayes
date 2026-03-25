# =============================================================================
# Spectral / Fixed-basis linear classifiers with principled feature scaling
# - Orthonormal FFT
# - DC/Nyquist imaginary removal
# - Optional Sobolev-weighted L2 via feature scaling
# - Optional standardization of spectral features (with intercept correction)
# - Multi-class safe
# - Clean mapping back to original-space weights
# =============================================================================

import time
import logging
import warnings
from typing import Optional, Literal, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score


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

    Feature scaling convention:
      z = (phi - mean) / scale
    Training is on z. To recover original-space coefficients:
      beta_unscaled = beta_trained / scale
      intercept_phys = b_trained - (mean/scale) @ beta_trained
    """

    def __init__(
        self,
        padded_dim: Optional[int] = None,
        K: Optional[int] = None,
        *,
        feature_scaling: Literal["none", "sobolev", "standardize"] = "none",
        sobolev_s: float = 1.0,
        verbose: bool = False,
    ):
        self.padded_dim = padded_dim
        self.K = K
        self.feature_scaling = feature_scaling
        self.sobolev_s = float(sobolev_s)
        self.verbose = verbose
        self.logger = _make_logger(self.__class__.__name__, verbose)

    # ---- fitted attributes (set in fit) ----
    # D_               : original feature dim
    # padded_dim_      : FFT length
    # k_half_          : padded_dim_ // 2 + 1
    # K_               : number of kept frequency bins
    # mask_            : boolean, which bins kept
    # col_k_, col_is_real_ : column->(frequency k, is_real)
    # n_cols_          : number of real columns in Phi
    # mean_, scale_    : feature scaling
    # (for logging only) train_build_sec_

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

        self.mask_ = np.zeros(self.k_half_, dtype=bool)
        self.mask_[: self.K_] = True

        nyquist_bin = self.k_half_ - 1 if (pd % 2 == 0) else None  # exists only if even

        # Column layout in Phi: for each kept k: Re; and Imag unless k in {0, nyquist}
        col_k = []
        col_is_real = []
        for k in range(self.k_half_):
            if not self.mask_[k]:
                continue
            # Real part always present
            col_k.append(k)
            col_is_real.append(True)
            # Imag part present except for DC (k=0) and Nyquist (if even and masked)
            if k != 0 and not (nyquist_bin is not None and k == nyquist_bin):
                col_k.append(k)
                col_is_real.append(False)
        self.col_k_ = np.asarray(col_k, dtype=int)
        self.col_is_real_ = np.asarray(col_is_real, dtype=bool)
        self.n_cols_ = self.col_k_.shape[0]

        t0 = time.time()
        # Prepare scaling
        if self.feature_scaling == "sobolev":
            # alpha_k = (1 + omega_k^2)^s, omega_k = 2π k / padded_dim
            omega = (
                2.0 * np.pi * self.col_k_.astype(np.float64) / float(self.padded_dim_)
            )
            alpha = (1.0 + omega**2) ** self.sobolev_s
            self.mean_ = np.zeros(self.n_cols_, dtype=np.float64)
            self.scale_ = np.sqrt(alpha)  # z = phi / sqrt(alpha)  -> uniform L2 on beta
        elif self.feature_scaling == "standardize":
            phi = self._phi_raw(X)  # (n, n_cols_)
            self.mean_ = phi.mean(axis=0)
            sd = phi.std(axis=0)
            self.scale_ = np.where(sd > 1e-12, sd, 1.0)
        else:  # 'none'
            self.mean_ = np.zeros(self.n_cols_, dtype=np.float64)
            self.scale_ = np.ones(self.n_cols_, dtype=np.float64)

        self.train_build_sec_ = time.time() - t0
        self.logger.info(
            f"Fourier map: pd={self.padded_dim_}, k_half={self.k_half_}, K={self.K_}, "
            f"Phi_cols={self.n_cols_}, built in {self.train_build_sec_:.3f}s"
        )
        return self

    def _phi_raw(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, dtype=np.float64)
        n, D = X.shape
        if D != self.D_:
            raise ValueError(f"Expected {self.D_} features, got {D}")
        X_pad = np.zeros((n, self.padded_dim_), dtype=np.float64)
        X_pad[:, :D] = X
        Xf = np.fft.rfft(X_pad, axis=1, norm="ortho")  # shape (n, k_half_)
        # Build real design with our column layout
        Phi = np.empty((n, self.n_cols_), dtype=np.float64)
        for j, (k, is_real) in enumerate(zip(self.col_k_, self.col_is_real_)):
            Phi[:, j] = Xf[:, k].real if is_real else Xf[:, k].imag
        return Phi

    def transform(self, X: np.ndarray) -> np.ndarray:
        phi = self._phi_raw(X)
        return (phi - self.mean_) / self.scale_

    # ---------- mappings back to interpretable params ----------

    def unscale_coef(self, coef_mat: np.ndarray) -> np.ndarray:
        """coef trained on z -> coef in raw-phi coordinates."""
        return coef_mat / self.scale_[None, :]

    def intercept_in_raw_phi(
        self, coef_mat: np.ndarray, intercept_vec: np.ndarray
    ) -> np.ndarray:
        """b in raw-phi domain (undo centering). Shape (n_classes,)."""
        if self.mean_ is None or self.scale_ is None:
            return intercept_vec
        shift = (self.mean_ / self.scale_) @ coef_mat.T  # (n_classes,)
        return intercept_vec - shift

    def coef_to_weight(self, coef_mat: np.ndarray) -> np.ndarray:
        """
        Map coef over Phi back to original-space linear weights w.
        Handles multi-class: returns (C, D).
        """
        coef_raw = self.unscale_coef(coef_mat)  # (C, n_cols)
        C = coef_raw.shape[0]
        F_real = np.zeros((C, self.k_half_), dtype=np.float64)
        F_imag = np.zeros((C, self.k_half_), dtype=np.float64)
        for j, (k, is_real) in enumerate(zip(self.col_k_, self.col_is_real_)):
            if is_real:
                F_real[:, k] = coef_raw[:, j]
            else:
                F_imag[:, k] = coef_raw[:, j]
        Ff = F_real + 1j * F_imag
        w_full = np.fft.irfft(Ff, n=self.padded_dim_, norm="ortho")  # (C, pd)
        return w_full[:, : self.D_]  # (C, D)


# --------------------- Fixed-basis (orthonormal) map --------------------------


class _FixedBasisFeatureMap:
    """
    Phi = X @ V  with V having orthonormal columns (random+QR or PCA).
    feature_scaling: 'none' | 'standardize' | 'weights' (use provided penalty_weights).
    """

    def __init__(
        self,
        n_spectral: Optional[int] = None,
        basis: Literal["random", "pca"] = "random",
        *,
        feature_scaling: Literal["none", "standardize", "weights"] = "none",
        penalty_weights: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_spectral = n_spectral
        self.basis = basis
        self.feature_scaling = feature_scaling
        self.penalty_weights = penalty_weights
        self.random_state = random_state
        self.verbose = verbose
        self.logger = _make_logger(self.__class__.__name__, verbose)

    # fitted:
    # D_, n_spectral_, V_, mean_, scale_

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
            Q, _ = np.linalg.qr(G)  # orthonormal columns
            self.V_ = Q[:, :k]
        elif self.basis == "pca":
            pca = PCA(n_components=k, random_state=self.random_state)
            pca.fit(X)
            self.V_ = pca.components_.T  # (D, k)
        else:
            raise ValueError(f"Unknown basis='{self.basis}'")
        self.logger.info(
            f"Built basis '{self.basis}' with shape V={self.V_.shape} in {time.time()-t0:.3f}s"
        )

        Phi = X @ self.V_  # (N, k)

        if self.feature_scaling == "weights":
            if self.penalty_weights is None:
                raise ValueError(
                    "feature_scaling='weights' requires penalty_weights (length k)."
                )
            pw = np.asarray(self.penalty_weights, dtype=np.float64)
            if pw.shape[0] != k:
                raise ValueError(
                    f"penalty_weights length must be {k}, got {pw.shape[0]}."
                )
            self.mean_ = np.zeros(k, dtype=np.float64)
            self.scale_ = np.sqrt(pw)  # z = Phi / sqrt(pw)
        elif self.feature_scaling == "standardize":
            self.mean_ = Phi.mean(axis=0)
            sd = Phi.std(axis=0)
            self.scale_ = np.where(sd > 1e-12, sd, 1.0)
        else:  # 'none'
            self.mean_ = np.zeros(k, dtype=np.float64)
            self.scale_ = np.ones(k, dtype=np.float64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, dtype=np.float64)
        return (X @ self.V_ - self.mean_) / self.scale_

    def unscale_coef(self, coef_mat: np.ndarray) -> np.ndarray:
        return coef_mat / self.scale_[None, :]  # (C, k)

    def intercept_in_raw_phi(
        self, coef_mat: np.ndarray, intercept_vec: np.ndarray
    ) -> np.ndarray:
        shift = (self.mean_ / self.scale_) @ coef_mat.T
        return intercept_vec - shift

    def coef_to_weight(self, coef_mat: np.ndarray) -> np.ndarray:
        beta = self.unscale_coef(coef_mat)  # (C, k)
        W = beta @ self.V_.T  # (C, D)
        return W


# ======================= Estimators: Logistic variants ========================


class SpectralCirculantLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic regression with Fourier-parameterized weights:
      w = irfft(F)[:D], with low-pass truncation to K bins.
    Clean convex program with optional Sobolev-weighted L2 via feature scaling.

    Parameters
    ----------
    padded_dim : int or None
        FFT length (>= D). If None, uses D.
    K : int or None
        Number of low frequencies to keep (<= padded_dim//2+1). If None, keep all.
    feature_scaling : {"none","sobolev","standardize"}
        'sobolev' applies (1+omega^2)^s weighting via column scaling; 'standardize' z-scores Phi.
    sobolev_s : float
        Smoothness exponent s for 'sobolev'.
    solver, dual, C, max_iter, tol, multi_class, random_state, verbose
        Passed to sklearn.linear_model.LogisticRegression (with safety checks).
    """

    def __init__(
        self,
        padded_dim: Optional[int] = None,
        K: Optional[int] = None,
        *,
        feature_scaling: Literal["none", "sobolev", "standardize"] = "none",
        sobolev_s: float = 1.0,
        solver: str = "lbfgs",
        dual: bool = False,
        C: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.padded_dim = padded_dim
        self.K = K
        self.feature_scaling = feature_scaling
        self.sobolev_s = sobolev_s
        self.solver = solver
        self.dual = dual
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        self.logger = _make_logger(self.__class__.__name__, verbose)

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "SpectralCirculantLogisticRegression":
        X, y = check_X_y(X, y, dtype=np.float64, ensure_min_samples=2)
        self._map = _FourierFeatureMap1D(
            padded_dim=self.padded_dim,
            K=self.K,
            feature_scaling=self.feature_scaling,
            sobolev_s=self.sobolev_s,
            verbose=self.verbose,
        ).fit(X)

        # Safety: dual only valid with liblinear in scikit
        dual = self.dual
        if dual and self.solver != "liblinear":
            self.logger.warning(
                "dual=True is only valid with solver='liblinear'; overriding to dual=False."
            )
            dual = False

        Phi = self._map.transform(X)
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
            ).fit(Phi, y)

        self.classes_ = self.model_.classes_
        self.n_features_in_ = X.shape[1]

        # Store interpretable spectral params (per class)
        self.F_real_, self.F_imag_ = self._coef_to_F(self.model_.coef_)
        # Physical-space linear params
        self.W_, self.b_ = self.linear_params_()
        return self

    def _coef_to_F(self, coef_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # build per-class F_real, F_imag (K_half length), unscaled
        coef_raw = self._map.unscale_coef(coef_mat)
        C = coef_raw.shape[0]
        F_real = np.zeros((C, self._map.k_half_), dtype=np.float64)
        F_imag = np.zeros((C, self._map.k_half_), dtype=np.float64)
        for j, (k, is_real) in enumerate(zip(self._map.col_k_, self._map.col_is_real_)):
            if is_real:
                F_real[:, k] = coef_raw[:, j]
            else:
                F_imag[:, k] = coef_raw[:, j]
        return F_real, F_imag

    # ---------- API ----------

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        return self.model_.decision_function(self._map.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        return self.model_.predict_proba(self._map.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        return self.model_.predict(self._map.transform(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    @property
    def weight_(self) -> np.ndarray:
        """Return original-space weights W; shape (C, D) or (D,) if binary."""
        check_is_fitted(self, ["model_"])
        W = self._map.coef_to_weight(self.model_.coef_)
        return W[0] if W.shape[0] == 1 else W

    def linear_params_(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (W, b) in original X-space such that decision(X) ≈ X @ W.T + b.
        W: (C, D), b: (C,)  (or (D,), scalar for binary)
        """
        check_is_fitted(self, ["model_"])
        W = self._map.coef_to_weight(self.model_.coef_)
        b = self._map.intercept_in_raw_phi(self.model_.coef_, self.model_.intercept_)
        if W.shape[0] == 1:
            return W[0], float(b[0])
        return W, b


class SpectralLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic regression on a fixed orthonormal basis Phi = X @ V.
    Optional 'weights' scaling lets you implement arbitrary quadratic penalties.

    Parameters
    ----------
    n_spectral : int or None
    basis : {"random","pca"}
    feature_scaling : {"none","standardize","weights"}
        'weights' requires penalty_weights (length n_spectral).
    penalty_weights : array-like or None
        Per-coordinate penalty α_j -> implemented as z = Phi / sqrt(α_j).
    Other args forwarded to LogisticRegression.
    """

    def __init__(
        self,
        n_spectral: Optional[int] = None,
        basis: Literal["random", "pca"] = "random",
        *,
        feature_scaling: Literal["none", "standardize", "weights"] = "none",
        penalty_weights: Optional[np.ndarray] = None,
        solver: str = "lbfgs",
        dual: bool = False,
        C: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_spectral = n_spectral
        self.basis = basis
        self.feature_scaling = feature_scaling
        self.penalty_weights = penalty_weights
        self.solver = solver
        self.dual = dual
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.logger = _make_logger(self.__class__.__name__, verbose)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralLogisticRegression":
        X, y = check_X_y(X, y, dtype=np.float64, ensure_min_samples=2)
        self._map = _FixedBasisFeatureMap(
            n_spectral=self.n_spectral,
            basis=self.basis,
            feature_scaling=self.feature_scaling,
            penalty_weights=self.penalty_weights,
            random_state=self.random_state,
            verbose=self.verbose,
        ).fit(X)

        dual = self.dual
        if dual and self.solver != "liblinear":
            self.logger.warning(
                "dual=True is only valid with solver='liblinear'; overriding to dual=False."
            )
            dual = False

        Phi = self._map.transform(X)
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
            ).fit(Phi, y)

        self.classes_ = self.model_.classes_
        self.n_features_in_ = X.shape[1]
        self.W_, self.b_ = self.linear_params_()
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        return self.model_.decision_function(self._map.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        return self.model_.predict_proba(self._map.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        return self.model_.predict(self._map.transform(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    @property
    def weight_(self) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        W = self._map.coef_to_weight(self.model_.coef_)
        return W[0] if W.shape[0] == 1 else W

    def linear_params_(self) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["model_"])
        W = self._map.coef_to_weight(self.model_.coef_)
        b = self._map.intercept_in_raw_phi(self.model_.coef_, self.model_.intercept_)
        if W.shape[0] == 1:
            return W[0], float(b[0])
        return W, b


# =========================== Estimators: SVM variants =========================


class SpectralCirculantSVM(BaseEstimator, ClassifierMixin):
    """
    Linear SVM on truncated RFFT features (orthonormal), with optional Sobolev scaling.
    For calibrated probabilities, set probability=True (Platt scaling).
    """

    def __init__(
        self,
        padded_dim: Optional[int] = None,
        K: Optional[int] = None,
        *,
        feature_scaling: Literal["none", "sobolev", "standardize"] = "none",
        sobolev_s: float = 1.0,
        C: float = 1.0,
        loss: Literal["hinge", "squared_hinge"] = "hinge",
        dual: bool = True,
        tol: float = 1e-4,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
        verbose: bool = False,
        probability: bool = False,
        prob_cv: int = 5,
    ):
        self.padded_dim = padded_dim
        self.K = K
        self.feature_scaling = feature_scaling
        self.sobolev_s = sobolev_s
        self.C = C
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.probability = probability
        self.prob_cv = prob_cv

        self.logger = _make_logger(self.__class__.__name__, verbose)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralCirculantSVM":
        X, y = check_X_y(X, y, dtype=np.float64)
        self._map = _FourierFeatureMap1D(
            padded_dim=self.padded_dim,
            K=self.K,
            feature_scaling=self.feature_scaling,
            sobolev_s=self.sobolev_s,
            verbose=self.verbose,
        ).fit(X)

        # LinearSVC requires dual=True for loss='hinge'
        dual = self.dual
        if self.loss == "hinge" and not dual:
            self.logger.warning(
                "LinearSVC requires dual=True with loss='hinge'; overriding dual=False→True."
            )
            dual = True

        Phi = self._map.transform(X)

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
            ).fit(Phi, y)

        if self.probability:
            self.calibrator_ = CalibratedClassifierCV(
                estimator=self.base_svc_, cv=self.prob_cv, method="sigmoid"
            ).fit(Phi, y)
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
        # Return *raw* margins from the base SVM (independent of calibration)
        return self.base_svc_.decision_function(self._map.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        return self.model_.predict(self._map.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.probability:
            raise AttributeError(
                "predict_proba is only available when probability=True"
            )
        check_is_fitted(self, ["model_"])
        return self.model_.predict_proba(self._map.transform(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    @property
    def weight_(self) -> np.ndarray:
        check_is_fitted(self, ["base_svc_"])
        W = self._map.coef_to_weight(self.base_svc_.coef_)
        return W[0] if W.shape[0] == 1 else W

    def linear_params_(self) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["base_svc_"])
        W = self._map.coef_to_weight(self.base_svc_.coef_)
        b = self._map.intercept_in_raw_phi(
            self.base_svc_.coef_, self.base_svc_.intercept_
        )
        if W.shape[0] == 1:
            return W[0], float(b[0])
        return W, b


class SpectralSVM(BaseEstimator, ClassifierMixin):
    """
    Linear SVM on fixed orthonormal basis Phi = X @ V.
    'weights' scaling allows arbitrary diagonal penalties in the spectral coords.
    """

    def __init__(
        self,
        n_spectral: Optional[int] = None,
        basis: Literal["random", "pca"] = "random",
        *,
        feature_scaling: Literal["none", "standardize", "weights"] = "none",
        penalty_weights: Optional[np.ndarray] = None,
        C: float = 1.0,
        loss: Literal["hinge", "squared_hinge"] = "hinge",
        dual: bool = True,
        tol: float = 1e-4,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
        verbose: bool = False,
        probability: bool = False,
        prob_cv: int = 5,
    ):
        self.n_spectral = n_spectral
        self.basis = basis
        self.feature_scaling = feature_scaling
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

        self.logger = _make_logger(self.__class__.__name__, verbose)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralSVM":
        X, y = check_X_y(X, y, dtype=np.float64, ensure_min_samples=2)
        self._map = _FixedBasisFeatureMap(
            n_spectral=self.n_spectral,
            basis=self.basis,
            feature_scaling=self.feature_scaling,
            penalty_weights=self.penalty_weights,
            random_state=self.random_state,
            verbose=self.verbose,
        ).fit(X)

        dual = self.dual
        if self.loss == "hinge" and not dual:
            self.logger.warning(
                "LinearSVC requires dual=True when loss='hinge'; overriding dual=False→True."
            )
            dual = True

        Phi = self._map.transform(X)
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
            ).fit(Phi, y)

        if self.probability:
            self.calibrator_ = CalibratedClassifierCV(
                estimator=self.base_svc_, cv=self.prob_cv, method="sigmoid"
            ).fit(Phi, y)
            self.model_ = self.calibrator_
        else:
            self.model_ = self.base_svc_

        self.classes_ = self.base_svc_.classes_
        self.n_features_in_ = X.shape[1]
        self.W_, self.b_ = self.linear_params_()
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["base_svc_"])
        return self.base_svc_.decision_function(self._map.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        return self.model_.predict(self._map.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.probability:
            raise AttributeError(
                "predict_proba is only available when probability=True"
            )
        check_is_fitted(self, ["model_"])
        return self.model_.predict_proba(self._map.transform(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    @property
    def weight_(self) -> np.ndarray:
        check_is_fitted(self, ["base_svc_"])
        W = self._map.coef_to_weight(self.base_svc_.coef_)
        return W[0] if W.shape[0] == 1 else W

    def linear_params_(self) -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["base_svc_"])
        W = self._map.coef_to_weight(self.base_svc_.coef_)
        b = self._map.intercept_in_raw_phi(
            self.base_svc_.coef_, self.base_svc_.intercept_
        )
        if W.shape[0] == 1:
            return W[0], float(b[0])
        return W, b
