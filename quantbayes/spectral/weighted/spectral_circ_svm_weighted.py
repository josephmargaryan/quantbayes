#!/usr/bin/env python3
"""
SpectralCirculantWeightedSVM
----------------------------
Circulant‑parameterised linear SVM with per‑frequency ℓ₂ penalty β_j.
Enforces dual=True when loss='hinge'.
"""

import time
import logging
import warnings
from typing import Optional, Sequence, Literal

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin


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


class SpectralCirculantWeightedSVM(BaseEstimator, ClassifierMixin):
    """
    Circulant‑parameterised linear SVM with per‑frequency ℓ₂ penalty β_j.
    Optionally provides probability estimates via Platt scaling.

    Parameters
    ----------
    padded_dim : int or None
        FFT length (>= original feature dim). If None, uses the original dim.
    K : int or None
        Number of low frequencies to keep. If None, keeps all.
    beta : array-like of shape (K,) or (2*k_half,), or None
        Per-frequency ℓ₂ weights. If None, all weights are 1.
    C : float
        Inverse regularization strength.
    loss : {'hinge','squared_hinge'}
        Loss function to use.
    dual : bool
        Dual formulation. (Will be forced to True if loss='hinge'.)
    tol : float
        Tolerance for stopping criterion.
    max_iter : int
        Max number of iterations.
    random_state : int, RandomState or None
        Random seed.
    verbose : bool
        If True, prints solver info.
    probability : bool
        If True, enable probability estimates (adds calibration step).
    prob_cv : int
        Number of folds for probability calibration (ignored if probability=False).
    """

    def __init__(
        self,
        padded_dim: Optional[int] = None,
        K: Optional[int] = None,
        beta: Optional[Sequence[float]] = None,
        C: float = 1.0,
        loss: Literal["hinge", "squared_hinge"] = "hinge",
        dual: bool = False,
        tol: float = 1e-4,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
        verbose: bool = False,
        probability: bool = False,
        prob_cv: int = 5,
    ):
        # Store all init arguments (no logic here)
        self.padded_dim = padded_dim
        self.K = K
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

        # Logger (not a hyperparameter)
        self._log = _setup_logger(self.__class__.__name__, verbose)

    def _make_fft_features(self, X: np.ndarray):
        n, D = X.shape

        # Determine padded dimension
        pd = D if self.padded_dim is None else int(self.padded_dim)
        if pd < D:
            raise ValueError("padded_dim must be >= original feature dim")
        self.padded_dim_ = pd

        # Half-spectrum size
        self.k_half_ = pd // 2 + 1

        # Number of freqs to keep
        Kkeep = (
            self.k_half_ if (self.K is None or self.K > self.k_half_) else int(self.K)
        )
        self.K_ = Kkeep
        mask = np.arange(self.k_half_) < self.K_

        # Zero-pad, FFT, mask high freqs
        Xpad = np.zeros((n, pd), dtype=float)
        Xpad[:, :D] = X
        Xf = np.fft.rfft(Xpad, axis=1)
        Xf[:, ~mask] = 0.0
        Phi = np.hstack([Xf.real, Xf.imag])  # shape (n, 2*k_half_)

        # Build sqrt-beta
        M = 2 * self.k_half_
        if self.beta is None:
            sqrt_beta = np.ones(M, dtype=float)
        else:
            b = np.asarray(self.beta, dtype=float)
            beta_full = np.ones(M, dtype=float)
            if b.shape[0] == self.K_:
                beta_full[: self.K_] = b
                beta_full[self.k_half_ : self.k_half_ + self.K_] = b
            elif b.shape[0] == M:
                beta_full = b.copy()
            else:
                raise ValueError(
                    f"beta must have length {self.K_} or {M}, got {b.shape}"
                )
            sqrt_beta = np.sqrt(beta_full)

        return Phi / sqrt_beta[np.newaxis, :], sqrt_beta

    def fit(self, X, y):
        # Validate inputs
        X, y = check_X_y(X, y, dtype=float)
        self.n_features_in_ = X.shape[1]

        # Enforce dual=True if hinge loss
        dual = self.dual
        if self.loss == "hinge" and not dual:
            self._log.warning(
                "LinearSVC requires dual=True when loss='hinge'; overriding dual=False→True"
            )
            dual = True

        # Build & scale FFT features
        Phi_scaled, sqrt_beta = self._make_fft_features(X)
        self._sqrt_beta_ = sqrt_beta

        # Fit underlying LinearSVC
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            t0 = time.time()
            base_svc = LinearSVC(
                C=self.C,
                loss=self.loss,
                dual=dual,
                tol=self.tol,
                max_iter=self.max_iter,
                random_state=self.random_state,
                verbose=int(self.verbose),
            ).fit(Phi_scaled, y)
            self._log.info(f"SVM converged in {time.time() - t0:.3f}s")

        # If probabilities requested, calibrate
        if self.probability:
            self.calibrator_ = CalibratedClassifierCV(
                base_estimator=base_svc, cv=self.prob_cv, method="sigmoid"
            )
            self.calibrator_.fit(Phi_scaled, y)
            self.model_ = self.calibrator_
        else:
            self.model_ = base_svc

        # Recover Fourier‑space coefficients
        coef = base_svc.coef_.ravel() / sqrt_beta
        self.F_real_ = coef[: self.k_half_].copy()
        self.F_imag_ = coef[self.k_half_ :].copy()
        self.intercept_ = float(base_svc.intercept_[0])
        self.classes_ = base_svc.classes_

        return self

    def _transform(self, X: np.ndarray):
        check_is_fitted(self, ["model_", "_sqrt_beta_"])
        X = check_array(X, dtype=float)

        n, D = X.shape
        if D != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {D}")

        # Zero-pad, FFT, mask
        pd = self.padded_dim_
        Xpad = np.zeros((n, pd), dtype=float)
        Xpad[:, :D] = X
        Xf = np.fft.rfft(Xpad, axis=1)
        mask = np.arange(self.k_half_) < self.K_
        Xf[:, ~mask] = 0.0

        Phi = np.hstack([Xf.real, Xf.imag])
        return Phi / self._sqrt_beta_[np.newaxis, :]

    def decision_function(self, X):
        return self.model_.decision_function(self._transform(X))

    def predict_proba(self, X):
        if not self.probability:
            raise AttributeError(
                "predict_proba is only available when probability=True"
            )
        return self.model_.predict_proba(self._transform(X))

    def predict(self, X):
        return self.model_.predict(self._transform(X))

    def score(self, X, y):
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X))


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score, roc_auc_score

    # Synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=5,
        n_redundant=2,
        flip_y=0.01,
        class_sep=1.0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Vanilla SVM (hinge→dual=True, more iterations)
    vanilla = LinearSVC(
        C=1.0,
        loss="hinge",
        dual=True,
        tol=1e-4,
        max_iter=5000,
        random_state=42,
    )
    vanilla.fit(X_train, y_train)
    y_pred_v = vanilla.predict(X_test)
    scores_v = vanilla.decision_function(X_test)

    # Circulant weighted SVM
    K = 10
    beta = 1.0 + (np.linspace(0, 1, K) ** 2)
    circ_svm = SpectralCirculantWeightedSVM(
        padded_dim=None,
        K=None,
        beta=None,
        C=1.0,
        loss="hinge",
        dual=False,  # will be overridden to True
        tol=1e-4,
        max_iter=5000,
        random_state=42,
        verbose=False,
    )
    circ_svm.fit(X_train, y_train)
    y_pred_c = circ_svm.predict(X_test)
    scores_c = circ_svm.decision_function(X_test)

    acc_v = accuracy_score(y_test, y_pred_v)
    auc_v = roc_auc_score(y_test, scores_v)
    acc_c = accuracy_score(y_test, y_pred_c)
    auc_c = roc_auc_score(y_test, scores_c)

    print("Vanilla LinearSVC")
    print(f"  Accuracy: {acc_v:.3f},  ROC AUC: {auc_v:.3f}")
    print("SpectralCirculantWeightedSVM")
    print(f"  Accuracy: {acc_c:.3f},  ROC AUC: {auc_c:.3f}")
