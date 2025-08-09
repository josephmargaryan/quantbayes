# strongly_convex_eval.py
# One-stop evaluation for strongly-convex (linear-on-fixed-features) models.
# - Works with your spectral scikit-learn-style estimators (LogReg/SVM variants)
# - Also runs baselines: LogisticRegression, LinearSVC (+ optional calibration)
# - Computes:
#     * Accuracy (train/test)
#     * ECE (if calibrated probabilities are available)
#     * Lipschitz of logits map: ||W||_2
#     * Frobenius of logits map: ||W||_F
#     * Train-set γ-margin loss L_hat_γ
#     * PAC-Bayes-style bound specialized to linear models
#     * Optional Sobolev quadratic (for your Fourier map if available)
#
# Usage examples are at the bottom under __main__.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Literal, Tuple, List
import time
import json
import math
import logging
import warnings
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import ConvergenceWarning

# ------- import your spectral estimators (adjust path if needed) -------
# from your_module import (
#   SpectralCirculantLogisticRegression, SpectralLogisticRegression,
#   SpectralCirculantSVM, SpectralSVM
# )
# For this script, we assume they are importable; uncomment and set the path.

from .spectral_modules import (  # rename this to your actual module path
        SpectralCirculantLogisticRegression,
        SpectralLogisticRegression,
        SpectralCirculantSVM,
        SpectralSVM,
    )


# ----------------------------- Logging ---------------------------------

def get_logger(name: str = "strongly_convex_eval", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
        logger.addHandler(h)
    return logger

LOGGER = get_logger()

# ------------------------- Dataset utilities ---------------------------

def load_openml_or_digits(
    name: Literal["mnist_784", "Fashion-MNIST", "digits"] = "digits",
    test_size: float = 0.2,
    random_state: int = 0,
    as_float64: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Try to fetch MNIST/Fashion-MNIST from OpenML. If no internet, fallback to sklearn 'digits'.
    Returns X_train, X_test, y_train, y_test with X in float64 and flattened.
    """
    if name == "digits":
        from sklearn.datasets import load_digits
        d = load_digits()
        X = d.data.astype(np.float64 if as_float64 else np.float32)
        y = d.target
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    try:
        from sklearn.datasets import fetch_openml
        if name == "mnist_784":
            ds = fetch_openml("mnist_784", version=1, as_frame=False, parser="pandas")
        elif name == "Fashion-MNIST":
            ds = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="pandas")
        else:
            raise ValueError("Unknown dataset name")
        X = ds.data.astype(np.float64 if as_float64 else np.float32)
        y = ds.target.astype(int) if ds.target.dtype.kind in "OUS" else ds.target
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    except Exception as e:
        LOGGER.warning(f"OpenML fetch failed ({e}). Falling back to sklearn 'digits'.")
        return load_openml_or_digits("digits", test_size, random_state, as_float64)

def load_sklearn_uci(
    which: Literal["breast_cancer", "wine", "iris"] = "breast_cancer",
    test_size: float = 0.2,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Small UCI-style datasets available offline via sklearn.
    """
    if which == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
    elif which == "wine":
        from sklearn.datasets import load_wine
        d = load_wine()
    elif which == "iris":
        from sklearn.datasets import load_iris
        d = load_iris()
    else:
        raise ValueError("Unknown UCI dataset")
    X = d.data.astype(np.float64)
    y = d.target
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

# --------------------- Metrics & theory utilities ----------------------

def spectral_norm(W: np.ndarray) -> float:
    """Largest singular value of W (2-norm)."""
    if W.ndim == 1:  # (D,)
        return float(np.linalg.norm(W, 2))
    # Use svd on smaller dimension for stability
    try:
        s = np.linalg.svd(W, compute_uv=False)
        return float(s[0])
    except np.linalg.LinAlgError:
        # fallback power iteration
        v = np.random.randn(W.shape[1])
        v /= np.linalg.norm(v) + 1e-12
        for _ in range(50):
            u = W @ v
            u_norm = np.linalg.norm(u) + 1e-12
            v = W.T @ (u / u_norm)
            v /= np.linalg.norm(v) + 1e-12
        return float(np.linalg.norm(W @ v))

def frobenius_norm(W: np.ndarray) -> float:
    return float(np.linalg.norm(W, "fro"))

def logits_from_estimator(est: Any, X: np.ndarray) -> np.ndarray:
    """
    Return logits (decision scores) as shape (N,K).
    For binary models that return (N,), reshape to (N,1) for margin handling.
    """
    if hasattr(est, "decision_function"):
        z = est.decision_function(X)
    else:
        # As a fallback, map log-probs to logits via inverse link if available.
        if hasattr(est, "predict_proba"):
            P = est.predict_proba(X)
            # For binary, logit(p) for class 1
            if P.ndim == 2 and P.shape[1] == 2:
                z = np.log((P[:, 1] + 1e-12) / (P[:, 0] + 1e-12))
            else:
                z = np.log(P + 1e-12)
        else:
            raise ValueError("Estimator must have decision_function or predict_proba.")
    z = np.asarray(z)
    if z.ndim == 1:
        z = z[:, None]
    return z

def margins_multiclass(logits: np.ndarray, y: np.ndarray, classes: Optional[np.ndarray] = None) -> np.ndarray:
    """
    For K>1: margin = z_y - max_{j != y} z_j.
    For K=1 (binary): interpret provided y; map to +/-1 using classes if given; margin = y' * z.
    """
    z = np.asarray(logits)
    if z.ndim == 1:
        z = z[:, None]
    N, K = z.shape
    y = np.asarray(y)
    if K == 1:
        # Binary: use classes ordering if provided (scikit convention: decision_function is for classes_[1])
        if classes is None:
            # assume y in {0,1}: +1 for 1, -1 for 0
            y_signed = np.where(y == 1, 1.0, -1.0)
        else:
            # +1 for classes_[1], -1 for classes_[0]
            assert len(classes) == 2
            y_signed = np.where(y == classes[1], 1.0, -1.0)
        return y_signed * z[:, 0]
    else:
        # Multiclass
        row = np.arange(N)
        # Map y to column indices if classes provided
        if classes is not None:
            # classes is array of class labels aligned with logits columns
            # Find index of each y in classes
            y_idx = np.searchsorted(classes, y) if np.all(np.diff(classes) >= 0) else np.array([np.where(classes == yi)[0][0] for yi in y])
        else:
            y_idx = y.astype(int)
        correct = z[row, y_idx]
        z_masked = z.copy()
        z_masked[row, y_idx] = -np.inf
        other = np.max(z_masked, axis=1)
        return correct - other

def empirical_margin_loss(margins: np.ndarray, gamma: float) -> float:
    """L̂_γ = mean[ margin <= gamma ]."""
    return float(np.mean(margins <= gamma))

def compute_radius_B(X: np.ndarray) -> float:
    """Data radius B = max_i ||x_i||_2."""
    return float(np.max(np.linalg.norm(X, axis=1)))

def pac_bayes_bound_linear(
    W: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    logits_train: np.ndarray,
    *,
    gamma: float = 1.0,
    delta: float = 0.05,
) -> Dict[str, float]:
    """
    Specialization of the Neyshabur-style bound to a single linear layer:
      L_01(f) ≤ min{ 1,  L̂_γ + sqrt( (B^2 * ||W||_F^2 + ln(m/δ)) / (γ^2 m) ) }.
    (Depth term and spectral product collapse to 1 for linear.)
    """
    m = X_train.shape[0]
    B = compute_radius_B(X_train)
    frob = frobenius_norm(W)
    margins = margins_multiclass(logits_train, y_train)
    Lhat = empirical_margin_loss(margins, gamma)
    penalty = math.sqrt(max(B * B * (frob ** 2) + math.log(max(m, 2) / max(delta, 1e-12)), 0.0) / max(gamma * gamma * m, 1e-12))
    bound = min(1.0, Lhat + penalty)
    return {
        "B": B,
        "frobenius": frob,
        "Lhat_gamma": Lhat,
        "penalty": penalty,
        "bound": bound,
    }

def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> Optional[float]:
    """
    ECE for multiclass: bin by max prob; gap between accuracy and confidence.
    If probabilities are not provided, return None.
    """
    if proba is None:
        return None
    P = np.asarray(proba)
    if P.ndim == 1:
        P = np.stack([1 - P, P], axis=1)
    conf = P.max(axis=1)
    preds = P.argmax(axis=1)
    y_true = np.asarray(y_true)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if np.any(mask):
            acc_bin = np.mean(preds[mask] == y_true[mask])
            conf_bin = np.mean(conf[mask])
            w = np.mean(mask)
            ece += w * abs(acc_bin - conf_bin)
    return float(ece)

# -------------------- Baselines with fold-back W,b ---------------------

@dataclass
class LinearWB:
    W: np.ndarray  # (K,D) or (D,)
    b: np.ndarray  # (K,) or scalar

def fold_back_scaler(clf: ClassifierMixin, scaler: StandardScaler) -> LinearWB:
    """
    Given a trained classifier on standardized features z=(x-mu)/sigma,
    compute equivalent logits w.r.t original x: W_x, b_x.
    """
    check_is_fitted(clf)
    Wz = clf.coef_  # (K,D) or (1,D)
    bz = clf.intercept_  # (K,) or (1,)
    sigma = scaler.scale_
    mu = scaler.mean_
    Wx = Wz / sigma[None, :]
    bx = bz - (mu / sigma) @ Wz.T
    Wx = Wx[0] if Wx.shape[0] == 1 else Wx
    bx = float(bx[0]) if bx.shape == (1,) else bx
    return LinearWB(W=Wx, b=bx)

def train_logreg_baseline(
    X_train: np.ndarray, y_train: np.ndarray, *,
    C: float = 1.0, max_iter: int = 200, tol: float = 1e-4, random_state: int = 0
) -> Tuple[Pipeline, LinearWB, float]:
    """
    StandardScaler + LogisticRegression (lbfgs), return pipeline, (W,b) in original space, and train seconds.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2", C=C, solver="lbfgs", max_iter=max_iter, tol=tol,
            fit_intercept=True, multi_class="auto", random_state=random_state))
    ])
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        pipe.fit(X_train, y_train)
    secs = time.time() - t0
    Wb = fold_back_scaler(pipe.named_steps["clf"], pipe.named_steps["scaler"])
    return pipe, Wb, secs

def train_linearsvc_baseline(
    X_train: np.ndarray, y_train: np.ndarray, *,
    C: float = 1.0, loss: Literal["hinge", "squared_hinge"] = "hinge",
    dual: Optional[bool] = None, max_iter: int = 2000, tol: float = 1e-4,
    random_state: int = 0, probability: bool = False, prob_cv: int = 5
) -> Tuple[Any, LinearWB, float]:
    """
    StandardScaler + LinearSVC; optional Platt calibration for probabilities.
    Returns model (Pipeline or CalibratedClassifierCV), (W,b) in original space, and train seconds.
    """
    if dual is None:
        dual = (loss == "hinge")
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(C=C, loss=loss, dual=dual, max_iter=max_iter, tol=tol, random_state=random_state))
    ])
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        base.fit(X_train, y_train)
    model = base
    if probability:
        model = CalibratedClassifierCV(estimator=base, cv=prob_cv, method="sigmoid")
        model.fit(X_train, y_train)
    secs = time.time() - t0
    Wb = fold_back_scaler(base.named_steps["svm"], base.named_steps["scaler"])
    return model, Wb, secs

# -------------------- Spectral estimator helpers ----------------------

def get_linear_params(est: Any) -> LinearWB:
    """
    Extract (W,b) in original X-space.
    - If estimator defines linear_params_(): use it.
    - Else if scikit classifier with coef_/intercept_ and has internal scaler: not handled here.
    """
    if hasattr(est, "linear_params_"):
        W, b = est.linear_params_()
        return LinearWB(W=W, b=b)
    if hasattr(est, "coef_") and hasattr(est, "intercept_"):
        # assume already in original space (your spectral models expose original mapping)
        W = est.coef_
        b = est.intercept_
        W = W[0] if W.ndim == 2 and W.shape[0] == 1 else W
        b = float(b[0]) if np.ndim(b) == 1 and b.shape == (1,) else b
        return LinearWB(W=W, b=b)
    raise ValueError("Estimator does not expose linear_params_ or coef_/intercept_.")

def maybe_ece(est: Any, X: np.ndarray, y: np.ndarray, n_bins: int = 15) -> Optional[float]:
    if hasattr(est, "predict_proba"):
        try:
            P = est.predict_proba(X)
            return expected_calibration_error(y, P, n_bins=n_bins)
        except Exception:
            return None
    return None

# -------------------------- Evaluation loop ---------------------------

@dataclass
class EvalResult:
    name: str
    train_acc: float
    test_acc: float
    ece_test: Optional[float]
    lipschitz_sigma: float
    frobenius: float
    gamma: float
    Lhat_gamma: float
    pacbayes_penalty: float
    pacbayes_bound: float
    train_time_sec: float
    extras: Dict[str, Any]

def evaluate_estimator(
    name: str,
    est: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    gamma: float = 1.0,
    delta: float = 0.05,
    extras: Optional[Dict[str, Any]] = None,
) -> EvalResult:
    """
    Fit a classifier 'est' (must implement fit / predict) and compute metrics + linear theory.
    """
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        est.fit(X_train, y_train)
    train_secs = time.time() - t0

    # Predictions
    yhat_train = est.predict(X_train)
    yhat_test = est.predict(X_test)
    acc_tr = float(accuracy_score(y_train, yhat_train))
    acc_te = float(accuracy_score(y_test, yhat_test))

    # Logits and linear mapping
    z_train = logits_from_estimator(est, X_train)
    Wb = get_linear_params(est)
    sigma = spectral_norm(Wb.W)
    frob = frobenius_norm(Wb.W)

    # PAC-Bayes-style bound (linear specialization) on TRAIN
    bound_parts = pac_bayes_bound_linear(Wb.W, X_train, y_train, z_train, gamma=gamma, delta=delta)

    # ECE if available on TEST
    ece = maybe_ece(est, X_test, y_test, n_bins=15)

    # Optional extras: e.g., Sobolev quadratic from your spectral Fourier LR
    ex = extras.copy() if extras is not None else {}
    # If estimator has Fourier params (your SpectralCirculantLogisticRegression), report magnitude and Sobolev energy
    for attr in ("F_real_", "F_imag_"):
        if hasattr(est, attr):
            F_r = getattr(est, "F_real_")
            F_i = getattr(est, "F_imag_")
            mag = np.sqrt(F_r ** 2 + F_i ** 2)  # (C,k_half)
            ex["fourier_mag_sum"] = float(np.sum(mag))
            # If the model used Sobolev scaling, we can approximate a quadratic:
            if hasattr(est, "_map") and hasattr(est._map, "padded_dim_"):
                pd = int(est._map.padded_dim_)
                omega = 2.0 * np.pi * np.arange(mag.shape[1], dtype=np.float64) / float(pd)
                s = float(getattr(est._map, "sobolev_s", 1.0))
                alpha = (1.0 + omega ** 2) ** s
                # sum_c sum_k alpha_k * |F_ck|^2
                ex["sobolev_quadratic"] = float(np.sum(alpha[None, :] * (mag ** 2)))

    return EvalResult(
        name=name,
        train_acc=acc_tr,
        test_acc=acc_te,
        ece_test=ece,
        lipschitz_sigma=sigma,
        frobenius=frob,
        gamma=gamma,
        Lhat_gamma=bound_parts["Lhat_gamma"],
        pacbayes_penalty=bound_parts["penalty"],
        pacbayes_bound=bound_parts["bound"],
        train_time_sec=train_secs,
        extras=ex,
    )

# ------------------------------ Runner --------------------------------

def run_suite_on_dataset(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    *,
    gamma: float = 1.0,
    delta: float = 0.05,
    random_state: int = 0,
) -> List[EvalResult]:
    results: List[EvalResult] = []

    # Baseline Logistic Regression
    pipe_lr, Wb_lr, t_lr = train_logreg_baseline(X_train, y_train, random_state=random_state)
    res_lr = evaluate_estimator(
        name="LogisticRegression+StandardScaler",
        est=pipe_lr,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        gamma=gamma, delta=delta,
        extras={"train_time_sec_fit_only": t_lr},
    )
    results.append(res_lr)

    # Baseline LinearSVC (hinge)
    model_svc, Wb_svc, t_svc = train_linearsvc_baseline(X_train, y_train, random_state=random_state, probability=True)
    res_svc = evaluate_estimator(
        name="LinearSVC+StandardScaler+Calibrated",
        est=model_svc,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        gamma=gamma, delta=delta,
        extras={"train_time_sec_fit_only": t_svc},
    )
    results.append(res_svc)

    # Fourier logistic
    spec_lr = SpectralCirculantLogisticRegression(
        padded_dim=None, K=None, feature_scaling="sobolev", sobolev_s=1.0,
        solver="lbfgs", C=1.0, max_iter=200, tol=1e-4, verbose=False
    )
    res_spec_lr = evaluate_estimator(
        name="SpectralCirculantLogReg(sobolev s=1.0)",
        est=spec_lr,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        gamma=gamma, delta=delta
    )
    results.append(res_spec_lr)

    # Fixed-basis logistic (random orthonormal features)
    spec_lin = SpectralLogisticRegression(
        n_spectral=None, basis="random", feature_scaling="standardize",
        C=1.0, max_iter=200, tol=1e-4, random_state=random_state
    )
    res_spec_lin = evaluate_estimator(
        name="SpectralLogReg(FixedOrtho+zscore)",
        est=spec_lin,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        gamma=gamma, delta=delta
    )
    results.append(res_spec_lin)

    # Spectral SVMs
    spec_svm_fourier = SpectralCirculantSVM(
        padded_dim=None, K=None, feature_scaling="sobolev", sobolev_s=1.0,
        C=1.0, loss="hinge", dual=True, max_iter=2000, tol=1e-4,
        probability=True, prob_cv=5
    )
    res_spec_svm_fourier = evaluate_estimator(
        name="SpectralCirculantSVM(sobolev s=1.0, calibrated)",
        est=spec_svm_fourier,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        gamma=gamma, delta=delta
    )
    results.append(res_spec_svm_fourier)

    spec_svm_fixed = SpectralSVM(
        n_spectral=None, basis="random", feature_scaling="standardize",
        C=1.0, loss="hinge", dual=True, max_iter=2000, tol=1e-4,
        probability=True, prob_cv=5, random_state=random_state
    )
    res_spec_svm_fixed = evaluate_estimator(
        name="SpectralSVM(FixedOrtho+zscore, calibrated)",
        est=spec_svm_fixed,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        gamma=gamma, delta=delta
    )
    results.append(res_spec_svm_fixed)

    return results

def results_to_table(results: List[EvalResult]) -> str:
    # Nicely formatted text table
    headers = [
        "model", "acc_train", "acc_test", "ECE", "||W||2", "||W||F",
        "gamma", "Lhat_gamma", "PAC-pen", "PAC-bound", "train_s"
    ]
    rows = []
    for r in results:
        rows.append([
            r.name,
            f"{r.train_acc:.4f}",
            f"{r.test_acc:.4f}",
            f"{r.ece_test:.4f}" if r.ece_test is not None else "n/a",
            f"{r.lipschitz_sigma:.3e}",
            f"{r.frobenius:.3e}",
            f"{r.gamma:.2f}",
            f"{r.Lhat_gamma:.4f}",
            f"{r.pacbayes_penalty:.3e}",
            f"{r.pacbayes_bound:.4f}",
            f"{r.train_time_sec:.2f}",
        ])
    colw = [max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    out = []
    out.append(" | ".join(h.ljust(colw[i]) for i, h in enumerate(headers)))
    out.append("-+-".join("-" * colw[i] for i in range(len(headers))))
    for row in rows:
        out.append(" | ".join(row[i].ljust(colw[i]) for i in range(len(headers))))
    return "\n".join(out)

# ----------------------------- __main__ --------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Eval strongly-convex spectral vs baseline models.")
    parser.add_argument("--dataset", type=str, default="digits",
                        choices=["digits", "mnist_784", "Fashion-MNIST", "breast_cancer", "wine", "iris"])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json_out", type=str, default="")
    args = parser.parse_args()

    if args.dataset in ("digits", "mnist_784", "Fashion-MNIST"):
        Xtr, Xte, ytr, yte = load_openml_or_digits(args.dataset, test_size=args.test_size, random_state=args.seed)
    else:
        Xtr, Xte, ytr, yte = load_sklearn_uci(args.dataset, test_size=args.test_size, random_state=args.seed)

    LOGGER.info(f"Dataset={args.dataset}  train={Xtr.shape}  test={Xte.shape}")

    results = run_suite_on_dataset(
        X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
        gamma=args.gamma, delta=args.delta, random_state=args.seed
    )

    print()
    print(results_to_table(results))
    print()

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        LOGGER.info(f"Wrote JSON to {args.json_out}")
