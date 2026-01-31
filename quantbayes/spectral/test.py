# quantbayes/spectral/test.py
# Tutorial runner for strongly-convex (linear-on-fixed-features) models.
# Uses your spectral estimators and the evaluator utilities.

from __future__ import annotations
import json
import math
import time
import warnings
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning

# --- Import the evaluation utilities ---
from quantbayes.spectral.strongly_convex_eval import (
    evaluate_estimator,
    gamma_sweep_bound,
    results_to_table,
    logits_from_estimator,
    get_linear_Wb_from_estimator,
)

# --- Import YOUR spectral models ---
from quantbayes.spectral.spectral_modules import (
    SpectralCirculantLogisticRegression,
    SpectralLogisticRegression,
    SpectralCirculantSVM,
    SpectralSVM,
)

from quantbayes.spectral.graph_spectral_modules import (
    GraphSobolevLogisticRegression,
    GraphSobolevSVM,
)


def _to_py(obj):
    """JSON-safe conversion for numpy types/arrays inside results."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    return obj


def run_binary_demo(seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=4000,
        n_features=40,
        n_informative=25,
        n_redundant=5,
        n_classes=2,
        class_sep=1.5,
        random_state=seed,
        # data are already in original space; evaluator will compute B there
    )
    Xtr, Xte = X[:3000], X[3000:]
    ytr, yte = y[:3000], y[3000:]

    results = []
    sweeps = {}

    # ---------------- Baselines ----------------
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Logistic Regression with StandardScaler
    lr = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=600, solver="lbfgs")
    )
    lr.fit(Xtr, ytr)
    res_lr = evaluate_estimator(
        "LR+Scaler", lr, Xtr, ytr, Xte, yte, gamma=1.0, delta=0.05, already_fit=True
    )
    ztr_lr = logits_from_estimator(lr, Xtr)
    Wb_lr = get_linear_Wb_from_estimator(lr)
    sweep_lr = gamma_sweep_bound(res_lr, Xtr, ytr, logits_train=ztr_lr, Wb=Wb_lr)
    results.append(res_lr)
    sweeps["LR"] = {
        "best_gamma": sweep_lr.best_gamma,
        "best_bound": sweep_lr.best_bound,
    }

    # LinearSVC + calibration (for ECE); also get W,b from the pre-calibrated pipeline
    base_svc = make_pipeline(StandardScaler(), LinearSVC(max_iter=3000))
    base_svc.fit(Xtr, ytr)
    Wb_svc = get_linear_Wb_from_estimator(base_svc)  # fold-back before calibration
    svc_cal = CalibratedClassifierCV(estimator=base_svc, cv=5, method="sigmoid").fit(
        Xtr, ytr
    )
    res_svc = evaluate_estimator(
        "LinearSVC+Scaler+Calibrated",
        svc_cal,
        Xtr,
        ytr,
        Xte,
        yte,
        gamma=1.0,
        delta=0.05,
        Wb_override=Wb_svc,
        already_fit=True,
    )
    ztr_svc = logits_from_estimator(svc_cal, Xtr)
    sweep_svc = gamma_sweep_bound(res_svc, Xtr, ytr, logits_train=ztr_svc, Wb=Wb_svc)
    results.append(res_svc)
    sweeps["SVC_cal"] = {
        "best_gamma": sweep_svc.best_gamma,
        "best_bound": sweep_svc.best_bound,
    }

    # -------------- Your spectral models --------------
    # Fourier logistic (RFFT), Sobolev scaling
    spec_lr_fourier = SpectralCirculantLogisticRegression(
        padded_dim=None,
        K=None,
        feature_scaling="sobolev",
        sobolev_s=1.0,
        C=1.0,
        max_iter=600,
        tol=1e-4,
        solver="lbfgs",
    )
    res_spec_lr_fourier = evaluate_estimator(
        "SpecCirculantLR(sobolev)",
        spec_lr_fourier,
        Xtr,
        ytr,
        Xte,
        yte,
        gamma=1.0,
        delta=0.05,
    )
    ztr_spec_lr_fourier = logits_from_estimator(spec_lr_fourier, Xtr)
    Wb_spec_lr_fourier = get_linear_Wb_from_estimator(spec_lr_fourier)
    sweep_spec_lr_fourier = gamma_sweep_bound(
        res_spec_lr_fourier,
        Xtr,
        ytr,
        logits_train=ztr_spec_lr_fourier,
        Wb=Wb_spec_lr_fourier,
    )
    results.append(res_spec_lr_fourier)
    sweeps["SpecCirculantLR"] = {
        "best_gamma": sweep_spec_lr_fourier.best_gamma,
        "best_bound": sweep_spec_lr_fourier.best_bound,
    }

    # Fixed-orthonormal-basis logistic (random + QR), standardized features
    spec_lr_fixed = SpectralLogisticRegression(
        n_spectral=None,
        basis="random",
        feature_scaling="standardize",
        C=1.0,
        max_iter=600,
        tol=1e-4,
        solver="lbfgs",
        random_state=seed,
    )
    res_spec_lr_fixed = evaluate_estimator(
        "SpecFixedBasisLR(zscore)",
        spec_lr_fixed,
        Xtr,
        ytr,
        Xte,
        yte,
        gamma=1.0,
        delta=0.05,
    )
    ztr_spec_lr_fixed = logits_from_estimator(spec_lr_fixed, Xtr)
    Wb_spec_lr_fixed = get_linear_Wb_from_estimator(spec_lr_fixed)
    sweep_spec_lr_fixed = gamma_sweep_bound(
        res_spec_lr_fixed, Xtr, ytr, logits_train=ztr_spec_lr_fixed, Wb=Wb_spec_lr_fixed
    )
    results.append(res_spec_lr_fixed)
    sweeps["SpecFixedBasisLR"] = {
        "best_gamma": sweep_spec_lr_fixed.best_gamma,
        "best_bound": sweep_spec_lr_fixed.best_bound,
    }

    # Fourier SVM with calibration for ECE
    spec_svm_fourier = SpectralCirculantSVM(
        padded_dim=None,
        K=None,
        feature_scaling="sobolev",
        sobolev_s=1.0,
        C=1.0,
        loss="hinge",
        dual=True,
        max_iter=5000,
        probability=True,
        prob_cv=5,
    )
    res_spec_svm_fourier = evaluate_estimator(
        "SpecCirculantSVM(sobolev,cal)",
        spec_svm_fourier,
        Xtr,
        ytr,
        Xte,
        yte,
        gamma=1.0,
        delta=0.05,
    )
    ztr_spec_svm_fourier = logits_from_estimator(spec_svm_fourier, Xtr)
    Wb_spec_svm_fourier = get_linear_Wb_from_estimator(spec_svm_fourier)
    sweep_spec_svm_fourier = gamma_sweep_bound(
        res_spec_svm_fourier,
        Xtr,
        ytr,
        logits_train=ztr_spec_svm_fourier,
        Wb=Wb_spec_svm_fourier,
    )
    results.append(res_spec_svm_fourier)
    sweeps["SpecCirculantSVM"] = {
        "best_gamma": sweep_spec_svm_fourier.best_gamma,
        "best_bound": sweep_spec_svm_fourier.best_bound,
    }

    # Graph-Sobolev logistic (default: chain Laplacian over feature index)
    graph_lr = GraphSobolevLogisticRegression(
        feature_scaling="sobolev",
        sobolev_s=1.0,
        sobolev_mu=1e-3,
        C=1.0,
        max_iter=600,
        tol=1e-4,
        solver="lbfgs",
    )
    res_graph_lr = evaluate_estimator(
        "GraphSobolevLR(chain)", graph_lr, Xtr, ytr, Xte, yte, gamma=1.0, delta=0.05
    )
    ztr_graph_lr = logits_from_estimator(graph_lr, Xtr)
    Wb_graph_lr = get_linear_Wb_from_estimator(graph_lr)
    sweep_graph_lr = gamma_sweep_bound(
        res_graph_lr, Xtr, ytr, logits_train=ztr_graph_lr, Wb=Wb_graph_lr
    )
    results.append(res_graph_lr)
    sweeps["GraphSobolevLR"] = {
        "best_gamma": sweep_graph_lr.best_gamma,
        "best_bound": sweep_graph_lr.best_bound,
    }

    # Graph-Sobolev SVM (uncalibrated by default; set probability=True if needed)
    graph_svm = GraphSobolevSVM(
        feature_scaling="sobolev",
        sobolev_s=1.0,
        sobolev_mu=1e-3,
        C=1.0,
        loss="hinge",
        max_iter=5000,
        probability=True,
    )
    res_graph_svm = evaluate_estimator(
        "GraphSobolevSVM(chain)", graph_svm, Xtr, ytr, Xte, yte, gamma=1.0, delta=0.05
    )
    ztr_graph_svm = logits_from_estimator(graph_svm, Xtr)
    Wb_graph_svm = get_linear_Wb_from_estimator(graph_svm)
    sweep_graph_svm = gamma_sweep_bound(
        res_graph_svm, Xtr, ytr, logits_train=ztr_graph_svm, Wb=Wb_graph_svm
    )
    results.append(res_graph_svm)
    sweeps["GraphSobolevSVM"] = {
        "best_gamma": sweep_graph_svm.best_gamma,
        "best_bound": sweep_graph_svm.best_bound,
    }

    return {
        "results": results,
        "sweeps": sweeps,
    }


def run_multiclass_demo(seed: int = 1) -> Dict[str, Any]:
    X, y = make_classification(
        n_samples=5000,
        n_features=60,
        n_informative=30,
        n_redundant=10,
        n_classes=5,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=seed,
    )
    Xtr, Xte = X[:4000], X[4000:]
    ytr, yte = y[:4000], y[4000:]

    results = []
    sweeps = {}

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Baseline multiclass LR + scaler
    lr_mc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=800))
    lr_mc.fit(Xtr, ytr)
    res_lr_mc = evaluate_estimator(
        "LR-mc+Scaler",
        lr_mc,
        Xtr,
        ytr,
        Xte,
        yte,
        gamma=1.0,
        delta=0.05,
        already_fit=True,
    )
    ztr_lr_mc = logits_from_estimator(lr_mc, Xtr)
    Wb_lr_mc = get_linear_Wb_from_estimator(lr_mc)
    sweep_lr_mc = gamma_sweep_bound(
        res_lr_mc, Xtr, ytr, logits_train=ztr_lr_mc, Wb=Wb_lr_mc
    )
    results.append(res_lr_mc)
    sweeps["LR_mc"] = {
        "best_gamma": sweep_lr_mc.best_gamma,
        "best_bound": sweep_lr_mc.best_bound,
    }

    # Spectral Fourier LR
    spec_lr_mc = SpectralCirculantLogisticRegression(
        feature_scaling="sobolev", sobolev_s=1.0, C=1.0, max_iter=800, solver="lbfgs"
    )
    res_spec_lr_mc = evaluate_estimator(
        "SpecCirculantLR-mc(sobolev)",
        spec_lr_mc,
        Xtr,
        ytr,
        Xte,
        yte,
        gamma=1.0,
        delta=0.05,
    )
    ztr_spec_lr_mc = logits_from_estimator(spec_lr_mc, Xtr)
    Wb_spec_lr_mc = get_linear_Wb_from_estimator(spec_lr_mc)
    sweep_spec_lr_mc = gamma_sweep_bound(
        res_spec_lr_mc, Xtr, ytr, logits_train=ztr_spec_lr_mc, Wb=Wb_spec_lr_mc
    )
    results.append(res_spec_lr_mc)
    sweeps["SpecCirculantLR_mc"] = {
        "best_gamma": sweep_spec_lr_mc.best_gamma,
        "best_bound": sweep_spec_lr_mc.best_bound,
    }

    # Graph-Sobolev LR (default chain Laplacian)
    graph_lr_mc = GraphSobolevLogisticRegression(
        feature_scaling="sobolev",
        sobolev_s=1.0,
        sobolev_mu=1e-3,
        C=1.0,
        max_iter=800,
        solver="lbfgs",
    )
    res_graph_lr_mc = evaluate_estimator(
        "GraphSobolevLR-mc(chain)",
        graph_lr_mc,
        Xtr,
        ytr,
        Xte,
        yte,
        gamma=1.0,
        delta=0.05,
    )
    ztr_graph_lr_mc = logits_from_estimator(graph_lr_mc, Xtr)
    Wb_graph_lr_mc = get_linear_Wb_from_estimator(graph_lr_mc)
    sweep_graph_lr_mc = gamma_sweep_bound(
        res_graph_lr_mc, Xtr, ytr, logits_train=ztr_graph_lr_mc, Wb=Wb_graph_lr_mc
    )
    results.append(res_graph_lr_mc)
    sweeps["GraphSobolevLR_mc"] = {
        "best_gamma": sweep_graph_lr_mc.best_gamma,
        "best_bound": sweep_graph_lr_mc.best_bound,
    }

    return {
        "results": results,
        "sweeps": sweeps,
    }


def main():
    # ---- Binary demo ----
    binary = run_binary_demo(seed=0)
    print("\n[BINARY]")
    print(results_to_table(binary["results"]))

    # ---- Multiclass demo ----
    multiclass = run_multiclass_demo(seed=1)
    print("\n[MULTICLASS]")
    print(results_to_table(multiclass["results"]))
    print()

    # ---- Save JSON (safe serialization) ----
    out = {
        "binary": {
            "results": [_to_py(asdict(r)) for r in binary["results"]],
            "sweeps": _to_py(binary["sweeps"]),
        },
        "multiclass": {
            "results": [_to_py(asdict(r)) for r in multiclass["results"]],
            "sweeps": _to_py(multiclass["sweeps"]),
        },
    }
    with open("strongly_convex_test_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote strongly_convex_test_summary.json")


if __name__ == "__main__":
    main()
