# File: regression_ensemble.py

import time
import math
import numpy as np
from typing import List, Optional, Callable, Dict, Type
from sklearn.base import clone, BaseEstimator, RegressorMixin
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

# adjust this import to match your project structure
from quantbayes.pac_bayes_analysis.WMV import (
    PBKLCriterion,
    PBLambdaCriterion,
    TandemCriterion,
    PBBernsteinCriterion,
)

CRITERIA_MAP = {
    "pbkl": PBKLCriterion,
    "pblambda": PBLambdaCriterion,
    "tandem": TandemCriterion,
    "pbbernstein": PBBernsteinCriterion,
}


class BoundEnsembleRegressor(RegressorMixin):
    """
    PAC-Bayes ensemble for regression with a bounded loss in [0,1].
    Default: clipped squared error loss, normalized to [0,1].
    """

    def __init__(
        self,
        *,
        base_estimators: Optional[List[BaseEstimator]] = None,
        base_estimator_cls: Optional[Type[BaseEstimator]] = None,
        base_estimator_kwargs: Optional[Dict] = None,
        bound_type: str = "pbkl",
        bound_delta: float = 0.05,
        random_state: Optional[int] = None,
        loss_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        L_max: float = 1.0,
    ):
        if (base_estimators is None) == (base_estimator_cls is None):
            raise ValueError(
                "Specify exactly one of base_estimators or base_estimator_cls."
            )
        if bound_type not in CRITERIA_MAP:
            raise ValueError(f"Unknown bound_type '{bound_type}'")
        self.base_estimators = base_estimators
        self.base_estimator_cls = base_estimator_cls
        self.base_estimator_kwargs = base_estimator_kwargs or {}
        self.bound_type = bound_type
        self.delta = bound_delta
        self.rs = np.random.default_rng(random_state)
        self.L_max = L_max
        self.loss_fn = (
            (lambda y, y_pred: np.minimum((y - y_pred) ** 2, L_max) / L_max)
            if loss_fn is None
            else loss_fn
        )
        self.is_fitted = False

    def _compute_jaakkola(self, X: np.ndarray) -> float:
        return (2 * np.median(pdist(X))) ** -2

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        m_values: List[int],
        n_runs: int = 10,
        max_iters: int = 200,
        tol: float = 1e-6,
    ) -> "BoundEnsembleRegressor":
        n, d = X.shape
        r = d + 1
        n_r = n - r
        all_runs = []
        for seed in range(n_runs):
            self.rs = np.random.default_rng(
                self.rs.bit_generator.state["state"]["state"] + seed
            )
            run = []
            for m in m_values:
                t0 = time.time()
                losses = np.zeros(m)
                models = []
                for i in range(m):
                    idx = np.arange(n)
                    Si = self.rs.choice(idx, size=r, replace=False)
                    Sic = np.setdiff1d(idx, Si)
                    tmpl = (
                        self.base_estimators[i % len(self.base_estimators)]
                        if self.base_estimators
                        else self.base_estimator_cls(**self.base_estimator_kwargs)
                    )
                    clf = clone(tmpl)
                    clf.fit(X[Si], y[Si])
                    models.append(clf)
                    losses[i] = np.mean(self.loss_fn(y[Sic], clf.predict(X[Sic])))

                # optimize chosen PAC-Bayes bound
                pi = np.full(m, 1 / m)
                lam = max(1 / math.sqrt(n_r), 0.5)
                rho = pi.copy()
                crit = CRITERIA_MAP[self.bound_type]()
                prev_b = np.inf
                for _ in range(max_iters):
                    kl_rho = float((rho * np.log(rho / pi)).sum())
                    stat, bnd = crit.compute(
                        losses, rho, kl_rho, n_r, self.delta, lam, n
                    )
                    if abs(prev_b - bnd) < tol:
                        break
                    prev_b = bnd
                    lam = 2.0 / (
                        math.sqrt(
                            1
                            + 2
                            * n_r
                            * stat
                            / (kl_rho + math.log(2 * math.sqrt(n_r) / self.delta))
                        )
                        + 1
                    )
                    shift = losses.min()
                    w = np.exp(-lam * n_r * (losses - shift))
                    rho = w / w.sum()

                run.append(
                    {
                        "m": m,
                        "loss": float(np.dot(rho, losses)),
                        "bound": bnd,
                        "time": time.time() - t0,
                    }
                )
                if seed == n_runs - 1 and m == m_values[-1]:
                    self.models_, self.rho_ = models, rho
            all_runs.append(run)

        self.all_runs, self.m_values, self.n_runs, self.is_fitted = (
            all_runs,
            m_values,
            n_runs,
            True,
        )
        return self

    def summary(self):
        if not self.is_fitted:
            raise RuntimeError("Call fit first.")
        header = f"{'m':>4s} | {'Loss±σ':>12s} | {'Bound±σ':>12s}"
        print(header)
        print("-" * len(header))
        for i, m in enumerate(self.m_values):
            L = np.array([run[i]["loss"] for run in self.all_runs])
            B = np.array([run[i]["bound"] for run in self.all_runs])
            print(
                f"{m:4d} | {L.mean():.4f}±{L.std():.4f} | {B.mean():.4f}±{B.std():.4f}"
            )

    def plot(self):
        if not self.is_fitted:
            raise RuntimeError("Call fit first.")
        ms = self.m_values
        loss_means = [
            np.mean([run[i]["loss"] for run in self.all_runs]) for i in range(len(ms))
        ]
        bound_means = [
            np.mean([run[i]["bound"] for run in self.all_runs]) for i in range(len(ms))
        ]
        plt.plot(ms, loss_means, "-o", label="Mean Loss")
        plt.plot(ms, bound_means, "-s", label="Mean Bound")
        plt.xscale("log")
        plt.xlabel("m (weak learners)")
        plt.ylabel("Loss / Bound")
        plt.legend()
        plt.tight_layout()
        plt.show()


# Quick test
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor

    X, y = make_regression(n_samples=200, n_features=5, noise=10.0, random_state=0)
    ens = BoundEnsembleRegressor(
        base_estimators=[DecisionTreeRegressor(max_depth=2)],
        bound_type="pbkl",
        bound_delta=0.05,
        random_state=0,
    )
    ens.fit(X, y, m_values=[5, 10, 20], n_runs=5)
    ens.summary()
    ens.plot()
