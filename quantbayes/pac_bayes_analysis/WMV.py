import time
import math
import numpy as np
from typing import List, Optional, Type, Dict
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.metrics import zero_one_loss
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

__all__ = [
    "BoundEnsemble",
    "PBLambdaCriterion",
    "TandemCriterion",
    "PBBernsteinCriterion",
]


class ConstantClassifier(BaseEstimator, ClassifierMixin):
    """Fallback for one-class subsets."""

    def __init__(self, constant_label: int):
        self.constant_label = constant_label

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.constant_label, dtype=int)


def _kl_div(p: float, q: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def _kl_inverse(p_hat: float, kl_term: float, n_r: int, tol: float = 1e-12) -> float:
    if p_hat >= 1.0:
        return 1.0
    target = kl_term / n_r
    lo, hi = p_hat, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _kl_div(p_hat, mid) > target:
            hi = mid
        else:
            lo = mid
    return lo


class BoundCriterion:
    def compute(self, *args, **kwargs):
        raise NotImplementedError


class PBLambdaCriterion(BoundCriterion):
    def compute(self, losses, rho, kl_rho, n_r, delta, lam, full_n):
        E_val = float(rho.dot(losses))
        term1 = E_val / (1 - lam / 2)
        term2 = (kl_rho + math.log(2 * math.sqrt(n_r) / delta)) / (
            lam * (1 - lam / 2) * n_r
        )
        gibbs = term1 + term2
        mv_bound = min(1.0, 2 * gibbs)
        return E_val, mv_bound


class PBKLCriterion(BoundCriterion):
    def compute(self, losses, rho, kl_rho, n_r, delta, lam, full_n):
        E_val = float(rho.dot(losses))
        kl_term = kl_rho + math.log(2 * math.sqrt(n_r) / delta)
        gibbs = _kl_inverse(E_val, kl_term, n_r)
        mv_bound = min(1.0, 2 * gibbs)
        return E_val, mv_bound


class TandemCriterion(BoundCriterion):
    def compute(self, pair_losses, rho, kl_rho, n_r, delta, lam, full_n):
        exp_t = float(rho @ pair_losses @ rho)
        term2 = (2 * kl_rho + math.log(2 * math.sqrt(full_n) / delta)) / (
            lam * (1 - lam / 2) * full_n
        )
        mv_bound = min(1.0, 4 * (exp_t / (1 - lam / 2) + term2))
        return exp_t, mv_bound


class PBBernsteinCriterion(BoundCriterion):
    def compute(self, losses, rho, kl_rho, n_r, delta, lam, full_n):
        E_val = float(rho.dot(losses))
        var = float(rho.dot((losses - E_val) ** 2))
        kl_term = kl_rho + math.log(2 * math.sqrt(n_r) / delta)
        term1 = math.sqrt(2 * var * kl_term / n_r)
        term2 = 2 * kl_term / (3 * n_r)
        gibbs = E_val + term1 + term2
        mv_bound = min(1.0, 2 * gibbs)
        return E_val, mv_bound


class BoundEnsemble(ClassifierMixin):
    _criteria_map = {
        "pblambda": PBLambdaCriterion,
        "pbkl": PBKLCriterion,
        "tandem": TandemCriterion,
        "pbbernstein": PBBernsteinCriterion,
    }
    _estimator_type = "classifier"

    def __init__(
        self,
        *,
        base_estimators: Optional[List[BaseEstimator]] = None,
        base_estimator_cls: Optional[Type[BaseEstimator]] = None,
        base_estimator_kwargs: Optional[Dict] = None,
        bound_type: str = "pbkl",
        bound_delta: float = 0.05,
        random_state: Optional[int] = None,
    ):
        if (base_estimators is None) == (base_estimator_cls is None):
            raise ValueError(
                "Specify exactly one of base_estimators or base_estimator_cls."
            )
        if bound_type not in self._criteria_map:
            raise ValueError(f"Unknown bound_type '{bound_type}'")
        self.base_estimators = base_estimators
        self.base_estimator_cls = base_estimator_cls
        self.base_estimator_kwargs = base_estimator_kwargs or {}
        self.bound_type = bound_type
        self.delta = bound_delta
        self.random_state = random_state
        self.rs = np.random.default_rng(random_state)
        self.is_fitted = False

    def _compute_jaakkola(self, X: np.ndarray) -> float:
        dists = pdist(X, metric="euclidean")
        return (2 * np.median(dists)) ** -2

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        m_values: List[int],
        n_runs: int = 10,
        max_iters: int = 200,
        tol: float = 1e-6,
    ) -> "BoundEnsemble":
        y = np.asarray(y)
        uniq = set(np.unique(y))
        if uniq <= {0, 1}:
            y = np.where(y == 0, -1, 1)
        elif uniq <= {-1, 1}:
            pass
        else:
            raise ValueError("Labels must be in {0,1} or {−1,1}.")

        n, d = X.shape
        r = d + 1
        n_r = n - r
        gamma_J = self._compute_jaakkola(X)
        gamma_grid = gamma_J * (10.0 ** np.arange(-4, 5))

        all_runs = []
        for seed in range(n_runs):
            self.rs = np.random.default_rng(
                self.random_state + seed if self.random_state is not None else None
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
                    if self.base_estimators is not None:
                        tmpl = self.base_estimators[i % len(self.base_estimators)]
                    else:
                        tmpl = self.base_estimator_cls(**self.base_estimator_kwargs)
                    clf = clone(tmpl)
                    # fallback if only one class in Si
                    if len(set(y[Si])) < 2:
                        clf = ConstantClassifier(constant_label=y[Si][0])
                    else:
                        clf.fit(X[Si], y[Si])
                    models.append(clf)
                    losses[i] = zero_one_loss(y[Sic], clf.predict(X[Sic]))

                pair_losses = None
                if self.bound_type == "tandem":
                    P = np.vstack([m_.predict(X) for m_ in models])
                    err = (P != y).astype(float)
                    pair_losses = err @ err.T / n

                pi = np.full(m, 1 / m)
                lam = max(1 / math.sqrt(n_r), 0.5)
                rho = pi.copy()
                crit = self._criteria_map[self.bound_type]()
                prev_b = np.inf

                for _ in range(max_iters):
                    kl_rho = float((rho * np.log(rho / pi)).sum())
                    stat, bound = crit.compute(
                        pair_losses if self.bound_type == "tandem" else losses,
                        rho,
                        kl_rho,
                        n_r,
                        self.delta,
                        lam,
                        n,
                    )
                    if abs(prev_b - bound) < tol:
                        break
                    prev_b = bound
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

                Pfull = np.vstack([m_.predict(X) for m_ in models]).T
                scr = Pfull.dot(rho)
                pred = np.where(scr >= 0, 1, -1)
                mv_loss = float(zero_one_loss(y, pred))

                run.append(
                    {
                        "m": m,
                        "mv_loss": mv_loss,
                        "bound": bound,
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
            raise RuntimeError("Call fit before summary.")
        hdr = f"{'m':>4s} | {'MV±σ':>12s} | {'Bnd±σ':>12s} | {'t±σ[s]':>12s}"
        print(hdr)
        print("-" * len(hdr))
        for i, m in enumerate(self.m_values):
            mv = np.array([run[i]["mv_loss"] for run in self.all_runs])
            bd = np.array([run[i]["bound"] for run in self.all_runs])
            tm = np.array([run[i]["time"] for run in self.all_runs])
            print(
                f"{m:4d} | {mv.mean():.4f}±{mv.std():.4f} | "
                f"{bd.mean():.4f}±{bd.std():.4f} | {tm.mean():.3f}±{tm.std():.3f}"
            )

    def plot(self, figsize=(8, 5)):
        if not self.is_fitted:
            raise RuntimeError("Call fit before plot.")
        ms = self.m_values
        arr_mv = np.array(
            [[run[j]["mv_loss"] for j in range(len(ms))] for run in self.all_runs]
        )
        arr_bd = np.array(
            [[run[j]["bound"] for j in range(len(ms))] for run in self.all_runs]
        )
        arr_tm = np.array(
            [[run[j]["time"] for j in range(len(ms))] for run in self.all_runs]
        )
        mv_m, mv_s = arr_mv.mean(0), arr_mv.std(0)
        bd_m, bd_s = arr_bd.mean(0), arr_bd.std(0)
        tm_m, tm_s = arr_tm.mean(0), arr_tm.std(0)
        ci = 1.96 / math.sqrt(self.n_runs)

        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.plot(ms, mv_m, "-o", label="MV loss")
        ax1.fill_between(ms, mv_m - ci * mv_s, mv_m + ci * mv_s, alpha=0.3)
        ax1.plot(ms, bd_m, "--s", label="Bound")
        ax1.fill_between(ms, bd_m - ci * bd_s, bd_m + ci * bd_s, alpha=0.3)
        ax1.set_xscale("log")
        ax1.set_xlabel("m")
        ax1.set_ylabel("Loss/Bound")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(ms, tm_m, "-.^", label="Time[s]")
        ax2.fill_between(ms, tm_m - ci * tm_s, tm_m + ci * tm_s, alpha=0.3)
        ax2.set_ylabel("Time[s]")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Call fit before predict.")
        P = np.vstack([clf.predict(X) for clf in self.models_]).T
        scores = P.dot(self.rho_)
        return np.where(scores >= 0, 1, -1)

    def get_params(self, deep=True) -> Dict:
        params = {
            "bound_type": self.bound_type,
            "bound_delta": self.delta,
            "random_state": self.random_state,
        }
        if self.base_estimators is not None:
            params["base_estimators"] = self.base_estimators
        else:
            params["base_estimator_cls"] = self.base_estimator_cls
            params["base_estimator_kwargs"] = self.base_estimator_kwargs
        return params

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self


# --------- Quick test ---------
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    # Create mixture of weak learners with user-defined hyperparameters
    templates = [
        LogisticRegression(C=0.01, solver="liblinear"),
        RandomForestClassifier(n_estimators=10, max_depth=2),
        SVC(kernel="poly", degree=3, C=0.5),
    ]

    Xb, yb = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        flip_y=0.1,
        class_sep=1.0,
        random_state=0,
    )
    ens = BoundEnsemble(
        base_estimators=templates,
        bound_type="pbkl",
        bound_delta=0.05,
        random_state=42,
    )
    ens.fit(Xb, yb, m_values=[5, 10], n_runs=3)
    ens.summary()
    ens.plot()
