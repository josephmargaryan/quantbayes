import copy
import time
import math
import numpy as np
from typing import List, Optional, Type, Dict
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.metrics import zero_one_loss

__all__ = [
    "PBLambdaCriterion", 
    "PBKLCriterion", 
    "TandemCriterion", 
    "PBBernsteinCriterion", 
    "SplitKLCriterion", 
    "UnexpectedBernsteinCriterion",
    "BoundEnsemble"
    ]

def _kl_div(p: float, q: float, eps: float = 1e-15) -> float:
    """
    Computes KL(p || q) for Bernoulli distributions, with clamping to avoid log(0).
    """
    p_clamped = min(max(p, eps), 1 - eps)
    q_clamped = min(max(q, eps), 1 - eps)
    return p_clamped * math.log(p_clamped / q_clamped) + (1 - p_clamped) * math.log((1 - p_clamped) / (1 - q_clamped))

def _kl_inverse(p_hat: float, kl_term: float, n_r: int, tol: float = 1e-12) -> float:
    """
    Finds the smallest q >= p_hat such that KL(p_hat || q) <= (kl_term / n_r), via binary search.
    """
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
    def compute(self, losses, rho, kl_rho, n_r, delta, lam, full_n):
        raise NotImplementedError

class PBLambdaCriterion(BoundCriterion):
    def compute(self, losses, rho, kl_rho, n_r, delta, lam, full_n):
        """
        Returns:
            E_val:      E_rho[losses]
            mv_bound:   majority-vote bound = min(1, 2 * GibbsBound),
                        where GibbsBound = E_val/(1 - lam/2) + (kl_rho + log(2 sqrt(n_r)/delta))/(lam (1 - lam/2) n_r)
        """
        E_val = float(rho.dot(losses))
        log_term = math.log(2 * math.sqrt(n_r) / delta)
        term1 = E_val / (1 - lam / 2)
        term2 = (kl_rho + log_term) / (lam * (1 - lam / 2) * n_r)
        gibbs = term1 + term2
        mv_bound = min(1.0, 2 * gibbs)
        return E_val, mv_bound

class PBKLCriterion(BoundCriterion):
    def compute(self, losses, rho, kl_rho, n_r, delta, lam, full_n):
        """
        Returns:
            E_val:      E_rho[losses]
            mv_bound:   majority-vote bound = min(1, 2 * q),
                        where q = inverse-KL(p_hat, (kl_rho + log(2 sqrt(n_r)/delta)), n_r)
        """
        E_val = float(rho.dot(losses))
        log_term = math.log(2 * math.sqrt(n_r) / delta)
        kl_term = kl_rho + log_term
        q = _kl_inverse(E_val, kl_term, n_r)
        mv_bound = min(1.0, 2 * q)
        return E_val, mv_bound

class TandemCriterion(BoundCriterion):
    def compute(self, pair_losses, rho, kl_rho, n_r, delta, lam, full_n):
        """
        Returns:
            exp_t:      E_{rho^2}[pairwise co-error]
            mv_bound:   tandem majority-vote bound = min(1, 4 * (exp_t/(1 - lam/2) + (2 kl_rho + log(2 sqrt(full_n)/delta))/(lam (1 - lam/2) full_n)))
        """
        exp_t = float(rho @ pair_losses @ rho)  # E_{rho^2}[co-error]
        log_term = math.log(2 * math.sqrt(full_n) / delta)
        term2 = (2 * kl_rho + log_term) / (lam * (1 - lam / 2) * full_n)
        mv_bound = min(1.0, 4 * (exp_t / (1 - lam / 2) + term2))
        return exp_t, mv_bound

class PBBernsteinCriterion(BoundCriterion):
    def compute(self, losses, rho, kl_rho, n_r, delta, lam, full_n):
        """
        Returns:
            E_val:      E_rho[losses]
            mv_bound:   Bernstein majority-vote bound = min(1, 2 * (E_val + sqrt(2 var kl_term / n_r) + (2 kl_term)/(3 n_r)))
        """
        E_val = float(rho.dot(losses))
        var = float(rho.dot((losses - E_val) ** 2))
        log_term = math.log(2 * math.sqrt(n_r) / delta)
        kl_term = kl_rho + log_term
        term1 = math.sqrt(2 * var * kl_term / n_r)
        term2 = 2 * kl_term / (3 * n_r)
        gibbs = E_val + term1 + term2
        mv_bound = min(1.0, 2 * gibbs)
        return E_val, mv_bound

# ───────────────────────────────────────────────────────────────────────────────
# 1) Split-KL Criterion
# ───────────────────────────────────────────────────────────────────────────────
class SplitKLCriterion(BoundCriterion):
    """
    PAC-Bayes “split-kl” bound for finite‐valued losses.

    Suppose each model i incurs an empirical loss `losses[i]` ∈ {b_0, b_1, …, b_K}.
    We write
        F(h) = E_Z[f(h, Z)] ∈ [b_0, b_K],
    where f(·) takes values in that finite set.  Then a PAC-Bayes‐split-kl bound is

        E_{h∼ρ}[F(h)]
           ≤ b_0
             + ∑_{j=1}^K α_j · kl^{-1+} (  E_{h∼ρ}[ ˆF_{|j}(h) ] ,
                                              (KL(ρ‖π) + ln(2 K √n_r / δ)) / n_r  )
    where α_j = b_j − b_{j−1}, and
        ˆF_{|j}(h) = (1/n_r) ∑_{t=1}^{n_r} 1{f(h, Z_t) ≥ b_j}.
    Here, `losses` is a length‐m array of empirical losses for each model; we assume
    each entry in `losses` lies in a small finite set.  In our WMV context, these
    “losses” are zero‐one or multi‐level excess‐loss values.

    Implementation details:
    - We extract the sorted unique values of `losses` as [b_0, b_1, …, b_K].
    - For each j=1..K, define:
          p_hat_j = ∑_{i : losses[i] ≥ b_j} ρ[i].
      Then kl^{-1+}(p_hat_j, (kl_rho + ln(2 K √n_r/δ))/n_r) is the usual Bernoulli kl-inverse.
    - Finally we multiply by α_j = b_j − b_{j−1} and sum.
    - We return (E_val, mv_bound) where
          E_val    = ρ · losses,
          mv_bound = min(1.0, 2·(b_0 + Σ_j α_j · q_j)).
    """
    def compute(self, losses, rho, kl_rho, n_r, delta, lam, full_n):
        # losses: 1D np.array of length m, each entry ∈ some finite set {b_0,...,b_K}
        # rho:     1D np.array of length m (weights summing to 1)
        # kl_rho:  float = KL(ρ‖π)
        # n_r:     number of hold-out samples used to estimate each model’s loss
        # delta:   confidence parameter
        # lam:     not used in split-kl (we ignore it)
        # full_n:  full training size, not used here

        # 1) Find the distinct sorted “loss‐levels” b_0 < b_1 < … < b_K:
        b_vals = np.unique(losses)
        K = len(b_vals) - 1
        if K < 1:
            # Degenerate case: all losses are equal to b_0
            E_val = float(rho.dot(losses))
            mv_bound = min(1.0, 2 * b_vals[0])
            return E_val, mv_bound

        # 2) Precompute the “log‐term” in the kl penalty:
        #    ln( 2·K·√n_r / δ )
        log_term = math.log(2 * K * math.sqrt(n_r) / delta)

        # 3) Start building the split-kl sum:
        bound_sum = b_vals[0]  # this is the base term
        for j in range(1, len(b_vals)):
            alpha_j = b_vals[j] - b_vals[j-1]

            # empirical p̂_j = ρ·[1{losses ≥ b_j}]
            mask = (losses >= b_vals[j]).astype(float)
            p_hat_j = float(rho.dot(mask))

            kl_term = kl_rho + log_term
            # q_j = kl^{-1+}( p_hat_j , kl_term / n_r )
            q_j = _kl_inverse(p_hat_j, kl_term, n_r)

            bound_sum += alpha_j * q_j

        # 4) We interpret this as a bound on E_ρ[F(h)].  For majority‐vote, we apply
        #    the standard “Gibbs→MV” factor of 2:
        mv_bound = min(1.0, 2 * bound_sum)

        E_val = float(rho.dot(losses))
        return E_val, mv_bound


# ───────────────────────────────────────────────────────────────────────────────
# 2) “Unexpected” Bernstein Criterion
# ───────────────────────────────────────────────────────────────────────────────
class UnexpectedBernsteinCriterion(BoundCriterion):
    """
    (“Unexpected” Bernstein PAC-Bayes bound from Theorem 3.10 / 3.11 in the notes.)

    In contrast to the PB-Bernstein bound, which requires sqrt(var) and clamps via
    a Hoeffding/Bernstein lemma, the “unexpected” Bernstein bound uses only the
    _empirical second moment_ (i.e. v̂ = E_ρ[loss_i^2]) and a grid over λ.

    Concretely, for losses in [0, 1], one shows w.p. ≥ 1−δ that, for every ρ:
      E_{h∼ρ}[L(h)]
        ≤ E_val
          + min_{λ ∈ Λ} [
              ( ψ(−λ·b) / (λ·b²) ) · σ̂²
              + ( KL(ρ‖π) + ln(|Λ|/δ) ) / (n_r · λ)
            ]
    where b = 1 (upper bound on loss_i), σ̂² = E_ρ[loss_i²], and
      ψ(u) = u − ln(1 + u).
    We implement that minimization over a small dyadic grid Λ ⊂ (0, 1), then
    again multiply by 2 to convert a Gibbs-type bound into a majority-vote bound.

    Usage:
      - losses: 1D np.array of length m, each in [0,1]
      - rho:     1D weights, ∑ rho = 1
      - kl_rho:  KL(ρ‖π)
      - n_r:     hold-out size
      - delta:   confidence
      - lam:     unused (we pick our own λs internally)
      - full_n:  unused

    Returns (E_val, mv_bound), where
      E_val    = ρ·losses,
      mv_bound = min(1, 2 · ( E_val + best_term )).
    """

    def __init__(self, lambdas=None):
        # If the user passed a custom list of λ-values, use it; otherwise default to a dyadic grid.
        if lambdas is not None:
            self.lambdas = lambdas
        else:
            # a small dyadic grid in (0, 1); here we stop at 2^-8≈0.0039
            self.lambdas = [2.0 ** (-i) for i in range(1, 9)]

    def compute(self, losses, rho, kl_rho, n_r, delta, lam, full_n):
        """
        Compute the “unexpected Bernstein” majority‐vote bound via:
          E_val = ρ·losses
          var̂   = ρ·(losses²)

          best_term = min_{λ∈Λ} [
                         coeff(λ) * var̂
                       + (kl_rho + ln(|Λ|/δ)) / (n_r · λ)
                     ]
          mv_bound = min(1, 2·(E_val + best_term))

        where
          coeff(λ) = ψ(−λ·b)/(λ·b²),  b=1,  ψ(u)=u−ln(1+u),  so here ψ(−λ)=−λ − ln(1−λ).
        """
        # 1) E_val = E_ρ[losses]
        E_val = float(rho.dot(losses))

        # 2) empirical second moment: var̂ = E_ρ[losses²]
        var_hat = float(rho.dot(losses**2))

        # 3) the penalty term in the numerator: KL(ρ‖π) + ln(|Λ|/δ)
        log_term = kl_rho + math.log(len(self.lambdas) / delta)

        best_term = float("inf")
        for λ in self.lambdas:
            if λ >= 1.0:
                continue  # we restrict λ < 1 because ψ(−λ) requires (1−λ)>0

            # ψ(−λ·b) with b=1 → ψ(−λ) = (−λ) − ln(1 − λ)
            psi = -λ - math.log(1.0 - λ)

            # coeff(λ) = ψ(−λ)/(λ·b²)  with b=1
            coeff = psi / λ

            term_candidate = coeff * var_hat + log_term / (n_r * λ)
            if term_candidate < best_term:
                best_term = term_candidate

        # final majority‐vote bound is 2·(E_val + best_term), clipped at 1
        mv_bound = min(1.0, 2.0 * (E_val + best_term))

        return E_val, mv_bound

# ───────────────────────────────────────────────────────────────────────────────
# 3) Unified BoundEnsemble class (binary or multiclass)
# ───────────────────────────────────────────────────────────────────────────────
class BoundEnsemble(BaseEstimator, ClassifierMixin):
    """
    Unified binary/multiclass PAC-Bayes ensemble (PB-KL, PB-λ, Tandem, PB-Bernstein).

    Parameters
    ----------
    task : {"binary", "multiclass"}
        Which kind of classification problem.
    base_estimators : Optional[List[BaseEstimator]]
        If provided, a fixed list of pre-instantiated estimator templates. We'll clone them in turn.
    base_estimator_cls : Optional[Type[BaseEstimator]]
        If no templates provided, we will instantiate this class with **base_estimator_kwargs**.
    base_estimator_kwargs : Optional[Dict]
        kwargs to pass to `base_estimator_cls(...)` if `base_estimators` is None.
    bound_type : {"pbkl", "pblambda", "tandem", "pbbernstein"}, default="pbkl"
        Which PAC-Bayes bound to optimize.
    bound_delta : float, default=0.05
        Confidence parameter δ ∈ (0,1).
    random_state : int or None, default=None
        Seed for reproducibility of random subsets.

    Attributes (after fit)
    ----------
    all_runs : List[List[dict]]
        A list (length = n_runs) of run results. Each run is a list of dicts, one per m in m_values,
        with keys {"m", "mv_loss"/"err", "bound", "time"}.
    best_bound_ : float
        Lowest bound achieved across all seeds and all m_values.
    best_m_ : int
        Ensemble size (m) that achieved best bound.
    best_seed_ : int
        Seed index (0 ≤ seed < n_runs) that achieved best bound.
    best_models_ : List[BaseEstimator]
        The fitted base models corresponding to (best_seed_, best_m_).
    best_rho_ : np.ndarray of shape (best_m_,)
        The Gibbs weights of the best ensemble.
    m_values : List[int]
        The list of ensemble sizes tried.
    n_runs : int
        Number of independent random‐subset runs (seeds).
    classes_ : np.ndarray
        In multiclass mode, array of unique classes.
    is_fitted : bool
        Whether `fit(...)` has been called.
    """

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
        task: str = "binary",
        base_estimators: Optional[List[BaseEstimator]] = None,
        base_estimator_cls: Optional[Type[BaseEstimator]] = None,
        base_estimator_kwargs: Optional[Dict] = None,
        bound_type: str = "pbkl",
        bound_delta: float = 0.05,
        random_state: Optional[int] = None,
    ):
        if task not in ("binary", "multiclass"):
            raise ValueError("`task` must be either 'binary' or 'multiclass'.")
        if (base_estimators is None) == (base_estimator_cls is None):
            raise ValueError("Specify exactly one of `base_estimators` or `base_estimator_cls`.")
        if bound_type not in self._criteria_map:
            raise ValueError(f"Unknown bound_type '{bound_type}'.")

        self.task = task
        self.base_estimators = base_estimators
        self.base_estimator_cls = base_estimator_cls
        self.base_estimator_kwargs = base_estimator_kwargs or {}
        self.bound_type = bound_type
        self.delta = bound_delta
        self.random_state = random_state

        # Will be set in fit():
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        m_values: List[int],
        n_runs: int = 10,
        max_iters: int = 200,
        tol: float = 1e-6,
    ) -> "BoundEnsemble":
        """
        Fit an ensemble of size m ∈ m_values, repeated n_runs times with different RNG seeds.
        Stores per‐seed results in self.all_runs. Selects the best (seed, m) by minimal bound.
        """
        # 1) Basic input processing
        X = np.asarray(X)
        y = np.asarray(y)
        n, d = X.shape
        r = d + 1
        n_r = n - r
        if n_r <= 0:
            raise ValueError("Training set too small: need n > d+1.")

        if self.task == "binary":
            # Map y ∈ {0,1} or {-1,1} strictly into {-1, +1}
            uniq = set(np.unique(y))
            if uniq <= {0, 1}:
                y = np.where(y == 0, -1, 1)
            elif uniq <= {-1, 1}:
                pass
            else:
                raise ValueError("For binary task, y must be in {0,1} or {-1,1}.")
            classes = None
        else:
            # multiclass: retain y as-is and record unique classes
            classes = np.unique(y)

        self.m_values = list(m_values)
        self.n_runs = n_runs

        all_runs: List[List[Dict]] = []
        best_bound = np.inf
        best_models: Optional[List[BaseEstimator]] = None
        best_rho: Optional[np.ndarray] = None
        best_m: Optional[int] = None
        best_seed: Optional[int] = None

        # 2) Loop over independent random‐state seeds
        for seed in range(n_runs):
            rng_seed = (self.random_state + seed) if (self.random_state is not None) else seed
            rng = np.random.default_rng(rng_seed)

            run_results: List[Dict] = []
            for m in self.m_values:
                t0 = time.time()
                losses = np.zeros(m, dtype=float)
                models: List[BaseEstimator] = []

                # 2a) Train m base learners on random subsets of size r=d+1
                for i in range(m):
                    idx = np.arange(n)
                    Si = rng.choice(idx, size=r, replace=False)
                    Sic = np.setdiff1d(idx, Si)  # those not in Si

                    # Build or clone a base estimator
                    if self.base_estimators is not None:
                        template = self.base_estimators[i % len(self.base_estimators)]
                    else:
                        template = self.base_estimator_cls(**self.base_estimator_kwargs)

                    clf = clone(template)
                    # If classifier supports random_state, set it to a new random int
                    if hasattr(clf, "get_params") and "random_state" in clf.get_params():
                        clf.set_params(random_state=int(rng.integers(0, 2**31 - 1)))

                    # If binary and only one class appears in Si, use a constant predictor
                    if self.task == "binary" and len(set(y[Si])) < 2:
                        const_label = int(y[Si][0])

                        class ConstantClassifier(BaseEstimator, ClassifierMixin):
                            def __init__(self, constant_label: int):
                                self.constant_label = constant_label

                            def fit(self, X_c, y_c):
                                return self

                            def predict(self, X_c):
                                return np.full(shape=(len(X_c),), fill_value=self.constant_label, dtype=int)

                        clf = ConstantClassifier(constant_label=const_label)
                        _ = clf.fit(None, None)
                    else:
                        clf.fit(X[Si], y[Si])

                    models.append(clf)
                    losses[i] = zero_one_loss(y[Sic], clf.predict(X[Sic]))

                # 2b) If tandem bound, build the m×m pairwise-loss matrix
                pair_losses = None
                if self.bound_type == "tandem":
                    # P: shape (m, n), where P[i,t] = model_i.predict(X[t])
                    P = np.vstack([m_.predict(X) for m_ in models])  # shape (m, n)
                    # err[i,t] = 1 if model i errs on sample t, else 0
                    err = (P != y).astype(float)  # works for both binary and multiclass
                    pair_losses = err @ err.T / n  # shape (m, m)

                # 3) PAC-Bayes: optimize λ and ρ
                pi = np.full(m, 1 / m, dtype=float)
                rho = pi.copy()
                lam = max(1.0 / math.sqrt(n_r), 0.5)
                crit = self._criteria_map[self.bound_type]()
                prev_bound = np.inf
                log_term = math.log(2 * math.sqrt(n_r) / self.delta)

                for _ in range(max_iters):
                    kl_rho = float((rho * np.log(rho / pi)).sum())
                    if self.bound_type == "tandem":
                        stat, bound = crit.compute(pair_losses, rho, kl_rho, n_r, self.delta, lam, n)
                    else:
                        stat, bound = crit.compute(losses, rho, kl_rho, n_r, self.delta, lam, n)

                    if abs(prev_bound - bound) < tol:
                        break
                    prev_bound = bound

                    denominator = kl_rho + log_term
                    lam = 2.0 / (math.sqrt(1 + 2 * n_r * stat / denominator) + 1)

                    shift = losses.min()
                    w = np.exp(-lam * n_r * (losses - shift))
                    rho = w / w.sum()

                # 4) Compute majority‐vote “in-sample” loss
                #    For binary, predictions are in {−1, +1}. For multiclass, in {0,1,2,…}
                Pfull = np.vstack([m_.predict(X) for m_ in models]).T  # shape (n, m)
                if self.task == "binary":
                    # Weighted sign‐vote:
                    scores = Pfull.dot(rho)  # shape (n,)
                    pred = np.where(scores >= 0, 1, -1)
                    mv_loss = float(zero_one_loss(y, pred))
                else:
                    # Multiclass: build vote‐matrix of shape (n, n_classes)
                    n_classes = len(classes)
                    votes = np.zeros((n, n_classes))
                    for k, c in enumerate(classes):
                        votes[:, k] = (Pfull == c).dot(rho)
                    pred = classes[np.argmax(votes, axis=1)]
                    mv_loss = float(zero_one_loss(y, pred))

                elapsed = time.time() - t0
                run_results.append({"m": m, "err": mv_loss, "bound": bound, "time": elapsed})

                if bound < best_bound:
                    best_bound = bound
                    best_models = [copy.deepcopy(m_) for m_ in models]
                    best_rho = rho.copy()
                    best_m = m
                    best_seed = seed

            all_runs.append(run_results)

        # 5) Save everything
        self.all_runs = all_runs
        self.best_bound_ = best_bound
        self.best_models_ = best_models
        self.best_rho_ = best_rho
        self.best_m_ = best_m
        self.best_seed_ = best_seed
        if self.task == "multiclass":
            self.classes_ = classes
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return majority-vote labels for X, using the best ensemble.
        For binary: returns array in {−1, +1}.
        For multiclass: returns in original class labels.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X)
        Pfull = np.vstack([m_.predict(X) for m_ in self.best_models_]).T  # shape (n_samples, m)
        rho = self.best_rho_
        n = X.shape[0]

        if self.task == "binary":
            scores = Pfull.dot(rho)  # shape (n,)
            return np.where(scores >= 0, 1, -1)
        else:
            n_classes = len(self.classes_)
            votes = np.zeros((n, n_classes))
            for k, c in enumerate(self.classes_):
                votes[:, k] = (Pfull == c).dot(rho)
            return self.classes_[np.argmax(votes, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Only valid when `task="binary"` and all base learners support `predict_proba`.
        Returns array shape (n_samples, 2): [P(y=−1), P(y=+1)].
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit before predict_proba().")
        if self.task != "binary":
            raise AttributeError("predict_proba is only valid for 'binary' task.")
        X = np.asarray(X)

        # Check that every best_model has predict_proba
        for clf in self.best_models_:
            if not hasattr(clf, "predict_proba"):
                raise AttributeError("One or more base estimators lack predict_proba().")

        # Gather probability of class +1 from each model
        probs_pos = np.vstack([clf.predict_proba(X)[:, 1] for clf in self.best_models_]).T  # (n, m)
        weighted_pos = probs_pos.dot(self.best_rho_)  # shape (n,)
        weighted_neg = 1.0 - weighted_pos
        return np.vstack([weighted_neg, weighted_pos]).T

    def summary(self):
        """
        Prints mean±std of MV‐loss/err, bound, and time across seeds, for each m in m_values.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before summary().")
        header = f"{'m':>4s} | {'Err±σ':>12s} | {'Bnd±σ':>12s} | {'t±σ[s]':>12s}"
        print(header)
        print("-" * len(header))
        for i, m in enumerate(self.m_values):
            arr_err = np.array([run[i]["err"] for run in self.all_runs])
            arr_bd  = np.array([run[i]["bound"] for run in self.all_runs])
            arr_tm  = np.array([run[i]["time"] for run in self.all_runs])
            print(
                f"{m:4d} | "
                f"{arr_err.mean():.4f}±{arr_err.std():.4f} | "
                f"{arr_bd.mean():.4f}±{arr_bd.std():.4f} | "
                f"{arr_tm.mean():.3f}±{arr_tm.std():.3f}"
            )

    def get_params(self, deep: bool = True) -> Dict:
        """
        Return parameters (for GridSearchCV compatibility).
        """
        params = {
            "task": self.task,
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

    def set_params(self, **kwargs):
        """
        Set parameters (for GridSearchCV compatibility).
        """
        for key, val in kwargs.items():
            setattr(self, key, val)
        return self


# ───────────────────────────────────────────────────────────────────────────────
# 4) Quick sanity‐check examples
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ----------------------------------------
    # (A) Binary‐classification example sanity-check
    # ----------------------------------------
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    # Generate a small separable binary dataset
    Xb, yb = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=0,
    )
    # Base learner templates
    base_templates_bin = [
        LogisticRegression(solver="liblinear"),
        DecisionTreeClassifier(max_depth=3),
    ]
    ens_bin = BoundEnsemble(
        task="binary",
        base_estimators=base_templates_bin,
        bound_type="tandem",
        bound_delta=0.05,
        random_state=42,
    )
    m_vals = [5, 10]
    print("Fitting binary BoundEnsembleUnified (PB-KL) on separable data...")
    ens_bin.fit(Xb, yb, m_values=m_vals, n_runs=3, max_iters=100, tol=1e-6)
    print("\nBinary summary:")
    ens_bin.summary()
    print(f"\nBest ensemble size: {ens_bin.best_m_}, best seed: {ens_bin.best_seed_}, best bound: {ens_bin.best_bound_:.4f}")

    y_pred_b_signed = ens_bin.predict(Xb)  # returns {-1, +1}
    # Convert back to {0,1} for accuracy
    y_pred_b = (y_pred_b_signed == 1).astype(int)
    acc_b = 1.0 - zero_one_loss(yb, y_pred_b)
    print(f"\nIn-sample accuracy (binary) = {acc_b:.4f}")

    # ----------------------------------------
    # (B) Multiclass‐classification example sanity-check
    # ----------------------------------------
    from sklearn.datasets import make_classification
    from sklearn.tree import DecisionTreeClassifier

    Xmc, ymc = make_classification(
        n_samples=300,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=0,
    )
    base_templates_mc = [
        DecisionTreeClassifier(max_depth=3),
    ]
    ens_mc = BoundEnsemble(
        task="multiclass",
        base_estimators=base_templates_mc,
        bound_type="tandem",
        bound_delta=0.05,
        random_state=0,
    )
    m_vals_mc = [5, 10]
    print("\nFitting multiclass BoundEnsembleUnified (PB-KL) on 3-class data...")
    ens_mc.fit(Xmc, ymc, m_values=m_vals_mc, n_runs=3, max_iters=100, tol=1e-6)
    print("\nMulticlass summary:")
    ens_mc.summary()
    print(f"\nBest ensemble size: {ens_mc.best_m_}, best seed: {ens_mc.best_seed_}, best bound: {ens_mc.best_bound_:.4f}")

    y_pred_mc = ens_mc.predict(Xmc)
    acc_mc = 1.0 - zero_one_loss(ymc, y_pred_mc)
    print(f"\nIn-sample accuracy (multiclass) = {acc_mc:.4f}")

    # Verify predicted classes are a subset of original classes
    unique_preds = np.unique(y_pred_mc)
    print(f"Unique predicted classes: {unique_preds}")
    print(f"Original classes: {ens_mc.classes_}")
    assert set(unique_preds).issubset(set(ens_mc.classes_)), "Predicted labels outside original class set!"
    print("All predicted labels are within the original classes.")