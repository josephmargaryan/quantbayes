# cohort_dp/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import hashlib
import numpy as np

from .metrics import Metric
from .accountant import PrivacyAccountant
from .api import CohortDiscoveryAPI
from .candidates import AllCandidates, E2LSHCandidates, PrototypeCandidates
from .baselines import NonPrivateKNNRetriever
from .mechanisms import ExponentialMechanismRetriever, NoisyTopKRetriever
from .mechanisms_topk import OneshotLaplaceTopKRetriever
from .novel_mechanisms import (
    AdaptiveBallUniformRetriever,
    AdaptiveBallExponentialRetriever,
    AdaptiveBallMixedRetriever,
    AdaptiveBallOptimalSkewRetriever,
)


def _stable_int(s: str, mod: int = 10_000_000) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % mod


@dataclass(frozen=True)
class MechanismSpec:
    """
    kind:
      NONPRIVATE | EM | NOISY_TOPK | LAPLACE_TOPK | ABU | ABE | ABM | ABO

    Notes:
      - eps_total is used for DP baselines (EM/NOISY/LAPLACE) and also as a sweep knob for ABM/ABE if gamma is None.
      - eps_ball is used by ABO (AB-Optimal). If eps_ball is None, we fall back to eps_total.
      - gamma is optional for ABE/ABM (draft-style sharpness knob).
    """

    kind: str
    k_service: int = 10
    eps_total: float = 0.0
    k0: int = 20
    mix_uniform: float = 0.7

    gamma: Optional[float] = None
    eps_ball: Optional[float] = None

    name: Optional[str] = None

    def label(self) -> str:
        if self.name is not None:
            return self.name
        k = self.kind.upper()
        if k == "NONPRIVATE":
            return "NonPrivate kNN"
        if k == "EM":
            return f"EM(k={self.k_service}) eps={self.eps_total:g}"
        if k == "NOISY_TOPK":
            return f"NoisyTopK(k={self.k_service}) eps={self.eps_total:g}"
        if k == "LAPLACE_TOPK":
            return f"LaplaceTopK(k={self.k_service}) eps={self.eps_total:g}"
        if k == "ABU":
            return f"AB-Uniform k0={self.k0}"
        if k == "ABE":
            if self.gamma is not None:
                return f"AB-Exp k0={self.k0} gamma={self.gamma:g}"
            return f"AB-Exp k0={self.k0} eps={self.eps_total:g}"
        if k == "ABM":
            if self.gamma is not None:
                return (
                    f"AB-Mix k0={self.k0} mix={self.mix_uniform:g} gamma={self.gamma:g}"
                )
            return (
                f"AB-Mix k0={self.k0} mix={self.mix_uniform:g} eps={self.eps_total:g}"
            )
        if k == "ABO":
            eb = self.eps_ball if self.eps_ball is not None else self.eps_total
            return f"AB-Optimal k0={self.k0} eps_ball={eb:g}"
        return f"{self.kind}"

    def fingerprint(self) -> str:
        return (
            f"kind={self.kind.upper()}|k={self.k_service}|eps={self.eps_total}|"
            f"k0={self.k0}|mix={self.mix_uniform}|gamma={self.gamma}|eps_ball={self.eps_ball}|name={self.name}"
        )


def seed_from_spec(base_seed: int, spec: MechanismSpec, namespace: str = "") -> int:
    tag = f"{namespace}::{spec.fingerprint()}"
    return int(base_seed) + _stable_int(tag)


def build_candidate_generator(
    mode: str,
    X: np.ndarray,
    metric: Metric,
    seed: int,
    *,
    min_candidates: int = 200,
    fallback_random: int = 500,
    # LSH defaults
    lsh_L: int = 10,
    lsh_K: int = 8,
    lsh_w: float = 2.0,
    # Proto defaults
    proto_n_clusters: int = 50,
    proto_n_iters: int = 25,
    proto_n_probe: int = 3,
):
    mode = str(mode).lower().strip()
    rng = np.random.default_rng(seed)
    if mode == "all":
        return AllCandidates(n=X.shape[0])
    if mode == "lsh":
        return E2LSHCandidates(
            X=X,
            L=lsh_L,
            K=lsh_K,
            w=lsh_w,
            rng=rng,
            min_candidates=min_candidates,
            fallback_random=fallback_random,
        )
    if mode == "proto":
        return PrototypeCandidates(
            X=X,
            n_clusters=proto_n_clusters,
            n_iters=proto_n_iters,
            n_probe=proto_n_probe,
            rng=rng,
            metric=metric,
            min_candidates=min_candidates,
            fallback_random=fallback_random,
        )
    raise ValueError("candidate generator mode must be one of: all | lsh | proto")


def build_retriever(
    spec: MechanismSpec, X: np.ndarray, metric: Metric, r: float, seed: int
):
    rng = np.random.default_rng(seed)
    kind = spec.kind.upper().strip()

    if kind == "NONPRIVATE":
        return NonPrivateKNNRetriever(X=X, metric=metric)

    if kind == "EM":
        return ExponentialMechanismRetriever(
            X=X, metric=metric, r=r, eps_total=spec.eps_total, rng=rng
        )

    if kind == "NOISY_TOPK":
        return NoisyTopKRetriever(
            X=X, metric=metric, r=r, eps_total=spec.eps_total, rng=rng
        )

    if kind == "LAPLACE_TOPK":
        return OneshotLaplaceTopKRetriever(
            X=X, metric=metric, r=r, eps_total=spec.eps_total, rng=rng
        )

    if kind == "ABU":
        return AdaptiveBallUniformRetriever(
            X=X, metric=metric, k0=spec.k0, rng=rng, eps_total=0.0
        )

    if kind == "ABE":
        return AdaptiveBallExponentialRetriever(
            X=X,
            metric=metric,
            k0=spec.k0,
            eps_total=spec.eps_total,
            rng=rng,
            gamma=spec.gamma,
        )

    if kind == "ABM":
        return AdaptiveBallMixedRetriever(
            X=X,
            metric=metric,
            k0=spec.k0,
            eps_total=spec.eps_total,
            mix_uniform=spec.mix_uniform,
            rng=rng,
            gamma=spec.gamma,
        )

    if kind == "ABO":
        eb = spec.eps_ball if spec.eps_ball is not None else spec.eps_total
        return AdaptiveBallOptimalSkewRetriever(
            X=X,
            metric=metric,
            k0=spec.k0,
            eps_ball=float(eb),
            rng=rng,
        )

    raise ValueError(f"Unknown spec.kind: {spec.kind}")


def build_api(
    spec: MechanismSpec,
    X_db: np.ndarray,
    metric: Metric,
    r: float,
    *,
    seed: int,
    eps_budget: float = 1e9,
    candidate_generator=None,
    candidate_mode: Optional[str] = "all",
    no_repeat: bool = False,
    sticky_policy=None,
) -> CohortDiscoveryAPI:
    retriever = build_retriever(spec, X_db, metric, r=r, seed=seed)

    if candidate_generator is None and candidate_mode is not None:
        candidate_generator = build_candidate_generator(
            candidate_mode, X_db, metric, seed=seed + 1
        )

    return CohortDiscoveryAPI(
        retriever=retriever,
        accountant=PrivacyAccountant(eps_budget=eps_budget),
        candidate_generator=candidate_generator,
        no_repeat=no_repeat,
        sticky_policy=sticky_policy,
    )
