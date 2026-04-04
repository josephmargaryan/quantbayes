from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Optional, Sequence

import numpy as np

from ..types import (
    AttackResult,
    DpCertificate,
    PriorFamily,
    RdpCurve,
    ReRoPoint,
    ReRoReport,
    ReleaseArtifact,
)
from .direct_poisson import compute_ball_sgd_direct_rero_report

_STD_NORMAL = NormalDist()


def _normalize_rero_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    mapping = {
        "auto": "auto",
        "dp": "dp",
        "rdp": "rdp",
        "gaussian_direct": "gaussian_direct",
        "ball_sgd_direct": "ball_sgd_direct",
        "sgd_direct": "ball_sgd_direct",
        "poisson_direct": "ball_sgd_direct",
        "poisson_ball_sgd_direct": "ball_sgd_direct",
    }
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(
            "mode must be one of {'auto', 'dp', 'rdp', 'gaussian_direct', 'ball_sgd_direct'}."
        ) from exc


def _save_rero_report(report: ReRoReport, out_path: str | Path | None) -> None:
    if out_path is None:
        return
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "mode": report.mode,
                "metadata": report.metadata,
                "points": [point.__dict__ for point in report.points],
            },
            indent=2,
            sort_keys=True,
        )
    )


class UniformL2BallPrior:
    def __init__(self, *, center: np.ndarray, radius: float, dimension: int):
        self.center = np.asarray(center, dtype=np.float32).reshape(-1)
        self.radius = float(radius)
        self.dimension = int(dimension)

    def kappa(self, eta: float) -> float:
        eta = float(eta)
        if eta <= 0.0:
            return 0.0
        if eta >= self.radius:
            return 1.0
        return float((eta / self.radius) ** self.dimension)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        g = rng.normal(size=(int(n), self.dimension)).astype(np.float32)
        g = g / np.linalg.norm(g, axis=1, keepdims=True)
        u = rng.random(int(n), dtype=np.float32) ** (1.0 / self.dimension)
        return self.center[None, :] + self.radius * u[:, None] * g


class EmpiricalBallPrior:
    """Uniform prior on a finite empirical support set.

    This class is useful for exploratory attacker-prior experiments, but it is
    NOT theorem-faithful for Ball-ReRo by default.

    Why:
      The Ball-ReRo theorem needs
          kappa(eta) = sup_{z0} P[d(Z, z0) <= eta].
      A common empirical shortcut is to test only centers z0 drawn from the
      sample support itself. That can strictly *underestimate* the true
      supremum, which would make the reported Ball-ReRo upper bound
      spuriously optimistic.

    Therefore:
      - safe_kappa_mode="raise"        -> default, refuse certified Ball-ReRo
      - safe_kappa_mode="trivial_upper"-> return kappa = 1.0 (sound but vacuous)

    For exploratory work only, call `sample_point_kappa_lower_bound(eta)`.
    """

    def __init__(self, samples: np.ndarray, *, safe_kappa_mode: str = "raise"):
        self.samples = np.asarray(samples, dtype=np.float32)
        if self.samples.ndim != 2:
            raise ValueError("EmpiricalBallPrior expects shape (n_samples, dim).")

        mode = str(safe_kappa_mode).lower()
        if mode not in {"raise", "trivial_upper"}:
            raise ValueError(
                "safe_kappa_mode must be one of {'raise', 'trivial_upper'}."
            )
        self.safe_kappa_mode = mode

    def sample_point_kappa_lower_bound(self, eta: float) -> float:
        """Exploratory only: lower bound on the true kappa.

        This searches only centers located at empirical support points, so it can
        be strictly smaller than the theorem-required supremum over all centers.
        Do NOT use this quantity inside a certified Ball-ReRo bound.
        """
        eta = float(eta)
        if eta <= 0.0:
            return 0.0

        dists = np.linalg.norm(
            self.samples[:, None, :] - self.samples[None, :, :], axis=-1
        )
        mass = (dists <= eta).mean(axis=1)
        return float(np.max(mass))

    def kappa(self, eta: float) -> float:
        """Theorem-safe kappa interface used by Ball-ReRo."""
        if self.safe_kappa_mode == "raise":
            raise ValueError(
                "EmpiricalBallPrior does not provide a theorem-faithful kappa by default. "
                "The Ball-ReRo theorem requires kappa(eta) = sup_{z0} P[d(Z, z0) <= eta], "
                "and the common empirical support-point shortcut can underestimate it. "
                "Use UniformL2BallPrior for certified Ball-ReRo, FiniteExactIdentificationPrior "
                "for finite-prior exact identification, or set safe_kappa_mode='trivial_upper' "
                "if you explicitly want the vacuous but sound bound kappa=1."
            )

        # Sound for the theorem, but usually vacuous.
        return 1.0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.choice(self.samples.shape[0], size=int(n), replace=True)
        return self.samples[idx]


class FiniteExactIdentificationPrior:
    """Theorem-faithful finite prior for exact candidate identification.

    This prior is intended for the discrete exact-identification setting with the
    0/1 loss
        rho_{0/1}(z, z') = 1[z != z'].
    For every threshold eta < 1, the theorem-required anti-concentration term is
        kappa(eta) = max_i pi_i,
    where pi_i are the candidate prior weights. This class implements exactly
    that quantity.

    Notes
    -----
    - The stored `samples` are present only so the object satisfies the same
      sampling interface as the other prior helpers. The theorem-backed kappa
      does not depend on the sample geometry in the exact-identification regime.
    - For eta >= 1, the event rho_{0/1}(Z, z0) <= eta is always true, so
      kappa(eta) = 1.
    """

    def __init__(
        self, samples: np.ndarray, *, weights: Optional[Sequence[float]] = None
    ):
        self.samples = np.asarray(samples, dtype=np.float32)
        if self.samples.ndim != 2:
            raise ValueError(
                "FiniteExactIdentificationPrior expects shape (n_candidates, dim)."
            )
        n = int(self.samples.shape[0])
        if n <= 0:
            raise ValueError(
                "FiniteExactIdentificationPrior requires at least one candidate."
            )

        if weights is None:
            probs = np.full((n,), 1.0 / float(n), dtype=np.float64)
        else:
            probs = np.asarray(tuple(float(v) for v in weights), dtype=np.float64)
            if probs.shape != (n,):
                raise ValueError(f"weights must have shape ({n},), got {probs.shape}.")
            if np.any(~np.isfinite(probs)) or np.any(probs <= 0.0):
                raise ValueError("weights must be finite and strictly positive.")
            probs = probs / float(np.sum(probs))

        self.weights = probs.astype(np.float64, copy=False)

    def kappa(self, eta: float) -> float:
        eta = float(eta)
        if eta < 1.0:
            return float(np.max(self.weights))
        return 1.0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.choice(
            self.samples.shape[0],
            size=int(n),
            replace=True,
            p=self.weights,
        )
        return self.samples[idx]


def _dp_bound(cert: DpCertificate, kappa: float) -> float:
    """Stable computation of min(1, exp(epsilon) * kappa + delta)."""
    kappa = float(kappa)
    delta = float(cert.delta)
    eps = float(cert.epsilon)

    if kappa <= 0.0:
        return float(min(1.0, delta))
    if delta >= 1.0:
        return 1.0

    threshold = max(1.0 - delta, 0.0)
    if threshold <= 0.0:
        return 1.0

    log_term = eps + math.log(kappa)
    if log_term >= math.log(threshold):
        return 1.0

    return float(min(1.0, math.exp(log_term) + delta))


def _rdp_bound(curve: RdpCurve, kappa: float) -> tuple[float, Optional[float]]:
    kappa = float(kappa)
    if kappa <= 0.0:
        return 0.0, None

    log_kappa = math.log(kappa)
    best = 1.0
    best_alpha = None

    for alpha, eps in zip(curve.orders, curve.epsilons):
        alpha = float(alpha)
        eps = float(eps)
        exponent = (alpha - 1.0) / alpha

        log_candidate = exponent * (log_kappa + eps)

        if log_candidate >= 0.0:
            candidate = 1.0
        else:
            candidate = math.exp(log_candidate)

        if candidate < best:
            best = float(candidate)
            best_alpha = float(alpha)

    return float(best), best_alpha


def gaussian_direct_ball_rero_bound(
    *,
    kappa: float,
    sensitivity: float,
    sigma: float,
) -> float:
    """Direct Ball-ReRo bound for Gaussian output perturbation.

    Returns
        Phi(Phi^{-1}(kappa) + sensitivity / sigma)
    with the boundary cases interpreted by continuity.
    """
    kappa = float(kappa)
    sensitivity = float(sensitivity)
    sigma = float(sigma)

    if not math.isfinite(sensitivity) or sensitivity < 0.0:
        raise ValueError("sensitivity must be finite and >= 0.")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and > 0.")
    if kappa <= 0.0:
        return 0.0
    if kappa >= 1.0:
        return 1.0

    z = _STD_NORMAL.inv_cdf(kappa) + sensitivity / sigma
    return float(min(1.0, max(0.0, _STD_NORMAL.cdf(z))))


def compute_ball_rero_report(
    release: ReleaseArtifact,
    prior: PriorFamily,
    eta_grid: Sequence[float],
    *,
    mode: str = "auto",
    out_path: str | Path | None = None,
) -> ReRoReport:
    mode = _normalize_rero_mode(mode)

    if mode == "auto":
        # Prefer the full Ball-RDP curve whenever it is available. This is the
        # theorem-backed path used by the optimized gamma_ball^RDP conversion in
        # the paper; falling back to the DP certificate would prematurely commit
        # to a single epsilon/delta point and lose the alpha optimization.
        if release.privacy.ball.rdp_curve is not None:
            mode = "rdp"
        elif release.privacy.ball.dp_certificates:
            mode = "dp"
        else:
            raise ValueError(
                "Release artifact carries neither a Ball-DP certificate nor a Ball-RDP curve."
            )

    points = []
    if mode == "dp":
        ball_cert = release.privacy.ball.dp_certificates[0]
        std_cert = (
            release.privacy.standard.dp_certificates[0]
            if release.privacy.standard.dp_certificates
            else None
        )
        for eta in eta_grid:
            kappa = float(prior.kappa(float(eta)))
            gamma_ball = _dp_bound(ball_cert, kappa)
            gamma_std = None if std_cert is None else _dp_bound(std_cert, kappa)
            points.append(
                ReRoPoint(
                    eta=float(eta),
                    kappa=kappa,
                    gamma_ball=float(gamma_ball),
                    gamma_standard=None if gamma_std is None else float(gamma_std),
                )
            )
    elif mode == "rdp":
        ball_curve = release.privacy.ball.rdp_curve
        std_curve = release.privacy.standard.rdp_curve
        if ball_curve is None:
            raise ValueError("Ball RDP curve missing.")
        for eta in eta_grid:
            kappa = float(prior.kappa(float(eta)))
            gamma_ball, alpha_ball = _rdp_bound(ball_curve, kappa)
            gamma_std = None
            alpha_std = None
            if std_curve is not None:
                gamma_std, alpha_std = _rdp_bound(std_curve, kappa)
            points.append(
                ReRoPoint(
                    eta=float(eta),
                    kappa=kappa,
                    gamma_ball=float(gamma_ball),
                    gamma_standard=None if gamma_std is None else float(gamma_std),
                    alpha_opt_ball=alpha_ball,
                    alpha_opt_standard=alpha_std,
                )
            )
    elif mode == "ball_sgd_direct":
        report = compute_ball_sgd_direct_rero_report(
            release,
            prior,
            eta_grid=eta_grid,
        )
        _save_rero_report(report, out_path)
        return report
    elif mode == "gaussian_direct":
        mechanism = str(getattr(release.privacy.ball, "mechanism", ""))
        sigma = release.privacy.ball.sigma
        delta_ball = release.sensitivity.delta_ball
        delta_std = release.sensitivity.delta_std

        if mechanism != "gaussian_output_perturbation":
            raise ValueError(
                "mode='gaussian_direct' is theorem-backed only for Gaussian output perturbation releases."
            )
        if sigma is None or float(sigma) <= 0.0:
            raise ValueError(
                "mode='gaussian_direct' requires a positive Gaussian output noise scale sigma."
            )
        if delta_ball is None:
            raise ValueError(
                "mode='gaussian_direct' requires release.sensitivity.delta_ball."
            )

        for eta in eta_grid:
            kappa = float(prior.kappa(float(eta)))
            gamma_ball = gaussian_direct_ball_rero_bound(
                kappa=kappa,
                sensitivity=float(delta_ball),
                sigma=float(sigma),
            )
            gamma_std = None
            if delta_std is not None:
                gamma_std = gaussian_direct_ball_rero_bound(
                    kappa=kappa,
                    sensitivity=float(delta_std),
                    sigma=float(sigma),
                )
            points.append(
                ReRoPoint(
                    eta=float(eta),
                    kappa=kappa,
                    gamma_ball=float(gamma_ball),
                    gamma_standard=None if gamma_std is None else float(gamma_std),
                )
            )
    else:
        raise ValueError(mode)

    report = ReRoReport(
        mode=mode,
        points=points,
        metadata={
            "radius": release.privacy.ball.radius,
            "release_kind": release.release_kind,
            "ball_mechanism": release.privacy.ball.mechanism,
            "sigma": release.privacy.ball.sigma,
            "delta_ball": release.sensitivity.delta_ball,
            "delta_standard": release.sensitivity.delta_std,
        },
    )
    _save_rero_report(report, out_path)
    return report


def summarize_attack_trials(
    attack_results: Sequence[AttackResult],
    *,
    eta_grid: Optional[Sequence[float]] = None,
    oblivious_kappa: Optional[float] = None,
) -> dict[str, float]:
    """Aggregate theorem-aligned attack metrics across repeated trials.

    Primary metrics:
      - exact identification success, when available,
      - thresholded success curves based on the per-trial `success@eta` keys,
      - the oblivious baseline kappa.

    Secondary descriptive metrics such as MSE and feature-space distances are
    reported separately and are not combined with the theorem-backed gamma bounds.
    """
    if not attack_results:
        raise ValueError("attack_results must be non-empty.")

    summary: dict[str, float] = {
        "n_trials": float(len(attack_results)),
    }

    def _collect_metric(name: str) -> list[float]:
        out: list[float] = []
        for res in attack_results:
            if name in res.metrics:
                val = float(res.metrics[name])
                if math.isfinite(val):
                    out.append(val)
        return out

    exact_vals = _collect_metric("exact_identification_success")
    if not exact_vals:
        exact_vals = _collect_metric("prior_exact_hit")
    if exact_vals:
        summary["exact_identification_success"] = float(np.mean(exact_vals))

    success_keys: list[tuple[float, str]] = []
    if eta_grid is not None:
        for eta in eta_grid:
            key = f"success@{float(eta):g}"
            success_keys.append((float(eta), key))
    else:
        seen = set()
        for res in attack_results:
            for key in res.metrics.keys():
                if key.startswith("success@"):
                    try:
                        eta = float(key.split("@", 1)[1])
                    except Exception:
                        continue
                    if key not in seen:
                        seen.add(key)
                        success_keys.append((eta, key))
        success_keys.sort(key=lambda item: item[0])

    for eta, key in success_keys:
        vals = _collect_metric(key)
        if vals:
            summary[f"p_succ@{eta:g}"] = float(np.mean(vals))

    kappa_vals = _collect_metric("oblivious_kappa")
    if oblivious_kappa is not None:
        summary["oblivious_kappa"] = float(oblivious_kappa)
    elif kappa_vals:
        summary["oblivious_kappa"] = float(np.mean(kappa_vals))

    for name in [
        "distance",
        "feature_l2",
        "mse",
        "label_correct",
        "objective_gap_to_truth",
    ]:
        vals = _collect_metric(name)
        if vals:
            summary[f"mean_{name}"] = float(np.mean(vals))

    return summary
