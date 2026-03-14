# quantbayes/ball_dp/evaluation/rero.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..types import (
    DpCertificate,
    PriorFamily,
    RdpCurve,
    ReRoPoint,
    ReRoReport,
    ReleaseArtifact,
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
                "Use UniformL2BallPrior for certified Ball-ReRo, or set "
                "safe_kappa_mode='trivial_upper' if you explicitly want the vacuous but "
                "sound bound kappa=1."
            )

        # Sound for the theorem, but usually vacuous.
        return 1.0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.choice(self.samples.shape[0], size=int(n), replace=True)
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


def compute_ball_rero_report(
    release: ReleaseArtifact,
    prior: PriorFamily,
    eta_grid: Sequence[float],
    *,
    mode: str = "auto",
    out_path: str | Path | None = None,
) -> ReRoReport:
    if mode == "auto":
        if release.privacy.ball.dp_certificates:
            mode = "dp"
        elif release.privacy.ball.rdp_curve is not None:
            mode = "rdp"
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
    else:
        raise ValueError(mode)

    report = ReRoReport(
        mode=mode,
        points=points,
        metadata={
            "radius": release.privacy.ball.radius,
            "release_kind": release.release_kind,
        },
    )
    if out_path is not None:
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
    return report
