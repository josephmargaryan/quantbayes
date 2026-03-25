# quantbayes/ball_dp/convex/sensitivity.py
from __future__ import annotations

import math


def lz_prototypes_exact() -> float:
    return 2.0


def output_sensitivity_upper(
    *, lz: float, lam: float, n_total: int, radius: float
) -> float:
    if lam <= 0.0:
        raise ValueError("lam must be > 0.")
    if n_total <= 0:
        raise ValueError("n_total must be positive.")
    if radius < 0.0:
        raise ValueError("radius must be >= 0.")
    return float((lz * radius) / (lam * n_total))


def approximate_solution_sensitivity(
    *, exact_sensitivity: float, parameter_error_bound: float
) -> float:
    if exact_sensitivity < 0.0 or parameter_error_bound < 0.0:
        raise ValueError("Sensitivities / error bounds must be >= 0.")
    return float(exact_sensitivity + 2.0 * parameter_error_bound)


def standard_radius_from_embedding_bound(B: float | None) -> float | None:
    if B is None:
        return None
    if B < 0.0:
        raise ValueError("Embedding bound B must be >= 0.")
    return float(2.0 * B)


def lz_softmax_linear_bound(
    *, B: float, lam: float, include_bias: bool = True
) -> float:
    if B < 0.0:
        raise ValueError("B must be >= 0.")
    if lam <= 0.0:
        raise ValueError("lam must be > 0.")
    b2 = float(B * B + (1.0 if include_bias else 0.0))
    return float(math.sqrt(2.0) * (1.0 + b2 / (2.0 * lam)))


def lz_binary_logistic_bound(
    *, B: float, lam: float, include_bias: bool = True
) -> float:
    """Binary logistic L_z bound with optional bias augmentation.

    The theorem-backed bound is:
        L_z <= 1 + \tilde B^2 / (4 lam),
    where \tilde B^2 = B^2 + 1 when include_bias=True.
    """
    if B < 0.0:
        raise ValueError("B must be >= 0.")
    if lam <= 0.0:
        raise ValueError("lam must be > 0.")
    b2 = float(B * B + (1.0 if include_bias else 0.0))
    return float(1.0 + b2 / (4.0 * lam))


def lz_squared_hinge_bound(*, B: float, lam: float, include_bias: bool = True) -> float:
    """Squared-hinge L_z bound.

    If include_bias=True, this is the augmented-feature version:
        L_z <= 2 + 4 * sqrt(B^2 + 1) * sqrt(2 / lam).

    If include_bias=False, it reduces to:
        L_z <= 2 + 4 * B * sqrt(2 / lam).
    """
    if B < 0.0:
        raise ValueError("B must be >= 0.")
    if lam <= 0.0:
        raise ValueError("lam must be > 0.")
    B_eff = math.sqrt(B * B + (1.0 if include_bias else 0.0))
    return float(2.0 + 4.0 * B_eff * math.sqrt(2.0 / lam))
