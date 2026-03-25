# quantbayes/ball_dp/attacks/ball_priors.py

from __future__ import annotations

import dataclasses as dc
from typing import Dict, Optional, Protocol, Tuple

from jax import lax
import jax.numpy as jnp
import numpy as np


def _normalize_box_bounds(
    box_bounds: Optional[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    if box_bounds is None:
        return None
    lo, hi = float(box_bounds[0]), float(box_bounds[1])
    if hi < lo:
        raise ValueError("box_bounds must satisfy lo <= hi.")
    return (lo, hi)


def _project_box_jax(
    x: jnp.ndarray,
    box_bounds: Optional[Tuple[float, float]],
) -> jnp.ndarray:
    out = jnp.asarray(x, dtype=jnp.float32)
    if box_bounds is None:
        return out
    lo, hi = float(box_bounds[0]), float(box_bounds[1])
    return jnp.clip(out, lo, hi)


def _project_box_np(
    x: np.ndarray,
    box_bounds: Optional[Tuple[float, float]],
) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32)
    if box_bounds is None:
        return out.astype(np.float32, copy=False)
    lo, hi = float(box_bounds[0]), float(box_bounds[1])
    return np.clip(out, lo, hi).astype(np.float32, copy=False)


def _project_ball_jax(
    x: jnp.ndarray,
    center: np.ndarray,
    radius: float,
) -> jnp.ndarray:
    out = jnp.asarray(x, dtype=jnp.float32)
    ctr = jnp.asarray(center, dtype=out.dtype).reshape(out.shape)
    r = float(radius)
    if r < 0.0:
        raise ValueError("radius must be >= 0.")
    if r == 0.0:
        return ctr

    diff = out - ctr
    norm = jnp.linalg.norm(jnp.ravel(diff))
    scale = jnp.minimum(
        jnp.asarray(1.0, dtype=out.dtype),
        jnp.asarray(r, dtype=out.dtype)
        / jnp.maximum(norm, jnp.asarray(1e-12, dtype=out.dtype)),
    )
    return ctr + scale * diff


def _project_ball_np(
    x: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32)
    ctr = np.asarray(center, dtype=np.float32).reshape(out.shape)
    r = float(radius)
    if r < 0.0:
        raise ValueError("radius must be >= 0.")
    if r == 0.0:
        return ctr.astype(np.float32, copy=False)

    diff = out - ctr
    norm = float(np.linalg.norm(diff.reshape(-1), ord=2))
    if norm <= r or norm <= 1e-12:
        return out.astype(np.float32, copy=False)
    return (ctr + (r / norm) * diff).astype(np.float32, copy=False)


def _box_ball_intersection_nonempty(
    center: np.ndarray,
    radius: float,
    box_bounds: Optional[Tuple[float, float]],
    *,
    atol: float = 1e-8,
) -> bool:
    if box_bounds is None:
        return True
    ctr = np.asarray(center, dtype=np.float64)
    lo, hi = float(box_bounds[0]), float(box_bounds[1])
    closest = np.clip(ctr, lo, hi)
    dist = float(np.linalg.norm((closest - ctr).reshape(-1), ord=2))
    return dist <= float(radius) + float(atol)


def _project_box_ball_intersection_np(
    x: np.ndarray,
    center: np.ndarray,
    radius: float,
    box_bounds: Optional[Tuple[float, float]],
    *,
    max_iters: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64)
    ctr = np.asarray(center, dtype=np.float64).reshape(out.shape)
    r = float(radius)

    if r < 0.0:
        raise ValueError("radius must be >= 0.")

    if box_bounds is None:
        return _project_ball_np(out, ctr, r)

    if not _box_ball_intersection_nonempty(ctr, r, box_bounds, atol=tol):
        raise ValueError("The feasible set box ∩ ball is empty.")

    if r == 0.0:
        return ctr.astype(np.float32, copy=False)

    lo, hi = float(box_bounds[0]), float(box_bounds[1])

    # Step 1: exact projection onto the box.
    y = np.clip(out, lo, hi)
    if float(np.linalg.norm((y - ctr).reshape(-1), ord=2)) <= r + float(tol):
        return y.astype(np.float32, copy=False)

    # Step 2: active-ball case.
    # KKT form:
    #   z_i(lambda) = clip((x_i + lambda * c_i) / (1 + lambda), lo, hi)
    # Use the numerically stable convex-combination form to avoid overflow.
    def z_of_lambda(lam: float) -> np.ndarray:
        beta = 1.0 / (1.0 + lam)
        alpha = 1.0 - beta
        base = beta * out + alpha * ctr
        return np.clip(base, lo, hi)

    # Bracket the unique root.
    lam_lo = 0.0
    lam_hi = 1.0
    for _ in range(int(max_iters)):
        z_hi = z_of_lambda(lam_hi)
        if float(np.linalg.norm((z_hi - ctr).reshape(-1), ord=2)) <= r:
            break
        lam_hi *= 2.0
    else:
        raise RuntimeError(
            "Failed to bracket the KKT multiplier for box ∩ ball projection."
        )

    # Fixed-iteration bisection.
    for _ in range(int(max_iters)):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        z_mid = z_of_lambda(lam_mid)
        if float(np.linalg.norm((z_mid - ctr).reshape(-1), ord=2)) > r:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

    return z_of_lambda(lam_hi).astype(np.float32, copy=False)


def _project_box_ball_intersection_jax(
    x: jnp.ndarray,
    center: np.ndarray,
    radius: float,
    box_bounds: Optional[Tuple[float, float]],
    *,
    max_iters: int = 100,
    tol: float = 1e-6,
) -> jnp.ndarray:
    out = jnp.asarray(x, dtype=jnp.float32)
    ctr = jnp.asarray(center, dtype=out.dtype).reshape(out.shape)
    r = float(radius)

    if r < 0.0:
        raise ValueError("radius must be >= 0.")

    if box_bounds is None:
        return _project_ball_jax(out, ctr, r)

    # Safe here because center/radius/box_bounds are static prior parameters
    # in the current code path.
    if not _box_ball_intersection_nonempty(np.asarray(center), r, box_bounds, atol=tol):
        raise ValueError("The feasible set box ∩ ball is empty.")

    if r == 0.0:
        return ctr

    lo, hi = float(box_bounds[0]), float(box_bounds[1])

    # Step 1: exact projection onto the box.
    y0 = jnp.clip(out, lo, hi)
    r_j = jnp.asarray(r, dtype=out.dtype)
    tol_j = jnp.asarray(tol, dtype=out.dtype)

    def radius_from_center(z: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(jnp.ravel(z - ctr))

    inside0 = radius_from_center(y0) <= r_j + tol_j

    def solve_active(_: None) -> jnp.ndarray:
        # Numerically stable version of
        # (out + lam * ctr) / (1 + lam)
        def z_of_lambda(lam: jnp.ndarray) -> jnp.ndarray:
            one = jnp.asarray(1.0, dtype=out.dtype)
            beta = one / (one + lam)
            alpha = one - beta
            base = beta * out + alpha * ctr
            return jnp.clip(base, lo, hi)

        # Fixed-iteration bracketing.
        def bracket_body(_: int, lam_hi: jnp.ndarray) -> jnp.ndarray:
            z_hi = z_of_lambda(lam_hi)
            need_more = radius_from_center(z_hi) > r_j
            return jnp.where(need_more, lam_hi * 2.0, lam_hi)

        lam_hi0 = jnp.asarray(1.0, dtype=out.dtype)
        lam_hi = lax.fori_loop(0, int(max_iters), bracket_body, lam_hi0)

        # Fixed-iteration bisection.
        def bisect_body(
            _: int,
            state: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            lam_lo, lam_hi = state
            lam_mid = 0.5 * (lam_lo + lam_hi)
            z_mid = z_of_lambda(lam_mid)
            too_far = radius_from_center(z_mid) > r_j
            new_lam_lo = jnp.where(too_far, lam_mid, lam_lo)
            new_lam_hi = jnp.where(too_far, lam_hi, lam_mid)
            return (new_lam_lo, new_lam_hi)

        lam_lo0 = jnp.asarray(0.0, dtype=out.dtype)
        _, lam_hi = lax.fori_loop(
            0,
            int(max_iters),
            bisect_body,
            (lam_lo0, lam_hi),
        )
        return z_of_lambda(lam_hi)

    return lax.cond(inside0, lambda _: y0, solve_active, operand=None)


def _sample_uniform_l2_ball(
    center: np.ndarray,
    radius: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    center = np.asarray(center, dtype=np.float32)
    n = int(n)
    if n <= 0:
        return np.zeros((0,) + center.shape, dtype=np.float32)

    flat_center = center.reshape(-1)
    d = int(flat_center.size)
    r = float(radius)
    if r < 0.0:
        raise ValueError("radius must be >= 0.")
    if d == 0 or r == 0.0:
        return np.repeat(center[None, ...], n, axis=0).astype(np.float32, copy=False)

    g = rng.normal(size=(n, d)).astype(np.float32)
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    u = rng.random(n).astype(np.float32) ** (1.0 / float(d))
    out = flat_center[None, :] + r * u[:, None] * g
    return out.reshape((n,) + center.shape).astype(np.float32, copy=False)


class BallAttackPrior(Protocol):
    center: np.ndarray
    radius: float
    box_bounds: Optional[Tuple[float, float]]

    def project(self, x: jnp.ndarray) -> jnp.ndarray: ...
    def project_np(self, x: np.ndarray) -> np.ndarray: ...
    def negative_log_density(self, x: jnp.ndarray) -> jnp.ndarray: ...
    def negative_log_density_np(self, x: np.ndarray) -> float: ...
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray: ...
    def metadata(self) -> Dict[str, object]: ...


@dc.dataclass(frozen=True)
class UniformBallAttackPrior:
    center: np.ndarray
    radius: float
    box_bounds: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float32)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", float(self.radius))
        object.__setattr__(self, "box_bounds", _normalize_box_bounds(self.box_bounds))
        if float(self.radius) < 0.0:
            raise ValueError("radius must be >= 0.")
        if not _box_ball_intersection_nonempty(
            center, float(self.radius), self.box_bounds
        ):
            raise ValueError(
                "The feasible set box ∩ ball is empty for the provided "
                "center, radius, and box_bounds."
            )

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        out = jnp.asarray(x, dtype=jnp.float32).reshape(self.center.shape)
        return _project_box_ball_intersection_jax(
            out,
            self.center,
            self.radius,
            self.box_bounds,
        ).reshape(self.center.shape)

    def project_np(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=np.float32).reshape(self.center.shape)
        return (
            _project_box_ball_intersection_np(
                out,
                self.center,
                self.radius,
                self.box_bounds,
            )
            .reshape(self.center.shape)
            .astype(np.float32, copy=False)
        )

    def negative_log_density(self, x: jnp.ndarray) -> jnp.ndarray:
        del x
        return jnp.asarray(0.0, dtype=jnp.float32)

    def negative_log_density_np(self, x: np.ndarray) -> float:
        del x
        return 0.0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raw = _sample_uniform_l2_ball(self.center, self.radius, int(n), rng)
        if self.box_bounds is None:
            return raw
        return np.stack([self.project_np(xi) for xi in raw], axis=0)

    def metadata(self) -> Dict[str, object]:
        return {
            "name": "uniform_l2_ball",
            "radius": float(self.radius),
            "center_shape": tuple(int(v) for v in self.center.shape),
            "projection": "box_intersection_ball_kkt_bisection",
            "box_bounds": (
                None
                if self.box_bounds is None
                else (float(self.box_bounds[0]), float(self.box_bounds[1]))
            ),
        }


@dc.dataclass(frozen=True)
class TruncatedGaussianBallAttackPrior:
    center: np.ndarray
    radius: float
    sigma: float
    box_bounds: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float32)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", float(self.radius))
        object.__setattr__(self, "sigma", float(self.sigma))
        object.__setattr__(self, "box_bounds", _normalize_box_bounds(self.box_bounds))
        if float(self.radius) < 0.0:
            raise ValueError("radius must be >= 0.")
        if float(self.sigma) <= 0.0:
            raise ValueError("sigma must be > 0.")
        if not _box_ball_intersection_nonempty(
            center, float(self.radius), self.box_bounds
        ):
            raise ValueError(
                "The feasible set box ∩ ball is empty for the provided "
                "center, radius, and box_bounds."
            )

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        out = jnp.asarray(x, dtype=jnp.float32).reshape(self.center.shape)
        return _project_box_ball_intersection_jax(
            out,
            self.center,
            self.radius,
            self.box_bounds,
        ).reshape(self.center.shape)

    def project_np(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=np.float32).reshape(self.center.shape)
        return (
            _project_box_ball_intersection_np(
                out,
                self.center,
                self.radius,
                self.box_bounds,
            )
            .reshape(self.center.shape)
            .astype(np.float32, copy=False)
        )

    def negative_log_density(self, x: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(x, dtype=jnp.float32).reshape(self.center.shape)
        ctr = jnp.asarray(self.center, dtype=arr.dtype)
        diff = arr - ctr
        return 0.5 * jnp.sum(diff * diff) / (float(self.sigma) * float(self.sigma))

    def negative_log_density_np(self, x: np.ndarray) -> float:
        arr = np.asarray(x, dtype=np.float32).reshape(self.center.shape)
        ctr = np.asarray(self.center, dtype=np.float32)
        diff = arr - ctr
        return float(
            0.5
            * np.sum(diff.reshape(-1) ** 2)
            / (float(self.sigma) * float(self.sigma))
        )

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        n = int(n)
        center = np.asarray(self.center, dtype=np.float32)
        flat_center = center.reshape(-1)
        d = int(flat_center.size)
        if n <= 0:
            return np.zeros((0,) + center.shape, dtype=np.float32)

        raw = flat_center[None, :] + float(self.sigma) * rng.normal(size=(n, d)).astype(
            np.float32
        )
        raw = raw.reshape((n,) + center.shape).astype(np.float32, copy=False)
        return np.stack([self.project_np(xi) for xi in raw], axis=0)

    def metadata(self) -> Dict[str, object]:
        return {
            "name": "truncated_gaussian_l2_ball",
            "radius": float(self.radius),
            "sigma": float(self.sigma),
            "center_shape": tuple(int(v) for v in self.center.shape),
            "projection": "box_intersection_ball_kkt_bisection",
            "box_bounds": (
                None
                if self.box_bounds is None
                else (float(self.box_bounds[0]), float(self.box_bounds[1]))
            ),
        }
