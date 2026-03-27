from __future__ import annotations

import dataclasses as dc
from typing import Dict, Optional, Protocol, Tuple

import jax.numpy as jnp
import numpy as np


def _normalize_box_bounds(
    box_bounds: Optional[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    """Backward-compatible validator for the deprecated box_bounds argument.

    Ball attacks are now defined on the Euclidean ball only. We still accept
    box_bounds so older call-sites do not break, but the projection and support
    are determined solely by the Ball-local constraint.
    """
    if box_bounds is None:
        return None
    lo, hi = float(box_bounds[0]), float(box_bounds[1])
    if hi < lo:
        raise ValueError("box_bounds must satisfy lo <= hi.")
    return (lo, hi)


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

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        out = jnp.asarray(x, dtype=jnp.float32).reshape(self.center.shape)
        return _project_ball_jax(out, self.center, self.radius).reshape(
            self.center.shape
        )

    def project_np(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=np.float32).reshape(self.center.shape)
        return _project_ball_np(out, self.center, self.radius).reshape(
            self.center.shape
        )

    def negative_log_density(self, x: jnp.ndarray) -> jnp.ndarray:
        del x
        return jnp.asarray(0.0, dtype=jnp.float32)

    def negative_log_density_np(self, x: np.ndarray) -> float:
        del x
        return 0.0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return _sample_uniform_l2_ball(self.center, self.radius, int(n), rng)

    def metadata(self) -> Dict[str, object]:
        return {
            "name": "uniform_l2_ball",
            "radius": float(self.radius),
            "center_shape": tuple(int(v) for v in self.center.shape),
            "projection": "euclidean_ball",
            "box_bounds": (
                None
                if self.box_bounds is None
                else (float(self.box_bounds[0]), float(self.box_bounds[1]))
            ),
            "box_bounds_ignored": bool(self.box_bounds is not None),
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

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        out = jnp.asarray(x, dtype=jnp.float32).reshape(self.center.shape)
        return _project_ball_jax(out, self.center, self.radius).reshape(
            self.center.shape
        )

    def project_np(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=np.float32).reshape(self.center.shape)
        return _project_ball_np(out, self.center, self.radius).reshape(
            self.center.shape
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
            "projection": "euclidean_ball",
            "box_bounds": (
                None
                if self.box_bounds is None
                else (float(self.box_bounds[0]), float(self.box_bounds[1]))
            ),
            "box_bounds_ignored": bool(self.box_bounds is not None),
        }
