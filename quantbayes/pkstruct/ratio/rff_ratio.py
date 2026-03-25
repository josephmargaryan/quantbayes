# quantbayes/pkstruct/ratio/rff_ratio.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def median_heuristic_lengthscale_1d(
    y: np.ndarray, *, max_pairs: int = 200_000, seed: int = 0
) -> float:
    """
    Median heuristic for an RBF lengthscale on 1D data.
    Uses median(|y_i - y_j|) over a random subset of pairs.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y.size
    if n < 2:
        return 1.0

    rng = np.random.default_rng(seed)
    m = min(max_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n, size=m)
    d = np.abs(y[i] - y[j])
    med = float(np.median(d))
    return max(med, 1e-3)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable sigmoid
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _logmeanexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    m = float(np.max(x))
    return float(m + np.log(np.mean(np.exp(x - m))))


@dataclass(frozen=True)
class RFFRatioModel1D:
    """
    Reference-free density-ratio estimator for 1D y:

      logit(y) ≈ log(q(y)/p(y)) + const

    We store:
      - RFF params (omega, phase) in *normalized* y-space
      - linear head weights (w, b)
      - affine normalization y_norm = (y - shift)/scale
      - calibration constant log_c so that E_{y~p}[exp(log_ratio(y))] ≈ 1

    log_ratio(y) := logit(y) - log_c
    d/dy log_ratio(y) = d/dy logit(y)  (constant cancels)
    """

    omega: np.ndarray  # (m,)
    phase: np.ndarray  # (m,)
    w: np.ndarray  # (m,)
    b: float

    y_shift: float
    y_scale: float

    log_c: float = 0.0  # calibration constant
    eps: float = 1e-12  # numerical

    @property
    def m(self) -> int:
        return int(self.omega.shape[0])

    def _yn(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        return (y - self.y_shift) / (self.y_scale + self.eps)

    def features(self, y: np.ndarray) -> np.ndarray:
        """
        Cosine RFF features for 1D:
          psi(y) = sqrt(2/m) * cos(omega * y_norm + phase)
        """
        yn = self._yn(y).reshape(-1, 1)  # (n,1)
        om = self.omega.reshape(1, -1)  # (1,m)
        ph = self.phase.reshape(1, -1)  # (1,m)
        s = np.sqrt(2.0 / float(self.m))
        return s * np.cos(yn * om + ph)  # (n,m)

    def dfeatures_dy(self, y: np.ndarray) -> np.ndarray:
        """
        d/dy psi(y) = (1/scale) * d/dy_norm psi(y_norm)
                    = (1/scale) * sqrt(2/m) * (-sin(omega*y_norm + phase)) * omega
        """
        yn = self._yn(y).reshape(-1, 1)
        om = self.omega.reshape(1, -1)
        ph = self.phase.reshape(1, -1)
        s = np.sqrt(2.0 / float(self.m))
        dpsi_dyn = s * (-np.sin(yn * om + ph)) * om  # (n,m)
        return dpsi_dyn / (self.y_scale + self.eps)

    def logit(self, y: np.ndarray) -> np.ndarray:
        Phi = self.features(y)  # (n,m)
        return Phi @ self.w.reshape(-1, 1) + float(self.b)

    def log_ratio(self, y: np.ndarray) -> np.ndarray:
        return self.logit(y) - float(self.log_c)

    def dlog_ratio_dy(self, y: np.ndarray) -> np.ndarray:
        dPhi = self.dfeatures_dy(y)  # (n,m)
        return dPhi @ self.w.reshape(-1, 1)

    def calibrate_log_c(self, y_neg: np.ndarray) -> "RFFRatioModel1D":
        """
        Choose log_c so that E_{y~p_neg}[exp(log_ratio(y))] ≈ 1
        => log_c := log E_{y~p_neg}[exp(logit(y))]
        """
        logits = self.logit(y_neg).reshape(-1)
        log_c = _logmeanexp(logits)
        return RFFRatioModel1D(
            omega=self.omega,
            phase=self.phase,
            w=self.w,
            b=float(self.b),
            y_shift=float(self.y_shift),
            y_scale=float(self.y_scale),
            log_c=float(log_c),
            eps=float(self.eps),
        )

    def save_npz(self, path: str) -> None:
        np.savez(
            path,
            omega=self.omega,
            phase=self.phase,
            w=self.w,
            b=np.array([self.b], dtype=np.float64),
            y_shift=np.array([self.y_shift], dtype=np.float64),
            y_scale=np.array([self.y_scale], dtype=np.float64),
            log_c=np.array([self.log_c], dtype=np.float64),
            eps=np.array([self.eps], dtype=np.float64),
        )

    @staticmethod
    def load_npz(path: str) -> "RFFRatioModel1D":
        obj = np.load(path)
        return RFFRatioModel1D(
            omega=np.asarray(obj["omega"], dtype=np.float64),
            phase=np.asarray(obj["phase"], dtype=np.float64),
            w=np.asarray(obj["w"], dtype=np.float64),
            b=float(np.asarray(obj["b"]).reshape(-1)[0]),
            y_shift=float(np.asarray(obj["y_shift"]).reshape(-1)[0]),
            y_scale=float(np.asarray(obj["y_scale"]).reshape(-1)[0]),
            log_c=float(np.asarray(obj["log_c"]).reshape(-1)[0]),
            eps=float(np.asarray(obj["eps"]).reshape(-1)[0]),
        )

    def to_jax(self) -> "RFFRatioModel1DJax":
        import jax.numpy as jnp

        return RFFRatioModel1DJax(
            omega=jnp.asarray(self.omega),
            phase=jnp.asarray(self.phase),
            w=jnp.asarray(self.w),
            b=float(self.b),
            y_shift=float(self.y_shift),
            y_scale=float(self.y_scale),
            log_c=float(self.log_c),
            eps=float(self.eps),
        )


@dataclass(frozen=True)
class RFFRatioModel1DJax:
    """
    JAX-compatible version for guidance.

    Methods:
      log_ratio(y): scalar
      dlog_ratio_dy(y): scalar
    """

    omega: "object"  # jax.Array
    phase: "object"  # jax.Array
    w: "object"  # jax.Array
    b: float

    y_shift: float
    y_scale: float
    log_c: float = 0.0
    eps: float = 1e-12

    def _yn(self, y):
        import jax.numpy as jnp

        return (jnp.asarray(y) - self.y_shift) / (self.y_scale + self.eps)

    def logit(self, y):
        import jax.numpy as jnp

        s = jnp.sqrt(2.0 / self.omega.shape[0])
        yn = self._yn(y)
        feats = s * jnp.cos(self.omega * yn + self.phase)  # (m,)
        return jnp.dot(feats, self.w) + self.b

    def log_ratio(self, y):
        return self.logit(y) - self.log_c

    def dlog_ratio_dy(self, y):
        import jax.numpy as jnp

        s = jnp.sqrt(2.0 / self.omega.shape[0])
        yn = self._yn(y)
        # d/dy: (1/scale) * s * (-sin(...)) * omega
        dfeat_dy = (s * (-jnp.sin(self.omega * yn + self.phase)) * self.omega) / (
            self.y_scale + self.eps
        )
        return jnp.dot(dfeat_dy, self.w)


def fit_rff_logistic_ratio_1d(
    y_pos: np.ndarray,
    y_neg: np.ndarray,
    *,
    m: int = 256,
    lengthscale: Optional[float] = None,
    reg: float = 1e-4,
    lr: float = 2e-2,
    num_steps: int = 6000,
    batch_size: int = 2048,
    seed: int = 0,
    standardize: bool = True,
    print_every: int = 500,
) -> RFFRatioModel1D:
    """
    Fit a 1D RFF logistic classifier to distinguish:
      y_pos ~ q  (label 1)
      y_neg ~ p  (label 0)

    Returns a calibrated ratio model approximating log(q/p).

    Notes:
      - Training uses minibatch Adam in NumPy (no sklearn dependency).
      - The learned logit is only defined up to an additive constant; we calibrate log_c
        via y_neg to satisfy E_p[exp(log_ratio)] ≈ 1.
    """
    rng = np.random.default_rng(seed)

    y_pos = np.asarray(y_pos, dtype=np.float64).reshape(-1)
    y_neg = np.asarray(y_neg, dtype=np.float64).reshape(-1)
    if y_pos.size < 50 or y_neg.size < 50:
        raise ValueError(
            "Need at least ~50 samples per class for stable ratio fitting."
        )

    # Combine for normalization only
    y_all = np.concatenate([y_pos, y_neg], axis=0)
    if standardize:
        y_shift = float(np.mean(y_all))
        y_scale = float(np.std(y_all))
        y_scale = max(y_scale, 1e-6)
    else:
        y_shift, y_scale = 0.0, 1.0

    y_pos_n = (y_pos - y_shift) / y_scale
    y_neg_n = (y_neg - y_shift) / y_scale

    if lengthscale is None:
        ls = median_heuristic_lengthscale_1d(
            np.concatenate([y_pos_n, y_neg_n]), seed=seed
        )
        lengthscale = float(ls)

    m = int(m)
    # RFF for RBF kernel: omega ~ N(0, 1/ls^2), b ~ Uniform(0,2pi)
    omega = rng.normal(loc=0.0, scale=1.0 / lengthscale, size=(m,)).astype(np.float64)
    phase = rng.uniform(low=0.0, high=2.0 * np.pi, size=(m,)).astype(np.float64)

    s = np.sqrt(2.0 / float(m))

    def feats(batch_y_norm: np.ndarray) -> np.ndarray:
        # batch_y_norm: (B,)
        yy = batch_y_norm.reshape(-1, 1)  # (B,1)
        return s * np.cos(yy * omega.reshape(1, -1) + phase.reshape(1, -1))  # (B,m)

    # Dataset
    y = np.concatenate([y_pos_n, y_neg_n], axis=0)
    lab = np.concatenate([np.ones_like(y_pos_n), np.zeros_like(y_neg_n)], axis=0)

    n = y.size
    if batch_size > n:
        batch_size = n

    # Params
    w = np.zeros((m,), dtype=np.float64)
    b = 0.0

    # Adam state
    mw = np.zeros_like(w)
    vw = np.zeros_like(w)
    mb = 0.0
    vb = 0.0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    for step in range(1, int(num_steps) + 1):
        idx = rng.integers(0, n, size=(int(batch_size),))
        yb = y[idx]
        lb = lab[idx]

        Phi = feats(yb)  # (B,m)
        logits = Phi @ w + b  # (B,)
        p = _sigmoid(logits)  # (B,)

        # grad of BCE: d/dlogit = (p - y)
        g = (p - lb) / float(batch_size)  # (B,)

        grad_w = Phi.T @ g + reg * w
        grad_b = float(np.sum(g))

        # Adam updates
        mw = beta1 * mw + (1.0 - beta1) * grad_w
        vw = beta2 * vw + (1.0 - beta2) * (grad_w * grad_w)
        mb = beta1 * mb + (1.0 - beta1) * grad_b
        vb = beta2 * vb + (1.0 - beta2) * (grad_b * grad_b)

        mw_hat = mw / (1.0 - beta1**step)
        vw_hat = vw / (1.0 - beta2**step)
        mb_hat = mb / (1.0 - beta1**step)
        vb_hat = vb / (1.0 - beta2**step)

        w = w - lr * mw_hat / (np.sqrt(vw_hat) + eps)
        b = b - lr * mb_hat / (np.sqrt(vb_hat) + eps)

        if (print_every > 0) and (step % int(print_every) == 0 or step == 1):
            # quick diagnostic on a small random batch
            loss = float(
                np.mean(np.log1p(np.exp(np.clip(logits, -40, 40))) - lb * logits)
                + 0.5 * reg * np.sum(w * w)
            )
            acc = float(np.mean((p >= 0.5) == (lb >= 0.5)))
            print(f"[ratio-fit] step={step:05d}  loss={loss:.5f}  acc={acc:.3f}")

    model = RFFRatioModel1D(
        omega=omega,
        phase=phase,
        w=w,
        b=float(b),
        y_shift=float(y_shift),
        y_scale=float(y_scale),
        log_c=0.0,
        eps=1e-12,
    )

    # Calibrate constant using NEG samples
    # (calibration uses unnormalized logit on original y, not normalized: model handles shift/scale internally)
    model = model.calibrate_log_c(y_neg)

    return model
