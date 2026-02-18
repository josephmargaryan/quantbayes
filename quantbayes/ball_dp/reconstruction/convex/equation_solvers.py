# quantbayes/ball_dp/reconstruction/convex/equation_solvers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..types import Array, DatasetMinus, ReconstructionResult


def _augment_X(X: Array) -> Array:
    X = np.asarray(X, dtype=np.float64)
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    return np.concatenate([X, ones], axis=1)


def _softmax_rows(logits: Array) -> Array:
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def _sigmoid(x: Array) -> Array:
    # numerically stable sigmoid
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


# ============================================================
# Ridge prototypes (closed-form exact recon)
# ============================================================


@dataclass
class RidgePrototypesEquationSolver:
    """
    Implements Theorem: exact reconstruction from a noiseless ridge-prototype release.

    Assumes:
      - release is mus: (K,d)
      - labels are integers 0..K-1
      - objective matches your `fit_ridge_prototypes` mean objective
    """

    lam: float
    n_total: int

    def reconstruct(
        self,
        *,
        release_mus: Array,  # (K,d)
        d_minus: DatasetMinus,  # (X_minus, y_minus)
        label_known: Optional[int] = None,
    ) -> ReconstructionResult:
        X_minus, y_minus = d_minus
        mus = np.asarray(release_mus, dtype=np.float64)
        y_minus = np.asarray(y_minus).reshape(-1)

        if mus.ndim != 2:
            return ReconstructionResult(
                None, None, "invalid_input", {"reason": "release_mus must be 2D (K,d)"}
            )

        K, d = mus.shape
        if X_minus.shape[1] != d:
            return ReconstructionResult(
                None, None, "invalid_input", {"reason": "X_minus dim mismatch vs mus"}
            )

        if label_known is None:
            # noiseless label-identification rule from your theorem:
            # find unique class where mu_c differs from mu_c^- (degeneracy aside)
            diffs = []
            for c in range(K):
                idx = np.where(y_minus == c)[0]
                n_c_minus = int(idx.size)
                S_c_minus = (
                    np.sum(X_minus[idx], axis=0)
                    if n_c_minus > 0
                    else np.zeros((d,), dtype=np.float64)
                )
                denom = 2.0 * float(n_c_minus) + float(self.lam) * float(self.n_total)
                mu_c_minus = (
                    (2.0 * S_c_minus) / denom if denom > 0 else np.zeros_like(S_c_minus)
                )
                diffs.append(np.linalg.norm(mus[c] - mu_c_minus))
            diffs = np.asarray(diffs)
            y_hat = int(np.argmax(diffs))  # robust: pick most-changed class
        else:
            y_hat = int(label_known)

        idx = np.where(y_minus == y_hat)[0]
        n_y_minus = int(idx.size)
        S_y_minus = (
            np.sum(X_minus[idx], axis=0)
            if n_y_minus > 0
            else np.zeros((d,), dtype=np.float64)
        )

        denom = 2.0 * float(n_y_minus + 1) + float(self.lam) * float(self.n_total)
        z_hat = (denom / 2.0) * mus[y_hat] - S_y_minus

        return ReconstructionResult(
            record_hat=z_hat.astype(np.float64),
            label_hat=y_hat,
            status="ok",
            details={"n_y_minus": n_y_minus, "denom": denom},
        )


# ============================================================
# Softmax linear (multiclass) exact recon from noiseless optimum
# ============================================================


@dataclass
class SoftmaxEquationSolver:
    """
    Implements Theorem: exact reconstruction from noiseless softmax-ERM release.

    Requires bias augmentation (include_bias=True) for exact identifiability.
    """

    lam: float
    n_total: int
    include_bias: bool = True
    batch_size: int = 8192

    def _missing_gradient(
        self,
        *,
        W: Array,  # (K,d)
        b: Array,  # (K,)
        d_minus: DatasetMinus,
    ) -> Array:
        X_minus, y_minus = d_minus
        X_minus = np.asarray(X_minus, dtype=np.float64)
        y_minus = np.asarray(y_minus, dtype=np.int64).reshape(-1)

        if not self.include_bias:
            raise ValueError(
                "SoftmaxEquationSolver requires include_bias=True for exact reconstruction (bias augmentation)."
            )

        Xt = _augment_X(X_minus)  # (n-1, d+1)
        Wt = np.concatenate(
            [
                np.asarray(W, dtype=np.float64),
                np.asarray(b, dtype=np.float64).reshape(-1, 1),
            ],
            axis=1,
        )

        K, d1 = Wt.shape
        if Xt.shape[1] != d1:
            raise ValueError("Dimension mismatch between augmented X and augmented W.")

        # Sum gradients over known records in batches:
        sum_grad = np.zeros_like(Wt)
        n = Xt.shape[0]
        bs = int(self.batch_size)

        for s in range(0, n, bs):
            e = min(n, s + bs)
            Xb = Xt[s:e]
            yb = y_minus[s:e]

            logits = Xb @ Wt.T  # (B,K)
            p = _softmax_rows(logits)  # (B,K)

            diff = p
            diff[np.arange(diff.shape[0]), yb] -= 1.0  # (B,K)

            # sum_i (diff_i outer X_i) = diff^T @ X
            sum_grad += diff.T @ Xb

        G_missing = -float(self.lam) * float(self.n_total) * Wt - sum_grad
        return G_missing

    def reconstruct(
        self,
        *,
        W: Array,
        b: Array,
        d_minus: DatasetMinus,
    ) -> ReconstructionResult:
        try:
            G = self._missing_gradient(W=W, b=b, d_minus=d_minus)
        except Exception as e:
            return ReconstructionResult(None, None, "failed", {"exception": repr(e)})

        a = G[:, -1].copy()  # last column = a = p(e) - e_y
        y_hat = int(
            np.argmin(a)
        )  # in noiseless case, this is the unique negative entry

        # pick a stable row c != y with largest positive a_c
        mask = np.ones_like(a, dtype=bool)
        mask[y_hat] = False
        a_pos = a.copy()
        a_pos[~mask] = -np.inf
        c = int(np.argmax(a_pos))

        if not np.isfinite(a[c]) or abs(a[c]) < 1e-15:
            return ReconstructionResult(
                None,
                y_hat,
                "failed",
                {"reason": "could not find stable non-y row for factorization"},
            )

        et_hat = (G[c, :] / a[c]).astype(np.float64)

        # enforce last coordinate 1 (scale ambiguity under noise)
        if abs(et_hat[-1]) < 1e-15:
            return ReconstructionResult(
                None,
                y_hat,
                "failed",
                {"reason": "augmented coord is ~0; cannot rescale"},
            )
        et_hat = et_hat / et_hat[-1]
        e_hat = et_hat[:-1]

        return ReconstructionResult(
            record_hat=e_hat,
            label_hat=y_hat,
            status="ok",
            details={"a": a, "chosen_row": c, "G_missing": G},
        )


# ============================================================
# Binary logistic exact recon from noiseless optimum
# ============================================================


@dataclass
class BinaryLogisticEquationSolver:
    """
    Implements Theorem: exact reconstruction from noiseless binary-logistic ERM release.

    Labels must be in {-1,+1}. Bias augmentation is required for the clean sign rule.
    """

    lam: float
    n_total: int
    batch_size: int = 16384

    def _missing_gradient(
        self,
        *,
        w: Array,  # (d,)
        b: float,  # scalar
        d_minus: DatasetMinus,
    ) -> Array:
        X_minus, y_minus = d_minus
        X_minus = np.asarray(X_minus, dtype=np.float64)
        y_minus = np.asarray(y_minus, dtype=np.int64).reshape(-1)

        Xt = _augment_X(X_minus)  # (n-1, d+1)
        wt = np.concatenate(
            [
                np.asarray(w, dtype=np.float64).reshape(-1),
                np.asarray([b], dtype=np.float64),
            ],
            axis=0,
        )

        if Xt.shape[1] != wt.size:
            raise ValueError("Dimension mismatch between augmented X and augmented w.")

        sum_grad = np.zeros_like(wt)
        n = Xt.shape[0]
        bs = int(self.batch_size)

        for s in range(0, n, bs):
            e = min(n, s + bs)
            Xb = Xt[s:e]
            yb = y_minus[s:e].astype(np.float64)

            t = Xb @ wt  # (B,)
            # grad = -y * sigmoid(-y t) * x
            coef = -yb * _sigmoid(-yb * t)  # (B,)
            sum_grad += (coef[:, None] * Xb).sum(axis=0)

        g_missing = -float(self.lam) * float(self.n_total) * wt - sum_grad
        return g_missing

    def reconstruct(
        self,
        *,
        w: Array,
        b: float,
        d_minus: DatasetMinus,
    ) -> ReconstructionResult:
        try:
            g = self._missing_gradient(w=w, b=b, d_minus=d_minus)
        except Exception as e:
            return ReconstructionResult(None, None, "failed", {"exception": repr(e)})

        a = float(g[-1])
        if abs(a) < 1e-15:
            return ReconstructionResult(
                None,
                None,
                "failed",
                {"reason": "a ~ 0; cannot factorize", "g_missing": g},
            )

        y_hat = -1 if a > 0 else +1  # since sign(a) = -y in the theorem
        et_hat = (g / a).astype(np.float64)
        if abs(et_hat[-1]) < 1e-15:
            return ReconstructionResult(
                None, y_hat, "failed", {"reason": "augmented coord ~0"}
            )
        et_hat = et_hat / et_hat[-1]
        e_hat = et_hat[:-1]

        return ReconstructionResult(
            record_hat=e_hat,
            label_hat=int(y_hat),
            status="ok",
            details={"a": a, "g_missing": g},
        )


# ============================================================
# Squared-hinge SVM exact recon for support vectors
# ============================================================


@dataclass
class SquaredHingeEquationSolver:
    """
    Implements Theorem: conditional exact reconstruction for squared hinge SVM.

    If missing record is not a support vector (margin >= 1), missing gradient is 0 and reconstruction is impossible.
    """

    lam: float
    n_total: int
    batch_size: int = 16384
    zero_tol: float = 1e-12

    def _missing_gradient(
        self,
        *,
        w: Array,
        b: float,
        d_minus: DatasetMinus,
    ) -> Array:
        X_minus, y_minus = d_minus
        X_minus = np.asarray(X_minus, dtype=np.float64)
        y_minus = np.asarray(y_minus, dtype=np.int64).reshape(-1)

        Xt = _augment_X(X_minus)
        wt = np.concatenate(
            [
                np.asarray(w, dtype=np.float64).reshape(-1),
                np.asarray([b], dtype=np.float64),
            ],
            axis=0,
        )

        sum_grad = np.zeros_like(wt)
        n = Xt.shape[0]
        bs = int(self.batch_size)

        for s in range(0, n, bs):
            e = min(n, s + bs)
            Xb = Xt[s:e]
            yb = y_minus[s:e].astype(np.float64)  # ±1

            t = Xb @ wt
            margin = 1.0 - yb * t
            hinge = np.maximum(0.0, margin)  # (B,)
            coef = -2.0 * yb * hinge  # (B,)
            sum_grad += (coef[:, None] * Xb).sum(axis=0)

        g_missing = -float(self.lam) * float(self.n_total) * wt - sum_grad
        return g_missing

    def reconstruct(
        self,
        *,
        w: Array,
        b: float,
        d_minus: DatasetMinus,
    ) -> ReconstructionResult:
        try:
            g = self._missing_gradient(w=w, b=b, d_minus=d_minus)
        except Exception as e:
            return ReconstructionResult(None, None, "failed", {"exception": repr(e)})

        if float(np.linalg.norm(g)) <= float(self.zero_tol):
            return ReconstructionResult(
                record_hat=None,
                label_hat=None,
                status="no_support_vector",
                details={
                    "reason": "missing gradient is ~0 (margin>=1), cannot reconstruct",
                    "g_missing": g,
                },
            )

        a = float(g[-1])
        if abs(a) < 1e-15:
            return ReconstructionResult(
                None,
                None,
                "failed",
                {"reason": "a ~ 0; cannot factorize", "g_missing": g},
            )

        y_hat = -1 if a > 0 else +1  # sign(a) = -y
        et_hat = (g / a).astype(np.float64)
        et_hat = et_hat / et_hat[-1]
        e_hat = et_hat[:-1]

        return ReconstructionResult(
            record_hat=e_hat,
            label_hat=int(y_hat),
            status="ok",
            details={"a": a, "g_missing": g},
        )
