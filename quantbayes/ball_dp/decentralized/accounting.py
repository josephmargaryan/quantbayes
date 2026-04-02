from __future__ import annotations

import dataclasses as dc
from typing import Any, Optional, Sequence, Literal

import numpy as np

from ..accountants.rdp import RdpCurve, rdp_to_dp
from ..accountants.subsampled_gaussian import (
    build_ball_sgd_rdp_ledgers,
    step_delta_ball_from_schedule,
    step_delta_standard_from_schedule,
)
from ..types import DpCertificate, DualPrivacyLedger


_DEFAULT_ORDERS = (2.0, 3.0, 4.0, 5.0, 8.0, 16.0, 32.0, 64.0, 128.0)


@dc.dataclass(frozen=True)
class LocalNodeSGDSchedule:
    """Attacked-node local schedule for the public-transcript bridge theorem.

    This is intentionally the *local* schedule of the attacked node only. Under the
    public-transcript decentralized theorem, arbitrary decentralized message passing,
    consensus, and local post-processing after the noisy local releases do not add any
    privacy cost beyond this node-local subsampled Gaussian accountant.
    """

    dataset_size: int
    batch_sizes: tuple[int, ...]
    clip_norms: tuple[float, ...]
    noise_stds: tuple[float, ...]
    radius: float
    lz: Optional[float] = None
    orders: tuple[float, ...] = _DEFAULT_ORDERS
    batch_sampler: str = "poisson"
    accountant_subsampling: str = "match_sampler"
    dp_delta: Optional[float] = None

    def __post_init__(self) -> None:
        n = int(self.dataset_size)
        if n <= 0:
            raise ValueError("dataset_size must be positive.")
        if float(self.radius) < 0.0:
            raise ValueError("radius must be >= 0.")
        if len(self.batch_sizes) == 0:
            raise ValueError("batch_sizes must be non-empty.")
        if not (len(self.batch_sizes) == len(self.clip_norms) == len(self.noise_stds)):
            raise ValueError(
                "batch_sizes, clip_norms, and noise_stds must have the same length."
            )
        if any(int(v) <= 0 for v in self.batch_sizes):
            raise ValueError("Every batch size must be positive.")
        if any(int(v) > n for v in self.batch_sizes):
            raise ValueError(
                "Every local batch size must be <= the attacked node's local dataset size."
            )
        if any(float(v) < 0.0 for v in self.clip_norms):
            raise ValueError("Every clip norm must be >= 0.")
        if any(float(v) <= 0.0 for v in self.noise_stds):
            raise ValueError("Every noise std must be > 0 for private accounting.")
        if self.lz is not None and float(self.lz) < 0.0:
            raise ValueError("lz must be >= 0 when provided.")
        if self.dp_delta is not None and not (0.0 < float(self.dp_delta) < 1.0):
            raise ValueError("dp_delta must lie in (0,1) when provided.")
        if any(float(a) <= 1.0 for a in self.orders):
            raise ValueError("All RDP orders must exceed 1.")

    @classmethod
    def from_noise_multipliers(
        cls,
        *,
        dataset_size: int,
        batch_sizes: Sequence[int],
        clip_norms: Sequence[float],
        noise_multipliers: Sequence[float] | float,
        radius: float,
        lz: Optional[float] = None,
        orders: Sequence[int | float] = _DEFAULT_ORDERS,
        batch_sampler: str = "poisson",
        accountant_subsampling: str = "match_sampler",
        dp_delta: Optional[float] = None,
    ) -> "LocalNodeSGDSchedule":
        batch_sizes_t = tuple(int(v) for v in batch_sizes)
        clip_norms_t = tuple(float(v) for v in clip_norms)
        if isinstance(noise_multipliers, (int, float)):
            noise_mult_t = tuple(float(noise_multipliers) for _ in batch_sizes_t)
        else:
            noise_mult_t = tuple(float(v) for v in noise_multipliers)
        if len(noise_mult_t) != len(batch_sizes_t):
            raise ValueError(
                "noise_multipliers must be scalar or match the step schedule length."
            )
        noise_stds = tuple(
            float(c) * float(m) for c, m in zip(clip_norms_t, noise_mult_t)
        )
        return cls(
            dataset_size=int(dataset_size),
            batch_sizes=batch_sizes_t,
            clip_norms=clip_norms_t,
            noise_stds=noise_stds,
            radius=float(radius),
            lz=None if lz is None else float(lz),
            orders=tuple(float(a) for a in orders),
            batch_sampler=str(batch_sampler),
            accountant_subsampling=str(accountant_subsampling),
            dp_delta=None if dp_delta is None else float(dp_delta),
        )


@dc.dataclass(frozen=True)
class PublicTranscriptAccountantResult:
    attacked_node: Optional[int]
    ledger: DualPrivacyLedger
    metadata: dict[str, Any] = dc.field(default_factory=dict)


@dc.dataclass(frozen=True)
class ObserverSpecificAccountantResult:
    """Observer-specific Ball-PN-RDP result for a linear Gaussian view.

    `sensitivity_sq` is the theorem-side quantity Δ_{A<-j}(r)^2 or a sound upper
    bound on it, depending on `exact` and `method`.
    """

    attacked_node: Optional[int]
    observer: Any
    sensitivity_sq: float
    exact: bool
    method: str
    rdp_curve: RdpCurve
    dp_certificate: Optional[DpCertificate]
    metadata: dict[str, Any] = dc.field(default_factory=dict)


def _as_float_tuple(values: Sequence[int | float]) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def _append_decentralized_notes(
    ledgers: DualPrivacyLedger,
    *,
    attacked_node: Optional[int],
) -> DualPrivacyLedger:
    note = (
        "Decentralized public-transcript bridge theorem: the full public transcript "
        "and any final released decentralized state are post-processing of the attacked "
        "node's local Poisson-subsampled Gaussian releases."
    )
    if attacked_node is not None:
        note = f"attacked_node={int(attacked_node)}. " + note

    ledgers.ball.mechanism = (
        f"decentralized_public_transcript::{ledgers.ball.mechanism}"
    )
    ledgers.standard.mechanism = (
        f"decentralized_public_transcript::{ledgers.standard.mechanism}"
    )
    ledgers.ball.notes.append(note)
    ledgers.standard.notes.append(note)
    return ledgers


def account_public_transcript_node_local_rdp(
    schedule: LocalNodeSGDSchedule,
    *,
    attacked_node: Optional[int] = None,
    step_delta_ball: Optional[Sequence[float]] = None,
    step_delta_standard: Optional[Sequence[float]] = None,
) -> PublicTranscriptAccountantResult:
    """Theorem-backed public-transcript decentralized Ball-SGD accountant.

    This is the bridge theorem implementation:
      - only the attacked node's local schedule matters,
      - decentralized consensus/mixing/message passing adds no extra privacy cost
        once the local noisy releases are public,
      - the resulting certificate is exactly the centralized Ball-SGD accountant
        applied to the attacked node's local parameters.

    Parameters
    ----------
    schedule:
        Local schedule of the attacked node only.
    attacked_node:
        Optional identifier stored in the result metadata.
    step_delta_ball:
        Optional explicit Ball sensitivity schedule. When omitted, the code uses the
        theorem-side bound min(L_z r, 2 C_t) whenever schedule.lz is available.
    step_delta_standard:
        Optional explicit standard replacement-adjacent sensitivity schedule.
        When omitted, the code uses 2 C_t.
    """
    if step_delta_ball is None:
        step_delta_ball = step_delta_ball_from_schedule(
            clip_schedule=schedule.clip_norms,
            lz=schedule.lz,
            radius=float(schedule.radius),
        )
    else:
        step_delta_ball = _as_float_tuple(step_delta_ball)

    if step_delta_standard is None:
        step_delta_standard = step_delta_standard_from_schedule(
            clip_schedule=schedule.clip_norms,
        )
    else:
        step_delta_standard = _as_float_tuple(step_delta_standard)

    ledgers = build_ball_sgd_rdp_ledgers(
        orders=schedule.orders,
        step_batch_sizes=schedule.batch_sizes,
        dataset_size=int(schedule.dataset_size),
        step_clip_norms=schedule.clip_norms,
        step_noise_stds=schedule.noise_stds,
        step_delta_ball=step_delta_ball,
        step_delta_std=step_delta_standard,
        radius=float(schedule.radius),
        dp_delta=schedule.dp_delta,
        batch_sampler=str(schedule.batch_sampler),
        accountant_subsampling=str(schedule.accountant_subsampling),
    )
    ledgers = _append_decentralized_notes(ledgers, attacked_node=attacked_node)
    return PublicTranscriptAccountantResult(
        attacked_node=None if attacked_node is None else int(attacked_node),
        ledger=ledgers,
        metadata={
            "dataset_size": int(schedule.dataset_size),
            "num_steps": int(len(schedule.batch_sizes)),
            "radius": float(schedule.radius),
            "lz": None if schedule.lz is None else float(schedule.lz),
            "batch_sampler": str(schedule.batch_sampler),
            "accountant_subsampling": str(schedule.accountant_subsampling),
        },
    )


def _validate_block_sensitivities(
    block_sensitivities: Sequence[float],
) -> np.ndarray:
    delta = np.asarray(tuple(float(v) for v in block_sensitivities), dtype=np.float64)
    if delta.ndim != 1 or delta.size == 0:
        raise ValueError(
            "block_sensitivities must be a non-empty one-dimensional sequence."
        )
    if np.any(~np.isfinite(delta)) or np.any(delta < 0.0):
        raise ValueError("block_sensitivities must be finite and >= 0.")
    return delta


def _validate_transfer_matrix(
    transfer_matrix: np.ndarray,
    *,
    q_expected: int,
) -> np.ndarray:
    H = np.asarray(transfer_matrix, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError("transfer_matrix must have shape (d_A, q).")
    if H.shape[1] != int(q_expected):
        raise ValueError(
            f"transfer_matrix has q={H.shape[1]}, but block_sensitivities has length {q_expected}."
        )
    return H


def _validate_spd(matrix: np.ndarray, *, name: str) -> np.ndarray:
    M = np.asarray(matrix, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if not np.allclose(M, M.T, atol=1e-10, rtol=0.0):
        raise ValueError(f"{name} must be symmetric.")
    try:
        np.linalg.cholesky(M)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite.") from exc
    return M


def _observer_gram_kron_eye(
    transfer_matrix: np.ndarray,
    covariance_time: np.ndarray,
) -> np.ndarray:
    H = np.asarray(transfer_matrix, dtype=np.float64)
    Sigma_t = _validate_spd(covariance_time, name="covariance_time")
    return H.T @ np.linalg.solve(Sigma_t, H)


def _scalar_vertex_exact(
    gram: np.ndarray,
    delta: np.ndarray,
) -> tuple[float, np.ndarray]:
    q = int(delta.size)
    best_val = -float("inf")
    best_sign = np.ones((q,), dtype=np.float64)
    for mask in range(1 << q):
        sign = np.ones((q,), dtype=np.float64)
        for i in range(q):
            if (mask >> i) & 1:
                sign[i] = -1.0
        x = sign * delta
        val = float(x @ gram @ x)
        if val > best_val:
            best_val = val
            best_sign = sign
    return float(best_val), best_sign


def _rdp_curve_from_sensitivity_sq(
    *,
    orders: Sequence[int | float],
    sensitivity_sq: float,
    source: str,
    radius: Optional[float],
) -> RdpCurve:
    sens_sq = float(sensitivity_sq)
    if sens_sq < 0.0 or not np.isfinite(sens_sq):
        raise ValueError("sensitivity_sq must be finite and >= 0.")
    eps = [0.5 * float(alpha) * sens_sq for alpha in orders]
    return RdpCurve(
        orders=tuple(float(a) for a in orders),
        epsilons=tuple(float(v) for v in eps),
        source=str(source),
        radius=None if radius is None else float(radius),
    )


def account_linear_gaussian_observer(
    *,
    transfer_matrix: np.ndarray,
    covariance: np.ndarray,
    block_sensitivities: Sequence[float],
    parameter_dim: int,
    orders: Sequence[int | float] = _DEFAULT_ORDERS,
    radius: Optional[float] = None,
    dp_delta: Optional[float] = None,
    attacked_node: Optional[int] = None,
    observer: Any = None,
    covariance_mode: Literal["auto", "kron_eye", "full"] = "auto",
    method: Literal[
        "auto",
        "operator_upper",
        "nonnegative_exact",
        "scalar_vertex_exact",
    ] = "auto",
    max_exact_q: int = 20,
    atol: float = 1e-10,
    source: str = "ball_pn_rdp_linear_gaussian",
) -> ObserverSpecificAccountantResult:
    r"""Observer-specific Ball-PN-RDP for a linear Gaussian decentralized view.

    This implements the theorem-side accountant for
        Y_A = c_A(D_{-j}) + (H_{A<-j} \otimes I_p) s_j(D) + zeta_A.

    Important correctness note
    --------------------------
    For generic blockwise Euclidean constraints, computing
        sup_{||delta_l|| <= Delta_l} ||Sigma^{-1/2} (H \otimes I_p) delta||_2^2
    exactly is nontrivial. The code therefore reports:
      - an exact value only in theorem-safe special cases,
      - otherwise the sound operator-norm upper bound from the theorem.

    Exact cases implemented here
    ----------------------------
    1. covariance_mode='kron_eye' and the reduced Gram matrix G = H^T Sigma_t^{-1} H
       has entrywise nonnegative coefficients:
           Delta^2 = d^T G d,  d = (Delta_1, ..., Delta_q).
       This follows from |<u,v>| <= ||u|| ||v||.
    2. parameter_dim == 1 and q <= max_exact_q:
       the problem reduces to a scalar box-constrained quadratic maximization, whose
       maximum is attained at a vertex and can be enumerated exactly.

    In all other cases, the function returns the theorem-backed upper bound
        lambda_max((H \otimes I_p)^T Sigma^{-1} (H \otimes I_p)) * sum_l Delta_l^2.
    """
    delta = _validate_block_sensitivities(block_sensitivities)
    q = int(delta.size)
    H = _validate_transfer_matrix(transfer_matrix, q_expected=q)
    p = int(parameter_dim)
    if p <= 0:
        raise ValueError("parameter_dim must be positive.")
    if dp_delta is not None and not (0.0 < float(dp_delta) < 1.0):
        raise ValueError("dp_delta must lie in (0,1) when provided.")
    if max_exact_q < 0:
        raise ValueError("max_exact_q must be >= 0.")

    cov = np.asarray(covariance, dtype=np.float64)
    if covariance_mode == "auto":
        if cov.shape == (H.shape[0], H.shape[0]):
            mode = "kron_eye"
        elif cov.shape == (H.shape[0] * p, H.shape[0] * p):
            mode = "full"
        else:
            raise ValueError(
                "Could not infer covariance_mode. Provide either a time-domain covariance "
                "of shape (d_A, d_A) for covariance_mode='kron_eye', or a full covariance "
                "of shape (d_A * p, d_A * p)."
            )
    else:
        mode = str(covariance_mode)
        if mode not in {"kron_eye", "full"}:
            raise ValueError(
                "covariance_mode must be one of {'auto', 'kron_eye', 'full'}."
            )

    exact = False
    resolved_method = None
    metadata: dict[str, Any] = {
        "parameter_dim": int(p),
        "num_sensitive_blocks": int(q),
        "covariance_mode": str(mode),
    }

    if mode == "kron_eye":
        Sigma_t = _validate_spd(cov, name="covariance_time")
        G = _observer_gram_kron_eye(H, Sigma_t)
        metadata["gram_matrix"] = G.astype(float).tolist()
        metadata["gram_eigenvalues"] = np.linalg.eigvalsh(G).astype(float).tolist()

        if method == "auto":
            if np.all(G >= -float(atol)):
                resolved_method = "nonnegative_exact"
            elif p == 1 and q <= int(max_exact_q):
                resolved_method = "scalar_vertex_exact"
            else:
                resolved_method = "operator_upper"
        else:
            resolved_method = str(method)

        if resolved_method == "nonnegative_exact":
            if not np.all(G >= -float(atol)):
                raise ValueError(
                    "nonnegative_exact is only valid when the reduced Gram matrix is entrywise nonnegative."
                )
            sensitivity_sq = float(delta @ G @ delta)
            exact = True
        elif resolved_method == "scalar_vertex_exact":
            if p != 1:
                raise ValueError("scalar_vertex_exact requires parameter_dim == 1.")
            if q > int(max_exact_q):
                raise ValueError(
                    f"scalar_vertex_exact requires q <= max_exact_q={int(max_exact_q)}."
                )
            sensitivity_sq, best_sign = _scalar_vertex_exact(G, delta)
            metadata["best_sign"] = best_sign.astype(float).tolist()
            exact = True
        elif resolved_method == "operator_upper":
            lam_max = float(np.max(np.linalg.eigvalsh(G)))
            sensitivity_sq = float(max(0.0, lam_max) * float(delta @ delta))
            metadata["lambda_max_reduced_gram"] = float(max(0.0, lam_max))
            exact = False
        else:
            raise ValueError(
                "method must be one of {'auto', 'operator_upper', 'nonnegative_exact', 'scalar_vertex_exact'}."
            )
    else:
        if method not in {"auto", "operator_upper"}:
            raise ValueError(
                "For covariance_mode='full', only the theorem-backed operator_upper method is implemented."
            )
        resolved_method = "operator_upper"
        Sigma = _validate_spd(cov, name="covariance")
        K = np.kron(H, np.eye(p, dtype=np.float64))
        gram_full = K.T @ np.linalg.solve(Sigma, K)
        lam_max = float(np.max(np.linalg.eigvalsh(gram_full)))
        sensitivity_sq = float(max(0.0, lam_max) * float(delta @ delta))
        metadata["lambda_max_full_gram"] = float(max(0.0, lam_max))
        metadata["full_gram_shape"] = tuple(int(v) for v in gram_full.shape)
        exact = False

    curve = _rdp_curve_from_sensitivity_sq(
        orders=orders,
        sensitivity_sq=sensitivity_sq,
        source=f"{source}::{resolved_method}",
        radius=radius,
    )
    cert = (
        None
        if dp_delta is None
        else rdp_to_dp(curve, delta=float(dp_delta), source="rdp_to_dp")
    )
    if cert is not None:
        cert.note = (
            "Observer-specific Ball-PN-RDP converted to DP through the standard "
            "RDP->DP minimization over orders."
        )

    metadata["exact"] = bool(exact)
    metadata["sensitivity_sq"] = float(sensitivity_sq)
    if not exact:
        metadata["note"] = (
            "This is a theorem-backed upper bound on the observer-specific sensitivity, "
            "not a generic exact value."
        )

    return ObserverSpecificAccountantResult(
        attacked_node=None if attacked_node is None else int(attacked_node),
        observer=observer,
        sensitivity_sq=float(sensitivity_sq),
        exact=bool(exact),
        method=str(resolved_method),
        rdp_curve=curve,
        dp_certificate=cert,
        metadata=metadata,
    )
