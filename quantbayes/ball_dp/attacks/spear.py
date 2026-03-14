# quantbayes/ball_dp/attacks/spear.py

from __future__ import annotations

import dataclasses as dc
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ..nonconvex.per_example import combine_model, partition_model, resolve_loss_fn


@dc.dataclass
class SpearAttackConfig:
    """Configuration for SPEAR.

    Exact-theorem mode:
      - noisy_mode=False
      - batch_size=None so b is inferred as rank(dL/dW)
      - sample submatrices of size (b-1) x b
      - early stop only when gamma reaches 1

    Noisy / DP-SGD adaptation mode:
      - noisy_mode=True
      - batch_size should be known
      - sample larger submatrices of size (b+1) x b by default
      - keep only candidates whose sampled L_A has rank exactly b-1
      - early stop when gamma reaches `noisy_gamma_target` < 1
      - use a looser zero_tol
    """

    max_samples: int = 50_000
    batch_size: Optional[int] = None
    tau: Optional[float] = None
    false_rejection_rate: float = 1e-5
    zero_tol: float = 1e-10
    dedup_cosine_tol: float = 1e-6
    rank_tol: Optional[float] = None
    random_seed: int = 0

    stop_when_lambda_one: bool = True
    greedy_swap_rule: str = "best_improvement"  # {'best_improvement', 'appendix_scan'}
    max_greedy_passes: int = 10_000

    # Noisy / DP-SGD adaptation.
    noisy_mode: bool = False
    noisy_gamma_target: Optional[float] = None
    noisy_submatrix_rows: Optional[int] = None


@dc.dataclass
class SpearAttackResult:
    attack_family: str
    x_hat: Optional[np.ndarray]
    status: str
    diagnostics: Dict[str, Any]
    metrics: Dict[str, float] = dc.field(default_factory=dict)
    x_hat_aligned: Optional[np.ndarray] = None
    permutation_to_truth: Optional[np.ndarray] = None


@dc.dataclass(frozen=True)
class _CandidateDirection:
    vector: np.ndarray
    sparsity_count: int
    sampled_rows: Tuple[int, ...]


def _as_float64_array(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return arr


def _relative_zero_mask(x: np.ndarray, zero_tol: float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(arr))))
    thr = float(zero_tol) * scale
    return np.abs(arr) <= thr


def _count_relative_zeros(x: np.ndarray, zero_tol: float) -> int:
    return int(np.sum(_relative_zero_mask(np.asarray(x, dtype=np.float64), zero_tol)))


def _matrix_rank(x: np.ndarray, rank_tol: Optional[float]) -> int:
    x = np.asarray(x, dtype=np.float64)
    return int(np.linalg.matrix_rank(x, tol=rank_tol))


def _normalize_direction(q: np.ndarray, zero_tol: float) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("Encountered a zero direction.")
    q = q / norm
    nz = np.where(np.abs(q) > float(zero_tol))[0]
    if nz.size > 0 and q[int(nz[0])] < 0.0:
        q = -q
    return q


def _same_direction(a: np.ndarray, b: np.ndarray, cosine_tol: float) -> bool:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 0.0 or bn <= 0.0:
        return False
    cos = abs(float(np.dot(a, b) / (an * bn)))
    return cos >= 1.0 - float(cosine_tol)


def false_rejection_rate_from_tau(m: int, tau: float) -> float:
    if m <= 0:
        raise ValueError("m must be positive.")
    k = int(math.floor(float(tau) * float(m)))
    if k < 0:
        return 0.0
    denom = 1 << int(m)
    num = 0
    for i in range(k + 1):
        num += math.comb(int(m), int(i))
    return float(num / denom)


def tau_from_false_rejection_rate(m: int, pfr_target: float) -> float:
    """Return the largest tau such that p_fr(tau, m) <= pfr_target."""
    if m <= 0:
        raise ValueError("m must be positive.")
    pfr_target = float(pfr_target)
    if not (0.0 < pfr_target < 1.0):
        raise ValueError("pfr_target must be in (0, 1).")

    denom = 1 << int(m)
    cumulative = 0.0
    best_k = -1
    for k in range(int(m) + 1):
        cumulative += math.comb(int(m), int(k)) / denom
        if cumulative <= pfr_target + 1e-18:
            best_k = k
        else:
            break
    if best_k < 0:
        return -1.0 / float(m)
    return float(best_k / float(m))


def _low_rank_decompose(
    grad_w: np.ndarray,
    *,
    batch_size: Optional[int],
    rank_tol: Optional[float],
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, int]:
    grad_w = np.asarray(grad_w, dtype=np.float64)
    u, s, vh = np.linalg.svd(grad_w, full_matrices=False)
    inferred_rank = _matrix_rank(grad_w, rank_tol)
    b = int(inferred_rank if batch_size is None else batch_size)
    if b <= 0:
        raise ValueError("SPEAR requires a positive batch size / rank.")
    if b > min(grad_w.shape):
        raise ValueError(
            f"Requested/inferred batch size {b} exceeds min(out_dim, in_dim)={min(grad_w.shape)}."
        )
    sqrt_s = np.sqrt(np.clip(s[:b], a_min=0.0, a_max=None))
    L = u[:, :b] * sqrt_s[None, :]
    R = sqrt_s[:, None] * vh[:b, :]
    return L, R, b, s, inferred_rank


def _null_vector_of_rank_b_minus_1_submatrix(
    la: np.ndarray,
    *,
    expected_rank: int,
    rank_tol: Optional[float],
    zero_tol: float,
) -> tuple[Optional[np.ndarray], int]:
    """Return a normalized null vector when rank(la) == expected_rank."""
    la = np.asarray(la, dtype=np.float64)
    rank = _matrix_rank(la, rank_tol)
    if rank != int(expected_rank):
        return None, int(rank)

    _, _, vh = np.linalg.svd(la, full_matrices=True)
    q = vh[-1, :]
    return _normalize_direction(q, zero_tol), int(rank)


def _fix_scale(
    basis: np.ndarray,
    L: np.ndarray,
    grad_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    basis = np.asarray(basis, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    grad_b = np.asarray(grad_b, dtype=np.float64).reshape(-1)
    if basis.ndim != 2 or basis.shape[0] != basis.shape[1]:
        raise ValueError(
            "basis must be square with columns equal to the selected directions."
        )
    L_left = np.linalg.pinv(L)
    s = np.linalg.solve(basis, L_left @ grad_b)
    Q = basis @ np.diag(s)
    return Q, s


def _compute_gamma_and_batch(
    *,
    L: np.ndarray,
    R: np.ndarray,
    W: np.ndarray,
    bias: np.ndarray,
    grad_b: np.ndarray,
    basis: np.ndarray,
    zero_tol: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Q, scales = _fix_scale(basis, L, grad_b)
    dLdZ = L @ Q
    x_hat = np.linalg.solve(Q, R)  # = X^T, shape (batch, input_dim)
    Z = W @ x_hat.T + bias[:, None]

    zero_mask = _relative_zero_mask(dLdZ, zero_tol)
    gamma_minus = int(np.sum((Z <= 0.0) & zero_mask))
    gamma_plus = int(np.sum((Z > 0.0) & (~zero_mask)))
    gamma = float((gamma_minus + gamma_plus) / float(Z.size))
    return gamma, x_hat, dLdZ, Q, scales


def _initial_basis_indices(
    candidates: Sequence[_CandidateDirection],
    *,
    batch_size: int,
    rank_tol: Optional[float],
) -> list[int]:
    order = sorted(
        range(len(candidates)),
        key=lambda i: (-int(candidates[i].sparsity_count), i),
    )
    chosen: list[int] = []
    for idx in order:
        trial = chosen + [idx]
        mat = np.stack([candidates[i].vector for i in trial], axis=1)
        if _matrix_rank(mat, rank_tol) > len(chosen):
            chosen.append(idx)
        if len(chosen) == int(batch_size):
            break
    if len(chosen) != int(batch_size):
        raise RuntimeError(
            "Candidate pool does not contain enough independent directions to initialize GREEDYFILTER."
        )
    return chosen


def _greedy_filter(
    *,
    candidates: Sequence[_CandidateDirection],
    L: np.ndarray,
    R: np.ndarray,
    W: np.ndarray,
    bias: np.ndarray,
    grad_b: np.ndarray,
    batch_size: int,
    zero_tol: float,
    rank_tol: Optional[float],
    swap_rule: str,
    max_greedy_passes: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], int]:
    selected = _initial_basis_indices(
        candidates,
        batch_size=batch_size,
        rank_tol=rank_tol,
    )
    basis = np.stack([candidates[i].vector for i in selected], axis=1)
    gamma, x_hat, dLdZ, Q, scales = _compute_gamma_and_batch(
        L=L,
        R=R,
        W=W,
        bias=bias,
        grad_b=grad_b,
        basis=basis,
        zero_tol=zero_tol,
    )

    swap_rule = str(swap_rule).lower()
    if swap_rule not in {"best_improvement", "appendix_scan"}:
        raise ValueError(
            "swap_rule must be one of {'best_improvement', 'appendix_scan'}."
        )

    passes = 0
    while passes < int(max_greedy_passes):
        passes += 1
        improved = False

        unselected = [i for i in range(len(candidates)) if i not in selected]
        if not unselected:
            break

        if swap_rule == "best_improvement":
            best_payload = None
            best_gamma = gamma
            for pos, _idx_in in enumerate(selected):
                for idx_out in unselected:
                    trial_sel = list(selected)
                    trial_sel[pos] = idx_out
                    trial_basis = np.stack(
                        [candidates[i].vector for i in trial_sel],
                        axis=1,
                    )
                    if _matrix_rank(trial_basis, rank_tol) < int(batch_size):
                        continue
                    trial_gamma, trial_x, trial_dLdZ, trial_Q, trial_scales = (
                        _compute_gamma_and_batch(
                            L=L,
                            R=R,
                            W=W,
                            bias=bias,
                            grad_b=grad_b,
                            basis=trial_basis,
                            zero_tol=zero_tol,
                        )
                    )
                    if trial_gamma > best_gamma + 1e-15:
                        best_gamma = trial_gamma
                        best_payload = (
                            trial_sel,
                            trial_basis,
                            trial_gamma,
                            trial_x,
                            trial_dLdZ,
                            trial_Q,
                            trial_scales,
                        )
            if best_payload is None:
                break
            selected, basis, gamma, x_hat, dLdZ, Q, scales = best_payload
            improved = True
        else:
            for pos, _idx_in in enumerate(list(selected)):
                current_unselected = [
                    i for i in range(len(candidates)) if i not in selected
                ]
                changed_here = False
                for idx_out in current_unselected:
                    trial_sel = list(selected)
                    trial_sel[pos] = idx_out
                    trial_basis = np.stack(
                        [candidates[i].vector for i in trial_sel],
                        axis=1,
                    )
                    if _matrix_rank(trial_basis, rank_tol) < int(batch_size):
                        continue
                    trial_gamma, trial_x, trial_dLdZ, trial_Q, trial_scales = (
                        _compute_gamma_and_batch(
                            L=L,
                            R=R,
                            W=W,
                            bias=bias,
                            grad_b=grad_b,
                            basis=trial_basis,
                            zero_tol=zero_tol,
                        )
                    )
                    if trial_gamma > gamma + 1e-15:
                        selected = trial_sel
                        basis = trial_basis
                        gamma = trial_gamma
                        x_hat = trial_x
                        dLdZ = trial_dLdZ
                        Q = trial_Q
                        scales = trial_scales
                        improved = True
                        changed_here = True
                        break
                if changed_here:
                    break
            if not improved:
                break

    return gamma, x_hat, dLdZ, Q, scales, selected, passes


def _hungarian(cost: np.ndarray) -> np.ndarray:
    """Minimum-cost assignment for a rectangular cost matrix."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    transposed = False
    if n > m:
        cost = cost.T
        n, m = cost.shape
        transposed = True

    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        minv = np.full(m + 1, np.inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assign = np.full(n, -1, dtype=np.int64)
    for j in range(1, m + 1):
        if p[j] != 0:
            assign[p[j] - 1] = j - 1

    if not transposed:
        return assign

    out = np.full(cost.shape[1], -1, dtype=np.int64)
    for row_small, col_small in enumerate(assign.tolist()):
        out[col_small] = row_small
    return out


def align_spear_reconstruction(
    true_batch: np.ndarray,
    pred_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_batch = np.asarray(true_batch, dtype=np.float64)
    pred_batch = np.asarray(pred_batch, dtype=np.float64)
    if true_batch.ndim != 2 or pred_batch.ndim != 2:
        raise ValueError(
            "align_spear_reconstruction expects shape (batch, flat_dim) arrays."
        )
    if true_batch.shape[0] != pred_batch.shape[0]:
        raise ValueError("true_batch and pred_batch must have the same batch size.")
    diff = pred_batch[:, None, :] - true_batch[None, :, :]
    cost = np.linalg.norm(diff, axis=-1)
    pred_to_true = _hungarian(cost)
    aligned = np.zeros_like(pred_batch)
    l2 = np.zeros((true_batch.shape[0],), dtype=np.float64)
    for pred_idx, true_idx in enumerate(pred_to_true.tolist()):
        aligned[true_idx] = pred_batch[pred_idx]
        l2[true_idx] = cost[pred_idx, true_idx]
    return aligned, pred_to_true, l2


def batch_reconstruction_metrics(
    true_batch: np.ndarray,
    pred_batch: np.ndarray,
    *,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
) -> tuple[Dict[str, float], np.ndarray, np.ndarray]:
    aligned, pred_to_true, l2 = align_spear_reconstruction(true_batch, pred_batch)
    out: Dict[str, float] = {
        "mean_assigned_l2": float(np.mean(l2)),
        "median_assigned_l2": float(np.median(l2)),
        "max_assigned_l2": float(np.max(l2)),
    }
    for eta in eta_grid:
        eta = float(eta)
        hits = l2 <= eta
        out[f"item_success_fraction@{eta:g}"] = float(np.mean(hits))
        out[f"batch_success@{eta:g}"] = float(np.all(hits))
    return out, aligned, pred_to_true


def _resolve_tau(m: int, cfg: SpearAttackConfig) -> tuple[float, float, int]:
    tau = cfg.tau
    if tau is None:
        tau = tau_from_false_rejection_rate(
            m=int(m),
            pfr_target=float(cfg.false_rejection_rate),
        )
    tau = float(tau)
    threshold_count = int(math.ceil(tau * float(m)))
    achieved_pfr = false_rejection_rate_from_tau(int(m), tau)
    return tau, achieved_pfr, threshold_count


def _resolve_sampling_policy(
    *,
    out_dim: int,
    batch_size: int,
    cfg: SpearAttackConfig,
) -> tuple[int, Optional[float]]:
    """Resolve row count for sampled L_A and the early-stop gamma threshold."""
    if not bool(cfg.noisy_mode):
        row_count = int(batch_size - 1)
        gamma_stop = 1.0 if bool(cfg.stop_when_lambda_one) else None
        return row_count, gamma_stop

    row_count = (
        int(cfg.noisy_submatrix_rows)
        if cfg.noisy_submatrix_rows is not None
        else int(batch_size + 1)
    )
    if row_count > int(out_dim):
        raise ValueError(
            f"Noisy SPEAR mode requests submatrices with {row_count} rows, "
            f"but out_dim={out_dim}. Increase layer width or set "
            f"noisy_submatrix_rows <= {out_dim}."
        )
    if row_count < int(batch_size - 1):
        raise ValueError(
            "noisy_submatrix_rows must be at least batch_size - 1 so that rank b-1 is achievable."
        )

    # The paper says to stop below 1 under noise, but does not prescribe one universal value.
    gamma_stop = (
        float(cfg.noisy_gamma_target) if cfg.noisy_gamma_target is not None else 0.98
    )
    return row_count, gamma_stop


def run_spear_batch_attack(
    W: np.ndarray,
    bias: np.ndarray,
    grad_W: np.ndarray,
    grad_b: np.ndarray,
    *,
    cfg: Optional[SpearAttackConfig] = None,
    true_batch: Optional[np.ndarray] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
) -> SpearAttackResult:
    """Run SPEAR on a single linear+ReLU layer from observed weight/bias gradients.

    Exact mode (paper-faithful):
      - cfg.noisy_mode=False
      - cfg.batch_size=None to infer b as rank(grad_W)

    Noisy / DP-SGD adaptation:
      - cfg.noisy_mode=True
      - cfg.batch_size should be known
      - larger sampled matrices L_A are used by default
      - candidates are kept only when rank(L_A) == b - 1
      - early stopping occurs when gamma reaches a user-chosen value below 1
    """
    cfg = SpearAttackConfig() if cfg is None else cfg

    W = _as_float64_array(W, name="W")
    bias = _as_float64_array(bias, name="bias").reshape(-1)
    grad_W = _as_float64_array(grad_W, name="grad_W")
    grad_b = _as_float64_array(grad_b, name="grad_b").reshape(-1)

    if W.ndim != 2:
        raise ValueError("W must be a 2D array.")
    if grad_W.shape != W.shape:
        raise ValueError(f"grad_W.shape={grad_W.shape} must equal W.shape={W.shape}.")
    if bias.ndim != 1 or bias.shape[0] != W.shape[0]:
        raise ValueError("bias must have shape (out_dim,).")
    if grad_b.shape != bias.shape:
        raise ValueError(
            f"grad_b.shape={grad_b.shape} must equal bias.shape={bias.shape}."
        )

    L, R, batch_size, singular_values, inferred_rank = _low_rank_decompose(
        grad_W,
        batch_size=cfg.batch_size,
        rank_tol=cfg.rank_tol,
    )
    if batch_size < 2:
        raise ValueError("SPEAR is intended for batch size >= 2.")

    out_dim, input_dim = W.shape
    if batch_size > min(out_dim, input_dim):
        raise ValueError(
            f"SPEAR requires batch_size <= min(out_dim, input_dim), got batch_size={batch_size}, "
            f"out_dim={out_dim}, input_dim={input_dim}."
        )

    tau, achieved_pfr, threshold_count = _resolve_tau(out_dim, cfg)
    row_count, gamma_stop_threshold = _resolve_sampling_policy(
        out_dim=out_dim,
        batch_size=batch_size,
        cfg=cfg,
    )

    rng = np.random.default_rng(int(cfg.random_seed))

    candidates: list[_CandidateDirection] = []
    n_rank_b_minus_1 = 0
    n_sparse_enough = 0
    n_duplicate = 0

    best_gamma = -float("inf")
    best_x_hat = None
    best_dLdZ = None
    best_Q = None
    best_scales = None
    best_selected: Optional[list[int]] = None
    best_passes = 0
    early_stop = False
    sample_idx = 0

    for sample_idx in range(1, int(cfg.max_samples) + 1):
        rows = tuple(
            sorted(rng.choice(out_dim, size=int(row_count), replace=False).tolist())
        )
        LA = L[np.asarray(rows, dtype=np.int64), :]

        q, _sampled_rank = _null_vector_of_rank_b_minus_1_submatrix(
            LA,
            expected_rank=int(batch_size - 1),
            rank_tol=cfg.rank_tol,
            zero_tol=cfg.zero_tol,
        )
        if q is None:
            continue
        n_rank_b_minus_1 += 1

        col = L @ q
        sparsity_count = _count_relative_zeros(col, cfg.zero_tol)
        if sparsity_count < threshold_count:
            continue
        n_sparse_enough += 1

        duplicate = any(
            _same_direction(q, cand.vector, cfg.dedup_cosine_tol) for cand in candidates
        )
        if duplicate:
            n_duplicate += 1
            continue

        candidates.append(
            _CandidateDirection(
                vector=q,
                sparsity_count=int(sparsity_count),
                sampled_rows=rows,
            )
        )

        if len(candidates) < batch_size:
            continue

        gamma, x_hat, dLdZ, Q, scales, selected, passes = _greedy_filter(
            candidates=candidates,
            L=L,
            R=R,
            W=W,
            bias=bias,
            grad_b=grad_b,
            batch_size=batch_size,
            zero_tol=cfg.zero_tol,
            rank_tol=cfg.rank_tol,
            swap_rule=cfg.greedy_swap_rule,
            max_greedy_passes=cfg.max_greedy_passes,
        )

        if gamma > best_gamma:
            best_gamma = gamma
            best_x_hat = np.asarray(x_hat, dtype=np.float64)
            best_dLdZ = np.asarray(dLdZ, dtype=np.float64)
            best_Q = np.asarray(Q, dtype=np.float64)
            best_scales = np.asarray(scales, dtype=np.float64)
            best_selected = list(selected)
            best_passes = int(passes)

        if (
            gamma_stop_threshold is not None
            and gamma >= float(gamma_stop_threshold) - 1e-12
        ):
            early_stop = True
            break

    if best_x_hat is None:
        diagnostics = {
            "batch_size": int(batch_size),
            "inferred_rank": int(inferred_rank),
            "tau": float(tau),
            "tau_zero_count_threshold": int(threshold_count),
            "false_rejection_rate_target": float(cfg.false_rejection_rate),
            "achieved_false_rejection_rate": float(achieved_pfr),
            "candidate_pool_size": int(len(candidates)),
            "n_rank_b_minus_1_submatrices": int(n_rank_b_minus_1),
            "n_sparse_enough": int(n_sparse_enough),
            "n_duplicates": int(n_duplicate),
            "max_samples": int(cfg.max_samples),
            "used_truncated_svd": bool(cfg.batch_size is not None),
            "singular_values": np.asarray(singular_values, dtype=np.float64),
            "noisy_mode": bool(cfg.noisy_mode),
            "sampled_row_count": int(row_count),
            "gamma_stop_threshold": gamma_stop_threshold,
        }
        return SpearAttackResult(
            attack_family="spear_batch_gradient",
            x_hat=None,
            status="failed_no_valid_candidate_basis",
            diagnostics=diagnostics,
        )

    metrics: Dict[str, float] = {}
    x_hat_aligned = None
    permutation_to_truth = None
    if true_batch is not None:
        true_batch = _as_float64_array(true_batch, name="true_batch")
        if true_batch.ndim != 2:
            raise ValueError("true_batch must have shape (batch, input_dim).")
        if true_batch.shape != best_x_hat.shape:
            raise ValueError(
                f"true_batch.shape={true_batch.shape} must equal recovered shape {best_x_hat.shape}."
            )
        metrics, x_hat_aligned, permutation_to_truth = batch_reconstruction_metrics(
            true_batch,
            best_x_hat,
            eta_grid=eta_grid,
        )

    diagnostics = {
        "batch_size": int(batch_size),
        "inferred_rank": int(inferred_rank),
        "tau": float(tau),
        "tau_zero_count_threshold": int(threshold_count),
        "false_rejection_rate_target": float(cfg.false_rejection_rate),
        "achieved_false_rejection_rate": float(achieved_pfr),
        "candidate_pool_size": int(len(candidates)),
        "n_rank_b_minus_1_submatrices": int(n_rank_b_minus_1),
        "n_sparse_enough": int(n_sparse_enough),
        "n_duplicates": int(n_duplicate),
        "n_samples_run": int(sample_idx),
        "best_gamma": float(best_gamma),
        "best_lambda": float(best_gamma),  # backward-compatible name
        "selected_candidate_indices": (
            None if best_selected is None else list(best_selected)
        ),
        "selected_candidate_rows": (
            None
            if best_selected is None
            else [
                tuple(int(v) for v in candidates[i].sampled_rows) for i in best_selected
            ]
        ),
        "best_dLdZ": best_dLdZ,
        "best_Q": best_Q,
        "best_scales": best_scales,
        "used_truncated_svd": bool(cfg.batch_size is not None),
        "swap_rule": str(cfg.greedy_swap_rule),
        "greedy_passes": int(best_passes),
        "early_stop": bool(early_stop),
        "gamma_stop_threshold": gamma_stop_threshold,
        "singular_values": np.asarray(singular_values, dtype=np.float64),
        "noisy_mode": bool(cfg.noisy_mode),
        "sampled_row_count": int(row_count),
    }

    if bool(cfg.noisy_mode):
        status = (
            "ok_noisy_gamma_target_reached" if early_stop else "ok_noisy_best_gamma"
        )
    else:
        status = (
            "ok_exact"
            if abs(float(best_gamma) - 1.0) <= 1e-12
            else "ok_best_gamma_below_one"
        )

    return SpearAttackResult(
        attack_family="spear_batch_gradient",
        x_hat=np.asarray(best_x_hat, dtype=np.float64),
        status=status,
        diagnostics=diagnostics,
        metrics=metrics,
        x_hat_aligned=(
            None
            if x_hat_aligned is None
            else np.asarray(x_hat_aligned, dtype=np.float64)
        ),
        permutation_to_truth=(
            None
            if permutation_to_truth is None
            else np.asarray(permutation_to_truth, dtype=np.int64)
        ),
    )


def _get_by_path(obj: Any, path: Sequence[Any]) -> Any:
    cur = obj
    for key in path:
        if isinstance(key, str):
            if hasattr(cur, key):
                cur = getattr(cur, key)
            elif isinstance(cur, dict):
                cur = cur[key]
            else:
                raise KeyError(
                    f"Could not resolve attribute/key {key!r} on object of type {type(cur).__name__}."
                )
        else:
            cur = cur[key]
    return cur


def extract_linear_layer_arrays(
    obj: Any,
    *,
    layer_path: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    layer = _get_by_path(obj, layer_path)
    if not hasattr(layer, "weight") or not hasattr(layer, "bias"):
        raise TypeError(
            "Resolved object does not expose .weight and .bias. "
            f"Resolved type: {type(layer).__name__}."
        )
    W = _as_float64_array(getattr(layer, "weight"), name="layer.weight")
    b = _as_float64_array(getattr(layer, "bias"), name="layer.bias").reshape(-1)
    if W.ndim != 2:
        raise ValueError("Extracted weight must be a matrix.")
    if b.shape != (W.shape[0],):
        raise ValueError("Extracted bias shape must equal (out_dim,).")
    return W, b


def run_spear_trace_step_attack(
    step: Any,
    *,
    layer_path: Sequence[Any] = ("layers", 0),
    cfg: Optional[SpearAttackConfig] = None,
    true_batch: Optional[np.ndarray] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
) -> SpearAttackResult:
    """Run SPEAR on the gradients stored in a DPSGDTraceStep.

    Exact only when the observed gradient is noiseless and satisfies SPEAR's assumptions.
    When additive Gaussian noise is present, this becomes the paper-inspired noisy adaptation:
      - looser sparsity threshold via zero_tol,
      - early stop below gamma=1,
      - larger sampled L_A by default,
      - keep only candidates from submatrices with rank exactly b-1.
    """
    if getattr(step, "model_before", None) is None:
        raise ValueError("Trace step is missing model_before.")
    if getattr(step, "observed_private_gradient", None) is None:
        raise ValueError("Trace step is missing observed_private_gradient.")

    W, bias = extract_linear_layer_arrays(step.model_before, layer_path=layer_path)
    grad_W, grad_b = extract_linear_layer_arrays(
        step.observed_private_gradient,
        layer_path=layer_path,
    )

    cfg = SpearAttackConfig() if cfg is None else dc.replace(cfg)

    batch_idx = np.asarray(
        getattr(step, "batch_indices", np.zeros((0,), dtype=np.int64))
    )
    if cfg.batch_size is None and batch_idx.size > 0:
        cfg.batch_size = int(batch_idx.size)

    trace_sigma = float(getattr(step, "effective_noise_std", 0.0))
    if trace_sigma > 0.0:
        cfg.noisy_mode = True

    result = run_spear_batch_attack(
        W,
        bias,
        grad_W,
        grad_b,
        cfg=cfg,
        true_batch=true_batch,
        eta_grid=eta_grid,
    )
    result.attack_family = "spear_trace_step"
    result.diagnostics["layer_path"] = tuple(layer_path)
    result.diagnostics["trace_step"] = int(getattr(step, "step", -1))
    result.diagnostics["trace_clip_norm"] = float(getattr(step, "clip_norm", np.nan))
    result.diagnostics["trace_noise_multiplier"] = float(
        getattr(step, "noise_multiplier", np.nan)
    )
    result.diagnostics["trace_effective_noise_std"] = trace_sigma
    result.diagnostics["batch_size_source"] = (
        "trace_batch_indices" if batch_idx.size > 0 else "user_or_rank"
    )
    result.diagnostics["paper_noisy_adaptation_structural"] = bool(cfg.noisy_mode)
    return result


def run_spear_model_batch_attack(
    model: Any,
    xb: np.ndarray,
    yb: np.ndarray,
    *,
    layer_path: Sequence[Any] = ("layers", 0),
    loss_name: str = "softmax_cross_entropy",
    state: Any = None,
    reduction: str = "mean",
    cfg: Optional[SpearAttackConfig] = None,
    true_batch: Optional[np.ndarray] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
    seed: int = 0,
) -> SpearAttackResult:
    """Run SPEAR on an exact batch gradient computed from a model and batch.

    This is the paper-faithful wrapper: it computes the exact gradient of the attacked
    linear layer on the supplied batch, then runs SPEAR on that layer.
    """
    xb = np.asarray(xb)
    yb = np.asarray(yb)
    if xb.ndim < 2:
        raise ValueError("xb must have shape (batch, ...).")
    if len(xb) != len(yb):
        raise ValueError("xb and yb must have the same batch size.")

    resolved_loss = resolve_loss_fn(loss_name)
    params, static = partition_model(model)
    key = jr.PRNGKey(int(seed))

    def _batch_loss_of_params(p: Any) -> jnp.ndarray:
        mdl = combine_model(p, static)
        x_j = jnp.asarray(xb, dtype=jnp.float32)
        y_j = jnp.asarray(yb)
        keys = jr.split(key, x_j.shape[0])
        losses = jax.vmap(
            lambda x, y, k: resolved_loss(mdl, state, x, y, k),
            in_axes=(0, 0, 0),
        )(x_j, y_j, keys)
        if str(reduction).lower() == "sum":
            return jnp.sum(losses)
        if str(reduction).lower() == "mean":
            return jnp.mean(losses)
        raise ValueError("reduction must be one of {'mean', 'sum'}.")

    grads = jax.grad(_batch_loss_of_params)(params)
    W, bias = extract_linear_layer_arrays(model, layer_path=layer_path)
    grad_W, grad_b = extract_linear_layer_arrays(grads, layer_path=layer_path)

    result = run_spear_batch_attack(
        W,
        bias,
        grad_W,
        grad_b,
        cfg=cfg,
        true_batch=true_batch,
        eta_grid=eta_grid,
    )
    result.attack_family = "spear_exact_model_batch"
    result.diagnostics["layer_path"] = tuple(layer_path)
    result.diagnostics["loss_name"] = str(loss_name)
    result.diagnostics["reduction"] = str(reduction)
    result.diagnostics["batch_size_from_data"] = int(len(xb))
    return result
