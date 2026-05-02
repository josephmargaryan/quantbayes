# quantbayes/ball_dp/decentralized/attacks.py

from __future__ import annotations

import dataclasses as dc
from typing import Any, Callable, Dict, Optional, Sequence, Literal

import jax
import jax.numpy as jnp
import numpy as np

import optax

from ..attacks.ball_priors import BallAttackPrior
from ..metrics import reconstruction_metrics
from ..types import AttackResult, Record


ArrayLike = Any
MeanFn = Callable[[ArrayLike, int], ArrayLike]
SensitiveBlocksFn = Callable[[ArrayLike, int], ArrayLike]


@dc.dataclass
class LinearGaussianMapAttackConfig:
    """Projected MAP attack for a theorem-aligned decentralized Gaussian view."""

    optimizer: Literal["adam", "sgd"] = "adam"
    num_steps: int = 500
    learning_rate: float = 1e-2
    num_restarts: int = 5
    seed: int = 0


@dc.dataclass(frozen=True)
class GaussianQuadraticForm:
    mode: Literal["kron_eye", "full"]
    quadratic_np: Callable[[np.ndarray], float]
    quadratic_jax: Callable[[jnp.ndarray], jnp.ndarray]
    metadata: dict[str, Any]
    batch_quadratic_np: Optional[Callable[[np.ndarray], np.ndarray]] = None
    batch_quadratic_jax: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None


def _make_optimizer(name: str, learning_rate: float):
    key = str(name).lower()
    if key == "adam":
        return optax.adam(float(learning_rate))
    if key == "sgd":
        return optax.sgd(float(learning_rate))
    raise ValueError("optimizer must be one of {'adam', 'sgd'}.")


def _resolve_label_space(
    *,
    known_label: Optional[int],
    label_space: Optional[Sequence[int]],
) -> list[int]:
    if known_label is not None:
        return [int(known_label)]
    if label_space is None:
        raise ValueError("Provide known_label or label_space explicitly.")
    labels = [int(v) for v in label_space]
    if not labels:
        raise ValueError("label_space must be non-empty.")
    return labels


def _restart_points(
    prior: BallAttackPrior,
    *,
    num_restarts: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    wanted = max(1, int(num_restarts))
    out: list[np.ndarray] = [
        prior.project_np(np.asarray(prior.center, dtype=np.float32))
    ]
    if wanted > 1:
        samples = prior.sample(wanted - 1, rng)
        out.extend(
            np.asarray(samples[i], dtype=np.float32) for i in range(len(samples))
        )
    return out[:wanted]


def _records_close_for_finite_prior(
    a: Record,
    b: Record,
    *,
    atol: float = 1e-7,
) -> bool:
    xa = np.asarray(a.features, dtype=np.float32).reshape(-1)
    xb = np.asarray(b.features, dtype=np.float32).reshape(-1)
    return (
        int(a.label) == int(b.label)
        and xa.shape == xb.shape
        and np.allclose(xa, xb, atol=atol, rtol=0.0)
    )


def apply_linear_gaussian_view(
    transfer_matrix: ArrayLike,
    sensitive_blocks: ArrayLike,
    *,
    base_offset: Optional[ArrayLike] = None,
) -> jnp.ndarray:
    """Apply the theorem-side linear Gaussian view map.

    Parameters
    ----------
    transfer_matrix:
        H_{A<-j} with shape (d_A, q).
    sensitive_blocks:
        Candidate sensitive contribution s_j(z), either with shape (q, p) or as a
        flat vector of length q * p.
    base_offset:
        Optional c_A(D_{-j}) offset, either with shape (d_A, p) or flat shape d_A * p.

    Returns
    -------
    Flattened mean view with shape (d_A * p,).
    """
    H = jnp.asarray(transfer_matrix, dtype=jnp.float32)
    if H.ndim != 2:
        raise ValueError("transfer_matrix must have shape (d_A, q).")

    s = jnp.asarray(sensitive_blocks, dtype=jnp.float32)
    if s.ndim == 1:
        q = int(H.shape[1])
        if int(s.size) % q != 0:
            raise ValueError(
                "Flat sensitive_blocks length must be divisible by transfer_matrix.shape[1]."
            )
        p = int(s.size // q)
        s = s.reshape((q, p))
    elif s.ndim == 2:
        if int(s.shape[0]) != int(H.shape[1]):
            raise ValueError(
                "sensitive_blocks.shape[0] must match transfer_matrix.shape[1]."
            )
    else:
        raise ValueError("sensitive_blocks must be 1D or 2D.")

    mean = H @ s

    if base_offset is not None:
        c = jnp.asarray(base_offset, dtype=mean.dtype)
        if c.ndim == 1:
            if int(c.size) != int(mean.size):
                raise ValueError(
                    "Flat base_offset length must equal d_A * p for the inferred view shape."
                )
            c = c.reshape(mean.shape)
        elif c.shape != mean.shape:
            raise ValueError(
                "base_offset must be flat with length d_A * p or have shape (d_A, p)."
            )
        mean = mean + c

    return mean.reshape(-1)


def make_linear_gaussian_mean_fn(
    *,
    transfer_matrix: ArrayLike,
    sensitive_blocks_fn: SensitiveBlocksFn,
    base_offset: Optional[ArrayLike] = None,
) -> MeanFn:
    """Build a candidate mean-view function from the theorem decomposition.

    The returned callable is suitable for both the continuous MAP optimizer and
    finite-prior scorer. Finite-prior scoring will use grouped vmap over x for
    each fixed label when the supplied sensitive_blocks_fn is JAX-compatible.
    """

    def mean_fn(x: ArrayLike, y: int) -> jnp.ndarray:
        s = sensitive_blocks_fn(x, int(y))
        return apply_linear_gaussian_view(
            transfer_matrix,
            s,
            base_offset=base_offset,
        )

    return mean_fn


def make_gaussian_quadratic_form(
    observed_view: np.ndarray,
    covariance: np.ndarray,
    *,
    covariance_mode: Literal["auto", "kron_eye", "full"] = "auto",
) -> GaussianQuadraticForm:
    """Prepare the exact Gaussian negative log-likelihood quadratic form.

    Uses Cholesky solves rather than explicitly forming Sigma^{-1}.

    If covariance has shape (d_A, d_A), it is interpreted as the time/network
    covariance in the Kronecker form Sigma_A = covariance ⊗ I_p, where p is
    inferred from observed_view.size / d_A.

    If covariance has shape (D, D) with D = observed_view.size, it is interpreted
    as the full covariance on the flattened observer view.
    """
    y = np.asarray(observed_view, dtype=np.float64).reshape(-1)
    if y.size == 0:
        raise ValueError("observed_view must be non-empty.")
    if np.any(~np.isfinite(y)):
        raise ValueError("observed_view must contain only finite values.")

    Sigma = np.asarray(covariance, dtype=np.float64)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("covariance must be square.")
    if np.any(~np.isfinite(Sigma)):
        raise ValueError("covariance must contain only finite values.")
    if not np.allclose(Sigma, Sigma.T, atol=1e-10, rtol=0.0):
        raise ValueError("covariance must be symmetric.")

    try:
        chol = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite.") from exc

    if covariance_mode == "auto":
        if Sigma.shape[0] == y.size:
            mode = "full"
        elif y.size % Sigma.shape[0] == 0:
            mode = "kron_eye"
        else:
            raise ValueError(
                "Could not infer covariance_mode from shapes. Provide either a full covariance "
                "of shape (D, D) with D=len(observed_view), or a Kronecker time covariance "
                "of shape (d_A, d_A) with len(observed_view) divisible by d_A."
            )
    else:
        mode = str(covariance_mode)
        if mode not in {"kron_eye", "full"}:
            raise ValueError(
                "covariance_mode must be one of {'auto', 'kron_eye', 'full'}."
            )

    if mode == "kron_eye":
        d_A = int(Sigma.shape[0])
        if y.size % d_A != 0:
            raise ValueError(
                "For covariance_mode='kron_eye', len(observed_view) must be divisible by covariance.shape[0]."
            )

        p = int(y.size // d_A)

        chol_t_np = chol
        chol_t_jax = jnp.asarray(chol_t_np, dtype=jnp.float32)

        y_mat_np = y.reshape((d_A, p))
        y_mat_jax = jnp.asarray(y_mat_np, dtype=jnp.float32)

        def quadratic_np(mean_vec: np.ndarray) -> float:
            mean_mat = np.asarray(mean_vec, dtype=np.float64).reshape((d_A, p))
            resid = y_mat_np - mean_mat
            whitened = np.linalg.solve(chol_t_np, resid)
            return float(0.5 * np.sum(whitened * whitened))

        def quadratic_jax(mean_vec: jnp.ndarray) -> jnp.ndarray:
            mean_mat = jnp.asarray(mean_vec, dtype=jnp.float32).reshape((d_A, p))
            resid = y_mat_jax - mean_mat
            whitened = jnp.linalg.solve(chol_t_jax, resid)
            return 0.5 * jnp.sum(whitened * whitened)

        def batch_quadratic_np(mean_vecs: np.ndarray) -> np.ndarray:
            means = np.asarray(mean_vecs, dtype=np.float64).reshape((-1, d_A, p))
            resid = y_mat_np[None, :, :] - means
            whitened = np.stack(
                [np.linalg.solve(chol_t_np, resid_i) for resid_i in resid],
                axis=0,
            )
            return 0.5 * np.sum(whitened * whitened, axis=(1, 2))

        def batch_quadratic_jax(mean_vecs: jnp.ndarray) -> jnp.ndarray:
            means = jnp.asarray(mean_vecs, dtype=jnp.float32).reshape((-1, d_A, p))
            resid = y_mat_jax[None, :, :] - means

            def solve_one(r: jnp.ndarray) -> jnp.ndarray:
                return jnp.linalg.solve(chol_t_jax, r)

            whitened = jax.vmap(solve_one)(resid)
            return 0.5 * jnp.sum(whitened * whitened, axis=(1, 2))

        return GaussianQuadraticForm(
            mode="kron_eye",
            quadratic_np=quadratic_np,
            quadratic_jax=quadratic_jax,
            batch_quadratic_np=batch_quadratic_np,
            batch_quadratic_jax=batch_quadratic_jax,
            metadata={
                "view_block_dim": int(d_A),
                "parameter_dim": int(p),
                "covariance_cholesky_used": True,
            },
        )

    if Sigma.shape != (y.size, y.size):
        raise ValueError(
            "For covariance_mode='full', covariance must have shape "
            "(len(observed_view), len(observed_view))."
        )

    chol_np = chol
    chol_jax = jnp.asarray(chol_np, dtype=jnp.float32)
    y_jax = jnp.asarray(y, dtype=jnp.float32)

    def quadratic_np(mean_vec: np.ndarray) -> float:
        resid = y - np.asarray(mean_vec, dtype=np.float64).reshape(-1)
        whitened = np.linalg.solve(chol_np, resid)
        return float(0.5 * np.dot(whitened, whitened))

    def quadratic_jax(mean_vec: jnp.ndarray) -> jnp.ndarray:
        resid = y_jax - jnp.asarray(mean_vec, dtype=jnp.float32).reshape(-1)
        whitened = jnp.linalg.solve(chol_jax, resid)
        return 0.5 * jnp.dot(whitened, whitened)

    def batch_quadratic_np(mean_vecs: np.ndarray) -> np.ndarray:
        means = np.asarray(mean_vecs, dtype=np.float64).reshape((-1, y.size))
        resid = y[None, :] - means
        whitened_t = np.linalg.solve(chol_np, resid.T)
        whitened = whitened_t.T
        return 0.5 * np.sum(whitened * whitened, axis=1)

    def batch_quadratic_jax(mean_vecs: jnp.ndarray) -> jnp.ndarray:
        means = jnp.asarray(mean_vecs, dtype=jnp.float32).reshape((-1, y.size))
        resid = y_jax[None, :] - means

        def solve_one(r: jnp.ndarray) -> jnp.ndarray:
            return jnp.linalg.solve(chol_jax, r)

        whitened = jax.vmap(solve_one)(resid)
        return 0.5 * jnp.sum(whitened * whitened, axis=1)

    return GaussianQuadraticForm(
        mode="full",
        quadratic_np=quadratic_np,
        quadratic_jax=quadratic_jax,
        batch_quadratic_np=batch_quadratic_np,
        batch_quadratic_jax=batch_quadratic_jax,
        metadata={
            "view_dim": int(y.size),
            "covariance_cholesky_used": True,
        },
    )


def _candidate_means_grouped_by_label(
    *,
    X: np.ndarray,
    y: np.ndarray,
    mean_fn: MeanFn,
) -> tuple[np.ndarray, bool, list[int]]:
    """Evaluate candidate means, using grouped vmap over x when possible.

    Grouping by label avoids passing a traced label into mean_fn, which is
    important because make_linear_gaussian_mean_fn intentionally casts labels to
    Python ints.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    m = int(X.shape[0])
    mean_rows: list[Optional[np.ndarray]] = [None for _ in range(m)]

    used_vectorized_for_all_labels = True
    vmap_failed_labels: list[int] = []

    for label in sorted(np.unique(y).astype(np.int64).tolist()):
        idx = np.where(y == int(label))[0]
        X_label = jnp.asarray(X[idx], dtype=jnp.float32)

        def one_mean(x_i: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(mean_fn(x_i, int(label)), dtype=jnp.float32).reshape(-1)

        try:
            batch = jax.vmap(one_mean)(X_label)
            batch_np = np.asarray(batch, dtype=np.float32)
        except Exception:
            used_vectorized_for_all_labels = False
            vmap_failed_labels.append(int(label))
            batch_np = np.stack(
                [
                    np.asarray(
                        mean_fn(np.asarray(X[int(i)], dtype=np.float32), int(label)),
                        dtype=np.float32,
                    ).reshape(-1)
                    for i in idx
                ],
                axis=0,
            )

        if batch_np.ndim != 2 or batch_np.shape[0] != int(idx.size):
            raise ValueError(
                "mean_fn must return a fixed-size vector for every candidate."
            )

        for local_j, global_i in enumerate(idx.tolist()):
            mean_rows[int(global_i)] = np.asarray(batch_np[local_j], dtype=np.float32)

    if any(row is None for row in mean_rows):
        raise RuntimeError("Failed to evaluate every candidate mean.")

    means = np.stack(
        [np.asarray(row, dtype=np.float32) for row in mean_rows],
        axis=0,
    )
    return means, used_vectorized_for_all_labels, vmap_failed_labels


def run_linear_gaussian_ball_map_attack(
    *,
    observed_view: np.ndarray,
    prior: BallAttackPrior,
    mean_fn: MeanFn,
    covariance: np.ndarray,
    cfg: Optional[LinearGaussianMapAttackConfig] = None,
    known_label: Optional[int] = None,
    label_space: Optional[Sequence[int]] = None,
    true_record: Optional[Record] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
    covariance_mode: Literal["auto", "kron_eye", "full"] = "auto",
) -> AttackResult:
    """Projected Ball-constrained MAP attack for an exact Gaussian observer view.

    This optimizes the exact negative log-posterior up to an additive constant:

        0.5 * ||Sigma^{-1/2}(y - mean_fn(z, y_label))||_2^2 - log pi(z).
    """
    cfg = LinearGaussianMapAttackConfig() if cfg is None else cfg

    labels = _resolve_label_space(known_label=known_label, label_space=label_space)
    quadratic = make_gaussian_quadratic_form(
        observed_view,
        covariance,
        covariance_mode=covariance_mode,
    )

    optimizer = _make_optimizer(cfg.optimizer, cfg.learning_rate)
    rng = np.random.default_rng(int(cfg.seed))
    starts = _restart_points(prior, num_restarts=cfg.num_restarts, rng=rng)
    center_shape = np.asarray(prior.center).shape

    best_obj = float("inf")
    best_x = None
    best_label = None
    candidate_rows: list[tuple[float, int, np.ndarray]] = []

    for label in labels:
        label_int = int(label)

        def objective(x_var: jnp.ndarray) -> jnp.ndarray:
            mean_vec = mean_fn(x_var, label_int)
            return quadratic.quadratic_jax(mean_vec) + prior.negative_log_density(x_var)

        loss_and_grad = jax.value_and_grad(objective)

        for x0 in starts:
            x = jnp.asarray(
                np.asarray(x0, dtype=np.float32).reshape(center_shape),
                dtype=jnp.float32,
            )
            x = prior.project(x)
            opt_state = optimizer.init(x)

            best_restart_obj = float(objective(x))
            best_restart_x = x

            for _ in range(max(0, int(cfg.num_steps))):
                _, grad = loss_and_grad(x)
                updates, opt_state = optimizer.update(grad, opt_state, x)
                x = optax.apply_updates(x, updates)
                x = prior.project(x)

                curr = float(objective(x))
                if curr < best_restart_obj:
                    best_restart_obj = curr
                    best_restart_x = x

            x_np = np.asarray(best_restart_x, dtype=np.float32)

            obj_np = float(
                quadratic.quadratic_np(
                    np.asarray(mean_fn(best_restart_x, label_int), dtype=np.float32)
                )
                + prior.negative_log_density_np(x_np)
            )

            candidate_rows.append((obj_np, label_int, x_np.copy()))
            if obj_np < best_obj:
                best_obj = obj_np
                best_x = x_np.copy()
                best_label = label_int

    if best_x is None or best_label is None:
        raise RuntimeError("MAP optimization failed to produce a candidate.")

    metrics = (
        {}
        if true_record is None
        else reconstruction_metrics(
            true_record,
            Record(features=best_x, label=int(best_label)),
            eta_grid=tuple(float(v) for v in eta_grid),
        )
    )

    true_obj = None
    if true_record is not None:
        true_x = np.asarray(true_record.features, dtype=np.float32).reshape(
            center_shape
        )
        true_label = int(true_record.label)
        true_obj = float(
            quadratic.quadratic_np(
                np.asarray(
                    mean_fn(true_x, true_label),
                    dtype=np.float32,
                )
            )
            + prior.negative_log_density_np(true_x)
        )
        metrics["objective_gap_to_truth"] = float(best_obj - true_obj)

        feat_pred = np.asarray(best_x, dtype=np.float32).reshape(-1)
        feat_true = np.asarray(true_x, dtype=np.float32).reshape(-1)
        metrics["mse"] = float(np.mean((feat_pred - feat_true) ** 2))

    top_rows = sorted(candidate_rows, key=lambda row: row[0])[
        : min(10, len(candidate_rows))
    ]
    candidates = [(int(lbl), np.asarray(x, dtype=np.float32)) for _, lbl, x in top_rows]

    diagnostics: Dict[str, Any] = {
        "algorithm": "linear_gaussian_ball_map",
        "objective": "exact_gaussian_negative_log_posterior_up_to_constants",
        "best_objective": float(best_obj),
        "objective_at_truth": true_obj,
        "num_labels_searched": int(len(labels)),
        "num_restarts": int(cfg.num_restarts),
        "num_steps": int(cfg.num_steps),
        "learning_rate": float(cfg.learning_rate),
        "optimizer": str(cfg.optimizer),
        "covariance_mode": str(quadratic.mode),
    }
    diagnostics.update(dict(quadratic.metadata))

    return AttackResult(
        attack_family="decentralized_linear_gaussian_ball_map",
        z_hat=np.asarray(best_x, dtype=np.float32),
        y_hat=int(best_label),
        status="ok_known_label" if known_label is not None else "ok",
        diagnostics=diagnostics,
        metrics=metrics,
        candidates=candidates,
    )


def run_linear_gaussian_finite_prior_attack(
    *,
    observed_view: np.ndarray,
    candidate_features: np.ndarray,
    candidate_labels: Sequence[int],
    mean_fn: MeanFn,
    covariance: np.ndarray,
    prior_weights: Optional[Sequence[float]] = None,
    true_record: Optional[Record] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
    covariance_mode: Literal["auto", "kron_eye", "full"] = "auto",
) -> AttackResult:
    """Exact finite-prior Bayes classifier for a Gaussian observer view.

    Candidate means are evaluated by grouped vmap over features for each fixed
    label when mean_fn is JAX-compatible. If a label group is not vmap-compatible,
    that group falls back to serial candidate evaluation.
    """
    X = np.asarray(candidate_features, dtype=np.float32)
    if X.ndim < 2:
        raise ValueError("candidate_features must have shape (m, ...).")

    y = np.asarray(tuple(int(v) for v in candidate_labels), dtype=np.int64)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "candidate_features and candidate_labels must have the same length."
        )
    if X.shape[0] == 0:
        raise ValueError("candidate_features must be non-empty.")

    if prior_weights is None:
        probs = np.full((X.shape[0],), 1.0 / float(X.shape[0]), dtype=np.float64)
    else:
        probs = np.asarray(tuple(float(v) for v in prior_weights), dtype=np.float64)
        if probs.shape != (X.shape[0],):
            raise ValueError(
                f"prior_weights must have shape ({X.shape[0]},), got {probs.shape}."
            )
        if np.any(~np.isfinite(probs)) or np.any(probs <= 0.0):
            raise ValueError("prior_weights must be finite and strictly positive.")
        probs = probs / float(np.sum(probs))

    quadratic = make_gaussian_quadratic_form(
        observed_view,
        covariance,
        covariance_mode=covariance_mode,
    )

    mean_matrix, means_all_vectorized, vmap_failed_labels = (
        _candidate_means_grouped_by_label(
            X=X,
            y=y,
            mean_fn=mean_fn,
        )
    )

    if quadratic.batch_quadratic_np is None:
        quadratics = np.asarray(
            [quadratic.quadratic_np(mean_matrix[i]) for i in range(X.shape[0])],
            dtype=np.float64,
        )
        quadratic_scoring_mode = "serial_quadratic"
    else:
        quadratics = np.asarray(
            quadratic.batch_quadratic_np(mean_matrix), dtype=np.float64
        )
        quadratic_scoring_mode = "batched_quadratic"

    if quadratics.shape != (X.shape[0],):
        raise ValueError(
            f"Batch quadratic scorer returned shape {quadratics.shape}; expected {(X.shape[0],)}."
        )

    log_scores_arr = np.log(probs) - quadratics
    best_idx = int(np.argmax(log_scores_arr))

    records = [
        Record(features=np.asarray(X[i], dtype=np.float32), label=int(y[i]))
        for i in range(X.shape[0])
    ]
    best_record = records[best_idx]

    shifted = log_scores_arr - float(np.max(log_scores_arr))
    posterior = np.exp(shifted)
    posterior = posterior / float(np.sum(posterior))

    posterior_entropy = float(
        -np.sum(posterior * np.log(np.maximum(posterior, 1e-300)))
    )
    posterior_effective_candidates = float(np.exp(posterior_entropy))

    sorted_scores = np.sort(log_scores_arr)[::-1]
    top2_gap = (
        float(sorted_scores[0] - sorted_scores[1])
        if sorted_scores.size >= 2
        else float("inf")
    )

    metrics = (
        {}
        if true_record is None
        else reconstruction_metrics(
            true_record,
            best_record,
            eta_grid=tuple(float(v) for v in eta_grid),
        )
    )
    metrics["oblivious_kappa"] = float(np.max(probs))
    metrics["posterior_top1_probability"] = float(posterior[best_idx])
    metrics["posterior_entropy"] = posterior_entropy
    metrics["posterior_effective_candidates"] = posterior_effective_candidates
    metrics["log_score_gap_top2"] = top2_gap

    true_prior_index = None
    if true_record is not None:
        true_prior_index = next(
            (
                i
                for i, rec in enumerate(records)
                if _records_close_for_finite_prior(rec, true_record)
            ),
            None,
        )

        feat_pred = np.asarray(best_record.features, dtype=np.float32).reshape(-1)
        feat_true = np.asarray(true_record.features, dtype=np.float32).reshape(-1)
        metrics["mse"] = float(np.mean((feat_pred - feat_true) ** 2))
        metrics["exact_identification_success"] = float(
            _records_close_for_finite_prior(best_record, true_record)
        )

        if true_prior_index is not None:
            order = np.argsort(-log_scores_arr)
            rank = int(np.where(order == true_prior_index)[0][0]) + 1
            metrics["prior_exact_hit"] = float(best_idx == true_prior_index)
            metrics["prior_rank"] = float(rank)
            metrics["posterior_true_probability"] = float(posterior[true_prior_index])
            metrics["log_score_gap_truth_to_top"] = float(
                log_scores_arr[true_prior_index] - log_scores_arr[best_idx]
            )
            for kk in (1, 5, 10):
                metrics[f"prior_hit@{kk}"] = float(rank <= kk)

    top_order = np.argsort(-log_scores_arr)[: min(10, len(records))]
    candidates = [
        (
            int(records[int(i)].label),
            np.asarray(records[int(i)].features, dtype=np.float32),
        )
        for i in top_order.tolist()
    ]

    diagnostics: Dict[str, Any] = {
        "algorithm": "linear_gaussian_finite_prior_exact_bayes",
        "prior_size": int(len(records)),
        "prior_weights": probs.astype(float).tolist(),
        "oblivious_kappa": float(np.max(probs)),
        "candidate_log_scores": log_scores_arr.astype(float).tolist(),
        "candidate_log_posteriors": np.log(np.maximum(posterior, 1e-300))
        .astype(float)
        .tolist(),
        "candidate_posterior_probs": posterior.astype(float).tolist(),
        "candidate_objectives": (-log_scores_arr).astype(float).tolist(),
        "posterior_entropy": posterior_entropy,
        "posterior_effective_candidates": posterior_effective_candidates,
        "log_score_gap_top2": top2_gap,
        "predicted_prior_index": int(best_idx),
        "true_prior_index": true_prior_index,
        "true_record_in_prior": bool(true_prior_index is not None),
        "covariance_mode": str(quadratic.mode),
        "candidate_mean_scoring_mode": (
            "grouped_vmap_all_labels"
            if means_all_vectorized
            else "grouped_vmap_with_serial_fallback"
        ),
        "vmap_failed_labels": [int(v) for v in vmap_failed_labels],
        "quadratic_scoring_mode": quadratic_scoring_mode,
    }
    diagnostics.update(dict(quadratic.metadata))

    return AttackResult(
        attack_family="decentralized_linear_gaussian_finite_prior_exact_bayes",
        z_hat=np.asarray(best_record.features, dtype=np.float32),
        y_hat=int(best_record.label),
        status="ok",
        diagnostics=diagnostics,
        metrics=metrics,
        candidates=candidates,
    )
