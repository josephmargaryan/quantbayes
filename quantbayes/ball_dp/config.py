# quantbayes/ball_dp/config.py
from __future__ import annotations

import dataclasses as dc
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

T = TypeVar("T")


def _convert_value(tp: Any, value: Any) -> Any:
    origin = get_origin(tp)
    args = get_args(tp)
    if value is None:
        return None
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        for cand in non_none:
            try:
                return _convert_value(cand, value)
            except Exception:
                continue
        return value
    if dc.is_dataclass(tp):
        return from_dict(tp, value)
    if origin in {list, List, Sequence, tuple, Tuple}:
        inner = args[0] if args else Any
        if origin in {tuple, Tuple} and len(args) > 1 and args[1] is not Ellipsis:
            return tuple(_convert_value(t, v) for t, v in zip(args, value))
        seq = [_convert_value(inner, v) for v in value]
        return tuple(seq) if origin in {tuple, Tuple} else seq
    if origin is Literal:
        return value
    if tp in {int, float, str, bool, dict, Dict, list, List}:
        return tp(value)
    return value


def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    kwargs = {}
    for field in dc.fields(cls):
        if field.name not in data:
            continue
        kwargs[field.name] = _convert_value(field.type, data[field.name])
    return cls(**kwargs)  # type: ignore[arg-type]


def to_dict(obj: Any) -> Any:
    if dc.is_dataclass(obj):
        return {k: to_dict(v) for k, v in dc.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj


@dc.dataclass
class GaussianCalibrationConfig:
    method: Literal["classic", "analytic"] = "analytic"
    tol: float = 1e-12


@dc.dataclass
class ConvexOptimizationConfig:
    solver: Literal[
        "lbfgs_fullbatch",
        "gd_fullbatch",
        "stochax_lbfgs",
        "stochax_zoom",
        "stochax_backtrack",
    ] = "lbfgs_fullbatch"
    max_iter: int = 500
    learning_rate: float = 1e-1

    # Stopping criteria. Set any of these to None to disable that criterion.
    grad_tol: Optional[float] = 1e-8
    param_tol: Optional[float] = 1e-10
    objective_tol: Optional[float] = 1e-12

    # Early-stopping control.
    early_stop: bool = True
    stop_rule: Literal["any", "all", "grad_only"] = "any"
    min_iter: int = 0

    certify_approximation: bool = True
    approximation_mode: Literal["optimality_residual", "none"] = "optimality_residual"
    theorem_backed_only: bool = True
    line_search_steps: int = 15


@dc.dataclass
class ConvexReleaseConfig:
    model_family: Literal[
        "softmax_logistic", "binary_logistic", "squared_hinge", "ridge_prototype"
    ]
    radius: float
    lam: float
    gaussian: GaussianCalibrationConfig = dc.field(
        default_factory=GaussianCalibrationConfig
    )
    optimization: ConvexOptimizationConfig = dc.field(
        default_factory=ConvexOptimizationConfig
    )
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    sigma: Optional[float] = None
    orders: Tuple[float, ...] = (2, 3, 4, 5, 8, 16, 32, 64, 128)
    dp_deltas_for_rdp: Tuple[float, ...] = (1e-6,)
    embedding_bound: Optional[float] = None
    standard_radius: Optional[float] = None
    num_classes: Optional[int] = None
    lz_mode: Literal[
        "paper_default",
        "provided",
        "glm_bound",
    ] = "paper_default"
    provided_lz: Optional[float] = None
    use_exact_sensitivity_if_available: bool = True
    seed: int = 0
    store_nonprivate_reference: bool = False


@dc.dataclass
class ArchitectureConstantConfig:
    lz: Optional[float] = None
    B: Optional[float] = None
    G: Optional[float] = None
    H: Optional[float] = None
    s: Tuple[float, ...] = ()
    c: Tuple[float, ...] = ()
    chi: Tuple[float, ...] = ()
    m: Tuple[float, ...] = ()
    beta: Tuple[float, ...] = ()


@dc.dataclass
class BallSGDConfig:
    """Minimal configuration for Ball-SGD / standard DP-SGD training.

    This configuration intentionally does *not* contain any model-building fields.
    The caller must always supply the Equinox model, state, optimizer, and loss.

    Notes
    -----
    - `noise_multipliers` follow the standard DP-SGD convention used by production
      libraries: the Gaussian noise added to the *summed clipped gradient* at step t
      has standard deviation `noise_multipliers[t] * clip_norms[t]`.
    - `lz` is a user-supplied theorem-backed constant. The library does not attempt
      to derive it automatically.
    - Public checkpoint selection should only use a non-private evaluation set.
    - `fixed_batch_indices_schedule`, when supplied, allows deterministic batch
      selection for specific steps. Each entry is either:
          * None   -> sample that step randomly as usual
          * tuple of indices -> force that exact minibatch at that step
      This is intended for controlled attack experiments, not public release metadata.
    """

    radius: float = 1.0
    lz: Optional[float] = None
    num_steps: int = 1000
    batch_sizes: Union[int, Tuple[int, ...]] = 128
    clip_norms: Union[float, Tuple[float, ...]] = 1.0
    noise_multipliers: Union[float, Tuple[float, ...]] = 1.0
    orders: Tuple[int, ...] = (2, 3, 4, 5, 8, 16, 32, 64, 128)
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    loss_name: Literal["softmax_cross_entropy", "binary_logistic"] = (
        "softmax_cross_entropy"
    )
    normalize_noisy_sum_by: Literal["batch_size", "none"] = "batch_size"

    fixed_batch_indices_schedule: Optional[Tuple[Optional[Tuple[int, ...]], ...]] = None

    # Optional built-in parameter-only penalties added *after* DP sanitization.
    frobenius_reg_strength: float = 0.0
    spectral_reg_strength: float = 0.0
    spectral_reg_kwargs: Dict[str, Any] = dc.field(default_factory=dict)

    # Public evaluation / checkpointing.
    eval_every: int = 250
    eval_batch_size: int = 1024
    checkpoint_selection: Literal[
        "last",
        "best_public_eval_loss",
        "best_public_eval_accuracy",
    ] = "last"

    # Public model-only diagnostics.
    record_operator_norms: bool = False
    operator_norms_every: int = 250
    store_full_operator_norm_history: bool = False
    operator_norm_kwargs: Dict[str, Any] = dc.field(default_factory=dict)

    warn_if_ball_equals_standard: bool = True
    seed: int = 0


# Backward-compatible alias for code that still imports the old name.
NonconvexReleaseConfig = BallSGDConfig


@dc.dataclass
class ShadowCorpusConfig:
    num_trials: int
    train_frac: float = 0.7
    val_frac: float = 0.15
    seed: int = 0
    store_releases: bool = False


@dc.dataclass
class ReconstructorTrainingConfig:
    hidden_dims: Tuple[int, ...] = (1000, 1000)
    batch_size: int = 128
    num_epochs: int = 30
    patience: int = 30
    learning_rate: float = 1e-3
    seed: int = 0
