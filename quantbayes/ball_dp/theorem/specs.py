from __future__ import annotations

import dataclasses as dc
from typing import Any, Literal, Optional


Task = Literal["binary", "multiclass"]
Parameterization = Literal["dense", "svd"]
Constraint = Literal["fro", "op"]


@dc.dataclass(frozen=True)
class TheoremModelSpec:
    """Declarative description of one theorem-backed model family."""

    d_in: int
    hidden_dim: int
    task: Task = "binary"
    num_classes: Optional[int] = None
    parameterization: Parameterization = "dense"
    constraint: Constraint = "fro"
    rank: Optional[int] = None

    def __post_init__(self) -> None:
        if int(self.d_in) <= 0:
            raise ValueError("d_in must be positive.")
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.task not in {"binary", "multiclass"}:
            raise ValueError("task must be 'binary' or 'multiclass'.")
        if self.parameterization not in {"dense", "svd"}:
            raise ValueError("parameterization must be 'dense' or 'svd'.")
        if self.constraint not in {"fro", "op"}:
            raise ValueError("constraint must be 'fro' or 'op'.")
        if self.task == "binary":
            if self.num_classes not in {None, 2}:
                raise ValueError(
                    "Binary models should leave num_classes=None (or set it to 2)."
                )
        else:
            if self.num_classes is None or int(self.num_classes) < 2:
                raise ValueError("Multiclass models require num_classes >= 2.")
        if self.parameterization == "svd" and self.constraint != "op":
            raise ValueError(
                "Fixed-basis SVD models are only implemented for the operator-norm theorem family."
            )
        if self.parameterization != "svd" and self.rank is not None:
            raise ValueError("rank is only meaningful for parameterization='svd'.")
        if self.rank is not None and int(self.rank) <= 0:
            raise ValueError("rank must be positive when provided.")

    @property
    def loss_name(self) -> str:
        return "binary_logistic" if self.task == "binary" else "softmax_cross_entropy"

    @property
    def expects_lambda(self) -> bool:
        return self.constraint == "op" or self.parameterization == "svd"

    @property
    def expects_s(self) -> bool:
        return self.constraint == "fro" and self.parameterization == "dense"

    def to_svd(self, *, rank: Optional[int] = None) -> "TheoremModelSpec":
        return TheoremModelSpec(
            d_in=int(self.d_in),
            hidden_dim=int(self.hidden_dim),
            task=self.task,
            num_classes=self.num_classes,
            parameterization="svd",
            constraint="op",
            rank=self.rank if rank is None else int(rank),
        )

    def as_dict(self) -> dict[str, Any]:
        return dc.asdict(self)


@dc.dataclass(frozen=True)
class TheoremBounds:
    """Theorem-side public bounds used by the certificates.

    B is the public input norm bound from the theorem.
    A bounds the output head norm.
    S is used by the dense Frobenius family.
    Lambda is used by the dense operator-norm and fixed-basis SVD families.
    """

    B: float
    A: float
    S: Optional[float] = None
    Lambda: Optional[float] = None

    def __post_init__(self) -> None:
        if float(self.B) < 0.0:
            raise ValueError("B must be nonnegative.")
        if float(self.A) < 0.0:
            raise ValueError("A must be nonnegative.")
        if self.S is not None and float(self.S) < 0.0:
            raise ValueError("S must be nonnegative when provided.")
        if self.Lambda is not None and float(self.Lambda) < 0.0:
            raise ValueError("Lambda must be nonnegative when provided.")

    def validate_for(self, spec: TheoremModelSpec) -> None:
        if spec.expects_s and self.S is None:
            raise ValueError("This model family requires bounds.S.")
        if spec.expects_lambda and self.Lambda is None:
            raise ValueError("This model family requires bounds.Lambda.")

    def constraint_kwargs(self, spec: TheoremModelSpec) -> dict[str, float]:
        self.validate_for(spec)
        if spec.expects_s:
            return {"S": float(self.S), "A": float(self.A)}
        return {"Lambda": float(self.Lambda), "A": float(self.A)}

    def certificate_kwargs(self, spec: TheoremModelSpec) -> dict[str, float]:
        kwargs = {"A": float(self.A), "B": float(self.B), "H": int(spec.hidden_dim)}
        if spec.expects_s:
            kwargs["S"] = float(self.S)
        else:
            kwargs["Lambda"] = float(self.Lambda)
        return kwargs

    def as_dict(self) -> dict[str, Any]:
        return dc.asdict(self)


@dc.dataclass(frozen=True)
class TrainConfig:
    """Trainer defaults for theorem-backed Ball-DP / standard-DP workflows.

    The defaults choose Poisson sampling and matched accounting because those are the
    cleanest privacy-aligned defaults for the theorem-facing public API.
    """

    radius: float = 1.0
    privacy: str = "ball_dp"
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    num_steps: int = 1000
    batch_size: int = 128
    batch_sampler: str = "poisson"
    accountant_subsampling: str = "match_sampler"
    clip_norm: float = 1.0
    noise_multiplier: float = 1.0
    learning_rate: float = 1e-3
    checkpoint_selection: str = "last"
    eval_every: int = 250
    eval_batch_size: int = 1024
    normalize_noisy_sum_by: str = "batch_size"
    poisson_static_batching: bool = False
    poisson_static_batch_buckets: Optional[tuple[int, ...]] = None
    seed: int = 0

    def as_fit_kwargs(self) -> dict[str, Any]:
        return {
            "radius": float(self.radius),
            "privacy": str(self.privacy),
            "epsilon": None if self.epsilon is None else float(self.epsilon),
            "delta": None if self.delta is None else float(self.delta),
            "num_steps": int(self.num_steps),
            "batch_size": int(self.batch_size),
            "batch_sampler": str(self.batch_sampler),
            "accountant_subsampling": str(self.accountant_subsampling),
            "clip_norm": float(self.clip_norm),
            "noise_multiplier": float(self.noise_multiplier),
            "checkpoint_selection": str(self.checkpoint_selection),
            "eval_every": int(self.eval_every),
            "eval_batch_size": int(self.eval_batch_size),
            "normalize_noisy_sum_by": str(self.normalize_noisy_sum_by),
            "poisson_static_batching": bool(self.poisson_static_batching),
            "poisson_static_batch_buckets": (
                None
                if self.poisson_static_batch_buckets is None
                else tuple(int(v) for v in self.poisson_static_batch_buckets)
            ),
            "seed": int(self.seed),
        }
