# quantbayes/ball_dp/types.py
from __future__ import annotations

import dataclasses as dc
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np


Array = np.ndarray


@dc.dataclass
class Record:
    features: Array
    label: int


@dc.dataclass
class ArrayDataset:
    X: Array
    y: Array
    name: str = "dataset"

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same length.")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def feature_shape(self) -> Tuple[int, ...]:
        return tuple(self.X.shape[1:])

    @property
    def flat_dim(self) -> int:
        return int(np.prod(self.feature_shape))

    @property
    def num_classes(self) -> int:
        if self.y.size == 0:
            return 0
        return int(len(np.unique(self.y)))

    def record(self, idx: int) -> Record:
        return Record(features=np.asarray(self.X[idx]), label=int(self.y[idx]))

    def remove_index(self, idx: int) -> Tuple["ArrayDataset", Record]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        mask = np.ones(len(self), dtype=bool)
        mask[idx] = False
        reduced = ArrayDataset(
            self.X[mask], self.y[mask], name=f"{self.name}_minus_{idx}"
        )
        return reduced, self.record(idx)

    def subset(
        self, indices: Sequence[int], *, name: Optional[str] = None
    ) -> "ArrayDataset":
        idx = np.asarray(indices, dtype=np.int64)
        nm = self.name if name is None else name
        return ArrayDataset(self.X[idx], self.y[idx], name=nm)

    def class_counts(self, num_classes: Optional[int] = None) -> Array:
        k = self.num_classes if num_classes is None else int(num_classes)
        return np.bincount(self.y.astype(np.int64), minlength=k)


@dc.dataclass
class DpCertificate:
    epsilon: float
    delta: float
    source: str
    radius: Optional[float] = None
    order_opt: Optional[float] = None
    note: str = ""


@dc.dataclass
class RdpCurve:
    orders: Tuple[float, ...]
    epsilons: Tuple[float, ...]
    source: str
    radius: Optional[float] = None

    def as_dict(self) -> Dict[float, float]:
        return {float(a): float(e) for a, e in zip(self.orders, self.epsilons)}

    def epsilon_at(self, order: float) -> float:
        table = self.as_dict()
        if float(order) not in table:
            raise KeyError(order)
        return float(table[float(order)])


@dc.dataclass
class StepPrivacyRecord:
    step: int
    batch_size: int
    sample_rate: float
    clip_norm: float
    noise_std: float
    delta_ball: float
    delta_std: float
    ball_rdp: Dict[float, float]
    std_rdp: Dict[float, float]


@dc.dataclass
class PrivacyLedger:
    mechanism: str
    sigma: Optional[float] = None
    radius: Optional[float] = None
    rdp_curve: Optional[RdpCurve] = None
    dp_certificates: List[DpCertificate] = dc.field(default_factory=list)
    step_records: List[StepPrivacyRecord] = dc.field(default_factory=list)
    notes: List[str] = dc.field(default_factory=list)


@dc.dataclass
class DualPrivacyLedger:
    ball: PrivacyLedger
    standard: PrivacyLedger


@dc.dataclass
class OptimizationCertificate:
    solver: str
    exact_solution: bool
    converged: bool
    n_iter: int
    objective_value: float
    grad_norm: float
    parameter_error_bound: float
    sensitivity_addon: float
    notes: List[str] = dc.field(default_factory=list)


@dc.dataclass
class SensitivityMetadata:
    lz: Optional[float]
    lz_source: str
    radius: Optional[float]
    delta_ball: Optional[float]
    delta_std: Optional[float]
    exact_vs_upper: str
    step_delta_ball: Optional[List[float]] = None
    step_delta_std: Optional[List[float]] = None


@dc.dataclass
class ReleaseArtifact:
    release_kind: str
    payload: Any
    model_family: str
    architecture: str
    training_config: Dict[str, Any]
    privacy: DualPrivacyLedger
    sensitivity: SensitivityMetadata
    optimization: Optional[OptimizationCertificate]
    attack_metadata: Dict[str, Any]
    dataset_metadata: Dict[str, Any]
    utility_metrics: Dict[str, float] = dc.field(default_factory=dict)
    extra: Dict[str, Any] = dc.field(default_factory=dict)


@dc.dataclass
class AttackResult:
    attack_family: str
    z_hat: Optional[Array]
    y_hat: Optional[int]
    status: str
    diagnostics: Dict[str, Any]
    metrics: Dict[str, float] = dc.field(default_factory=dict)
    candidates: Optional[List[Tuple[Optional[int], Optional[Array]]]] = None


@dc.dataclass
class ShadowExample:
    attack_features: Array
    target_features: Array
    target_label: int
    metadata: Dict[str, Any]


@dc.dataclass
class ShadowCorpus:
    train_examples: List[ShadowExample]
    val_examples: List[ShadowExample]
    test_examples: List[ShadowExample]
    config_snapshot: Dict[str, Any]
    codec_metadata: Dict[str, Any]
    feature_metadata: Dict[str, Any]


@dc.dataclass
class ReconstructorArtifact:
    model: Any
    state: Any
    config_snapshot: Dict[str, Any]
    validation_metrics: Dict[str, float]
    feature_dim: int
    target_feature_dim: int
    num_classes: int
    extra: Dict[str, Any] = dc.field(default_factory=dict)


@dc.dataclass
class ReRoPoint:
    eta: float
    kappa: float
    gamma_ball: float
    gamma_standard: Optional[float] = None
    alpha_opt_ball: Optional[float] = None
    alpha_opt_standard: Optional[float] = None


@dc.dataclass
class ReRoReport:
    mode: str
    points: List[ReRoPoint]
    metadata: Dict[str, Any]


class RecordMetric(Protocol):
    def distance(self, a: Record, b: Record) -> float: ...


class PriorFamily(Protocol):
    def kappa(self, eta: float) -> float: ...
    def sample(self, n: int, rng: np.random.Generator) -> Array: ...


class AttackFeatureMap(Protocol):
    def __call__(
        self, release: ReleaseArtifact, d_minus: ArrayDataset, aux: Optional[Any]
    ) -> Array: ...
    def metadata(self) -> Mapping[str, Any]: ...


class RecordCodec(Protocol):
    def encode_record(self, record: Record) -> Tuple[Array, int]: ...
    def decode_record(self, features: Array, label: int) -> Record: ...
    def metadata(self) -> Mapping[str, Any]: ...
