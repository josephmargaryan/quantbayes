# quantbayes/ball_dp/reconstruction/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional, Protocol, Sequence, Tuple

import numpy as np

Array = np.ndarray
DatasetMinus = Tuple[Array, Array]  # (X_minus, y_minus)

Status = Literal["ok", "no_support_vector", "invalid_input", "failed"]


@dataclass
class Candidate:
    """
    A single candidate record for reconstruction/identification.

    record: feature vector (pixel-space flattened vector OR embedding).
    label:  optional label.
      - multiclass: int in {0,...,K-1}
      - binary: int in {-1,+1}
    meta: arbitrary extra info (e.g., original dataset index).
    """

    record: Array
    label: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconstructionResult:
    """
    Standardized output across all reconstruction attacks.

    record_hat: reconstructed record in the same representation as the attack (pixels or embeddings).
    label_hat:  reconstructed label (if recoverable / relevant).
    status:     ok | no_support_vector | invalid_input | failed
    details:    optional debug info / intermediate quantities.
    """

    record_hat: Optional[Array]
    label_hat: Optional[int]
    status: Status = "ok"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IdentificationResult:
    """
    Output for multi-hypothesis identification attacks (candidate set selection).

    best_idx: index into the input candidate list.
    scores:   higher is better (e.g., log-likelihood).
    ranked_idx: indices sorted from best->worst.
    """

    best_idx: int
    scores: Array
    ranked_idx: Array
    details: Dict[str, Any] = field(default_factory=dict)


class Vectorizer(Protocol):
    def __call__(self, params: Any, *, state: Any | None = None) -> Array: ...


class DeterministicTrainer(Protocol):
    """
    Deterministic f(D) trainer used for Gaussian-output ML identification.

    Should produce the same output given the same data (fix seeds internally if needed).
    """

    def fit(self, X: Array, y: Array) -> Any: ...


class StochasticTrainer(Protocol):
    """
    Stochastic trainer used for nonconvex shadow-model attacks.

    Different seeds should lead to different trained outputs (or at least different noise).
    """

    def fit(self, X: Array, y: Array, *, seed: int) -> Tuple[Any, Any | None]:
        """
        Returns (params, optional_state).
        """
        ...


MetricFn = Callable[[Array, Array], float]
MetricBatchFn = Callable[[Array, Array], Array]  # metric(center, pool)->(N,)
