# quantbayes/pkstruct/core/protocols.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from ..typing import Array, PRNGKey


class Prior(Protocol):
    """Fine-variable prior π(ω)."""

    def sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array: ...

    def log_prob(self, omega: Array) -> Array: ...


class LogDensity(Protocol):
    """Log-density over a variable (typically coarse ξ)."""

    def log_prob(self, x: Array) -> Array: ...


# A coarse map ξ = g(ω)
CoarseMap = Callable[[Array], Array]


@dataclass(frozen=True)
class PKComponents:
    prior: Prior
    coarse_map: CoarseMap
    evidence: LogDensity  # p(ξ)
    reference: LogDensity  # π(ξ)
