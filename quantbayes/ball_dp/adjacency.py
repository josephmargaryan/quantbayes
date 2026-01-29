# quantbayes/ball_dp/adjacency.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class BallAdjacency(Generic[T]):
    """
    Policy/threat-model adjacency:

    D ~_r D' iff they differ in exactly one record and the differing records
    have distance <= r under a chosen record metric.

    This class is mostly "semantic glue" for code clarity; experiments mainly
    use r directly to compute sensitivity/noise.
    """

    dist: Callable[[T, T], float]
    r: float

    def is_neighbor_record(self, x: T, x_prime: T) -> bool:
        return float(self.dist(x, x_prime)) <= float(self.r)
