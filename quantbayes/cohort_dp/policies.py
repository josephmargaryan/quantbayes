# cohort_dp/policies.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
import hashlib


def _hash_query(z: np.ndarray, decimals: int = 3) -> str:
    z = np.asarray(z, dtype=float).reshape(-1)
    zr = np.round(z, decimals=decimals)
    h = hashlib.sha256(zr.tobytes()).hexdigest()
    return h


@dataclass
class StickyOutputPolicy:
    """
    Cache outputs per (session_id, query_hash) so repeated queries return the same
    cohort. This prevents enumeration via repeated identical queries.

    - decimals controls how aggressively queries are considered "the same".
    - max_cache bounds memory; older entries are evicted FIFO-ish.
    """

    rng: np.random.Generator
    decimals: int = 3
    max_cache: int = 50_000

    _cache: Dict[Tuple[str, str], np.ndarray] = field(
        default_factory=dict, init=False, repr=False
    )
    _order: list[Tuple[str, str]] = field(default_factory=list, init=False, repr=False)

    def get_cached(self, session_id: str, z: np.ndarray) -> Optional[np.ndarray]:
        key = (str(session_id), _hash_query(z, decimals=self.decimals))
        return self._cache.get(key, None)

    def set_cached(self, session_id: str, z: np.ndarray, out: np.ndarray) -> None:
        key = (str(session_id), _hash_query(z, decimals=self.decimals))
        if key in self._cache:
            return
        self._cache[key] = np.asarray(out, dtype=int)
        self._order.append(key)
        if len(self._order) > self.max_cache:
            old = self._order.pop(0)
            self._cache.pop(old, None)
