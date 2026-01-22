# cohort_dp/api.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Optional, Dict, Set, TYPE_CHECKING
import numpy as np

from .accountant import PrivacyAccountant

if TYPE_CHECKING:
    from .policies import StickyOutputPolicy


class CandidateGenerator(Protocol):
    def candidates(self, z: np.ndarray) -> np.ndarray: ...


class Retriever(Protocol):
    def privacy_cost(self, k: int) -> float: ...
    def query(
        self, z: np.ndarray, k: int, candidates: Optional[np.ndarray] = None
    ) -> np.ndarray: ...


@dataclass
class CohortDiscoveryAPI:
    """
    Service surface:
      - optional candidate generation (scaling)
      - privacy accountant (budgeting)
      - optional per-session no-repeat policy (deployment control against query amplification)
      - optional sticky output policy (prevents enumeration via repeated identical queries)
    """

    retriever: Retriever
    accountant: Optional[PrivacyAccountant] = None
    candidate_generator: Optional[CandidateGenerator] = None

    # Deployment policy knobs
    no_repeat: bool = False
    max_session_history: int = 50_000  # safety to avoid unbounded memory

    # Optional: cache outputs per (session, query-hash)
    sticky_policy: Optional["StickyOutputPolicy"] = None

    _session_seen: Dict[str, Set[int]] = field(
        default_factory=dict, init=False, repr=False
    )

    def query(
        self, z: np.ndarray, k: int, session_id: Optional[str] = None
    ) -> np.ndarray:
        if k <= 0:
            raise ValueError("k must be >= 1.")

        sid = None if session_id is None else str(session_id)

        # 0) Sticky cache check
        if (sid is not None) and (self.sticky_policy is not None):
            cached = self.sticky_policy.get_cached(sid, z)
            if cached is not None and cached.size >= k:
                return cached[:k].astype(int)

        # 1) Candidate generation
        cand = None
        if self.candidate_generator is not None:
            cand = self.candidate_generator.candidates(z)

        # Helper: get full pool indices (when we need to reset/ensure enough candidates)
        def _full_pool_indices() -> np.ndarray:
            X = getattr(self.retriever, "X", None)
            if X is None:
                raise RuntimeError(
                    "no_repeat/reset requires candidate_generator or retriever.X to infer n."
                )
            return np.arange(X.shape[0], dtype=int)

        # 2) no-repeat policy (reset when exhausted)
        if self.no_repeat and sid is not None:
            seen = self._session_seen.setdefault(sid, set())

            # ensure we have a candidate pool
            if cand is None:
                cand = _full_pool_indices()

            # filter out previously seen items
            if len(seen) > 0:
                seen_arr = np.fromiter(seen, dtype=int)
                cand = np.setdiff1d(cand, seen_arr, assume_unique=False)

            # If not enough candidates remain to serve k unique IDs, reset session history.
            # This prevents downstream sampling crashes and matches "no-repeat until exhaustion".
            if cand.size < k:
                seen.clear()
                # rebuild candidates after reset
                if self.candidate_generator is not None:
                    cand = self.candidate_generator.candidates(z)
                else:
                    cand = _full_pool_indices()

                # (paranoia) If STILL too small, we cannot guarantee k unique outputs.
                # In that case, let the retriever handle it (it may sample w/ replacement).
                # But we avoid empty/invalid candidate arrays here.
                if cand.ndim != 1 or cand.size == 0:
                    raise RuntimeError("Candidate pool became invalid after reset.")

        # 3) Spend privacy budget
        if self.accountant is not None:
            self.accountant.spend(self.retriever.privacy_cost(k))

        # 4) Retrieve
        out = self.retriever.query(z=z, k=k, candidates=cand).astype(int)

        # 5) Update session history
        if self.no_repeat and sid is not None:
            seen = self._session_seen.setdefault(sid, set())
            for idx in out.tolist():
                seen.add(int(idx))
            if len(seen) > self.max_session_history:
                seen.clear()

        # 6) Sticky cache write
        if (sid is not None) and (self.sticky_policy is not None):
            self.sticky_policy.set_cached(sid, z, out)

        return out
