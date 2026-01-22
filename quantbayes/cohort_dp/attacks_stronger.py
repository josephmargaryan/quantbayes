# cohort_dp/attacks_stronger.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Set
import numpy as np


@dataclass
class CoverageIntersectionAttacker:
    """
    Stronger attacker than pure frequency:

    1) Choose multiple nearby queries around a target embedding x_t (Gaussian jitter).
    2) For each query z_j, repeatedly call the API (same session_id) to enumerate
       as many distinct IDs as possible.
       - This is *especially powerful* if the system uses no_repeat,
         because it returns unseen IDs (accelerates enumeration).
    3) Intersect the enumerated sets across query points; predict the most frequent
       ID in intersections, or the unique intersection if it collapses.

    This directly tests the "no-repeat can backfire" hypothesis.
    """

    query_noise_std: float
    n_query_points: int
    enum_rounds_per_point: int
    k_attack: int
    rng: np.random.Generator

    session_id: str = "attacker"
    new_session_per_point: bool = False  # bypass policies if True

    def attack(self, api, X: np.ndarray, target_idx: int) -> int:
        X = np.asarray(X, dtype=float)
        x_t = X[int(target_idx)]
        sets: List[Set[int]] = []

        for j in range(int(self.n_query_points)):
            z = x_t + self.rng.normal(
                loc=0.0, scale=self.query_noise_std, size=x_t.shape[0]
            )

            sid = self.session_id
            if self.new_session_per_point:
                sid = f"{self.session_id}::pt{j}"

            seen: Set[int] = set()
            for _ in range(int(self.enum_rounds_per_point)):
                try:
                    out = api.query(z=z, k=self.k_attack, session_id=sid)
                except TypeError:
                    out = api.query(z=z, k=self.k_attack)
                for idx in out.tolist():
                    seen.add(int(idx))
            sets.append(seen)

        # Intersect sets to shrink candidate pool
        inter = sets[0].copy()
        for s in sets[1:]:
            inter.intersection_update(s)

        if len(inter) == 0:
            # fallback: vote by occurrence across sets
            counts: Dict[int, int] = {}
            for s in sets:
                for idx in s:
                    counts[idx] = counts.get(idx, 0) + 1
            return max(counts.items(), key=lambda kv: kv[1])[0] if counts else 0

        if len(inter) == 1:
            return next(iter(inter))

        # If intersection still large, pick most common across sets restricted to inter
        counts = {i: 0 for i in inter}
        for s in sets:
            for i in inter:
                if i in s:
                    counts[i] += 1
        return max(counts.items(), key=lambda kv: kv[1])[0]
