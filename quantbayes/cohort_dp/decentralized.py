# cohort_dp/decentralized.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from .metrics import Metric


def majority_vote(labels: np.ndarray) -> int:
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return 0
    return int(np.argmax(np.bincount(labels)))


@dataclass
class HospitalNode:
    """
    Holds a local cohort service and the mapping to global indices.
    """

    hid: int
    global_indices: np.ndarray  # indices into global train arrays
    X_local: np.ndarray  # local embeddings
    y_local: np.ndarray  # local labels
    api: object  # CohortDiscoveryAPI
    prototypes: np.ndarray  # (p, d) prototype centers (shared to router)

    def query(
        self, z: np.ndarray, k: int, session_id: Optional[str] = None
    ) -> np.ndarray:
        local_idx = self.api.query(z=z, k=k, session_id=session_id)
        return self.global_indices[np.asarray(local_idx, dtype=int)].astype(int)


@dataclass
class PrototypeFirstRouter:
    """
    Coordinator/router holds only prototypes; routes query to closest hospitals,
    merges local results, reranks globally.

    Policies:
      - output_no_repeat: prevent returning same global IDs within a router session
      - overfetch_factor: overfetch locally to preserve utility after filtering
    """

    hospitals: List[HospitalNode]
    X_train_global: np.ndarray
    y_train_global: np.ndarray
    metric: Metric
    n_probe_hospitals: int

    output_no_repeat: bool = False
    overfetch_factor: int = 3

    _seen_by_session: Dict[str, set] = field(
        default_factory=dict, init=False, repr=False
    )

    def _select_hospitals(self, z: np.ndarray) -> List[int]:
        z = np.asarray(z, dtype=float).reshape(1, -1)
        best = []
        for node in self.hospitals:
            d = self.metric.pairwise(z, node.prototypes).reshape(-1)
            best.append((float(np.min(d)), node.hid))
        best.sort(key=lambda t: t[0])
        return [hid for _, hid in best[: int(self.n_probe_hospitals)]]

    def query(
        self, z: np.ndarray, k_total: int, session_id: Optional[str] = None
    ) -> np.ndarray:
        if k_total <= 0:
            raise ValueError("k_total must be >= 1.")

        chosen_h = self._select_hospitals(z)
        if len(chosen_h) == 0:
            chosen_h = [0]

        # overfetch to preserve utility after no-repeat filtering
        k_over = int(max(1, self.overfetch_factor) * k_total)
        k_per = int(np.ceil(k_over / float(len(chosen_h))))

        gathered: List[int] = []
        for hid in chosen_h:
            node = self.hospitals[hid]
            # namespace the session to avoid collisions in local per-session state
            sid = None if session_id is None else f"{session_id}::H{hid}"
            gidx = node.query(z=z, k=k_per, session_id=sid)
            gathered.extend(np.asarray(gidx, dtype=int).tolist())

        gathered = np.unique(np.array(gathered, dtype=int))
        if gathered.size == 0:
            return gathered

        # global re-rank by true distance to query
        z1 = np.asarray(z, dtype=float).reshape(1, -1)
        d = self.metric.pairwise(z1, self.X_train_global[gathered]).reshape(-1)
        ranked = gathered[np.argsort(d)]

        # router-level output no-repeat
        if self.output_no_repeat and session_id is not None:
            sid = str(session_id)
            seen = self._seen_by_session.setdefault(sid, set())
            ranked = np.array(
                [int(i) for i in ranked.tolist() if int(i) not in seen], dtype=int
            )

        kk = min(int(k_total), int(ranked.size))
        out = ranked[:kk].astype(int)

        if self.output_no_repeat and session_id is not None:
            sid = str(session_id)
            seen = self._seen_by_session.setdefault(sid, set())
            for i in out.tolist():
                seen.add(int(i))

        return out

    def predict_label(
        self, z: np.ndarray, k_total: int, session_id: Optional[str] = None
    ) -> int:
        idx = self.query(z=z, k_total=k_total, session_id=session_id)
        return majority_vote(self.y_train_global[idx])
