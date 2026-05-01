from __future__ import annotations

import dataclasses as dc
import hashlib
from typing import Any, Literal, Optional, Sequence

import numpy as np

from ..types import Record


SupportSelection = Literal["random", "nearest", "farthest"]
AnchorSelection = Literal["random", "rare_class", "large_bank"]
TargetPolicy = Literal["all", "sample"]


def _rng_from_parts(*parts: int) -> np.random.Generator:
    words = [int(p) % (2**32) for p in parts]
    return np.random.default_rng(np.random.SeedSequence(words))


def _feature_distance(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    if aa.shape != bb.shape:
        raise ValueError(f"feature shape mismatch: {aa.shape} vs {bb.shape}")
    return float(np.linalg.norm(aa - bb, ord=2))


def _dedup_key(x: np.ndarray, y: int, *, decimals: int) -> tuple[int, bytes]:
    arr = np.round(np.asarray(x, dtype=np.float32), int(decimals))
    return int(y), arr.tobytes()


def support_source_hash(source_ids: Sequence[str]) -> str:
    h = hashlib.sha1()
    for sid in source_ids:
        h.update(str(sid).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def finite_support_hash(
    X: np.ndarray,
    y: np.ndarray,
    source_ids: Sequence[str],
    weights: Sequence[float],
    *,
    decimals: int = 7,
) -> str:
    h = hashlib.sha1()
    Xr = np.round(np.asarray(X, dtype=np.float32), int(decimals))
    yr = np.asarray(y, dtype=np.int32).reshape(-1)
    wr = np.asarray(weights, dtype=np.float64).reshape(-1)

    h.update(Xr.tobytes())
    h.update(yr.tobytes())
    h.update(np.round(wr, 12).tobytes())
    for sid in source_ids:
        h.update(str(sid).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:20]


def remove_index(
    X: np.ndarray,
    y: np.ndarray,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_arr = np.asarray(X)
    y_arr = np.asarray(y).reshape(-1)
    idx = int(index)

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same first dimension.")
    if idx < 0 or idx >= X_arr.shape[0]:
        raise IndexError(f"index={idx} out of range for n={X_arr.shape[0]}.")

    mask = np.ones((X_arr.shape[0],), dtype=bool)
    mask[idx] = False
    return (
        np.asarray(X_arr[mask], dtype=np.float32),
        np.asarray(y_arr[mask], dtype=np.int32),
    )


def append_candidate_at_end(
    X_minus: np.ndarray,
    y_minus: np.ndarray,
    x_candidate: np.ndarray,
    y_candidate: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    X_minus_arr = np.asarray(X_minus, dtype=np.float32)
    y_minus_arr = np.asarray(y_minus, dtype=np.int32).reshape(-1)
    x_arr = np.asarray(x_candidate, dtype=np.float32)

    if X_minus_arr.shape[0] != y_minus_arr.shape[0]:
        raise ValueError("X_minus and y_minus must have the same first dimension.")
    if x_arr.shape != X_minus_arr.shape[1:]:
        raise ValueError(
            f"candidate shape {x_arr.shape} does not match feature shape "
            f"{X_minus_arr.shape[1:]}."
        )

    X_full = np.concatenate([X_minus_arr, x_arr[None, ...]], axis=0).astype(
        np.float32, copy=False
    )
    y_full = np.concatenate(
        [y_minus_arr, np.asarray([int(y_candidate)], dtype=np.int32)],
        axis=0,
    )
    return X_full, y_full, int(len(y_full) - 1)


@dc.dataclass(frozen=True)
class CandidateSource:
    """A public or optional diagnostic candidate source.

    The canonical paper-aligned setup should use public_only candidates. A train
    source can be supplied for diagnostics, but be careful: train candidates may
    already appear in D^-.
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    indices: Optional[Sequence[int]] = None

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=np.float32)
        y = np.asarray(self.y, dtype=np.int32).reshape(-1)

        if X.ndim < 2:
            raise ValueError("CandidateSource.X must have shape (n, ...).")
        if X.shape[0] != y.shape[0]:
            raise ValueError("CandidateSource.X and .y must have matching length.")

        if self.indices is None:
            indices = np.arange(X.shape[0], dtype=np.int64)
        else:
            indices = np.asarray(self.indices, dtype=np.int64).reshape(-1)
            if indices.shape[0] != X.shape[0]:
                raise ValueError("CandidateSource.indices must have length len(X).")

        name = str(self.name).strip()
        if not name:
            raise ValueError("CandidateSource.name must be non-empty.")
        if ":" in name:
            raise ValueError("CandidateSource.name must not contain ':'.")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "indices", indices)


@dc.dataclass(frozen=True)
class CandidateBank:
    center_source_id: str
    center_index: int
    center_x: np.ndarray
    center_y: int
    X: np.ndarray
    y: np.ndarray
    source_ids: tuple[str, ...]
    distances_to_center: np.ndarray
    radius: Optional[float]
    metadata: dict[str, Any] = dc.field(default_factory=dict)

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=np.float32)
        y = np.asarray(self.y, dtype=np.int32).reshape(-1)
        d = np.asarray(self.distances_to_center, dtype=np.float32).reshape(-1)
        center_x = np.asarray(self.center_x, dtype=np.float32)

        if X.ndim < 2:
            raise ValueError("CandidateBank.X must have shape (n, ...).")
        if X.shape[0] != y.shape[0]:
            raise ValueError("CandidateBank.X and .y must have matching length.")
        if X.shape[0] != len(self.source_ids):
            raise ValueError("source_ids must have length len(X).")
        if X.shape[0] != d.shape[0]:
            raise ValueError("distances_to_center must have length len(X).")
        if X.shape[1:] != center_x.shape:
            raise ValueError("bank feature shape must match center_x shape.")
        if len(set(map(str, self.source_ids))) != len(self.source_ids):
            raise ValueError("CandidateBank.source_ids must be unique.")

        radius = None if self.radius is None else float(self.radius)
        if radius is not None and radius < 0.0:
            raise ValueError("radius must be nonnegative.")

        object.__setattr__(self, "center_source_id", str(self.center_source_id))
        object.__setattr__(self, "center_index", int(self.center_index))
        object.__setattr__(self, "center_x", center_x)
        object.__setattr__(self, "center_y", int(self.center_y))
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "source_ids", tuple(str(s) for s in self.source_ids))
        object.__setattr__(self, "distances_to_center", d)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "metadata", dict(self.metadata))

    # Backward-compatible property names for old convex scripts.
    @property
    def anchor_index(self) -> int:
        return int(self.center_index)

    @property
    def anchor_label(self) -> int:
        return int(self.center_y)

    @property
    def anchor_vector(self) -> np.ndarray:
        return np.asarray(self.center_x, dtype=np.float32)

    @property
    def bank_vectors(self) -> np.ndarray:
        return np.asarray(self.X, dtype=np.float32)

    @property
    def bank_source_ids(self) -> list[str]:
        return list(self.source_ids)

    @property
    def bank_distances(self) -> np.ndarray:
        return np.asarray(self.distances_to_center, dtype=np.float32)


@dc.dataclass(frozen=True)
class FinitePriorSupport:
    center_source_id: str
    center_index: int
    center_x: np.ndarray
    center_y: int
    X: np.ndarray
    y: np.ndarray
    source_ids: tuple[str, ...]
    distances_to_center: np.ndarray
    weights: np.ndarray
    support_hash: str
    selection: str
    radius: Optional[float]
    metadata: dict[str, Any] = dc.field(default_factory=dict)

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=np.float32)
        y = np.asarray(self.y, dtype=np.int32).reshape(-1)
        d = np.asarray(self.distances_to_center, dtype=np.float32).reshape(-1)
        w = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        center_x = np.asarray(self.center_x, dtype=np.float32)

        if X.ndim < 2:
            raise ValueError("FinitePriorSupport.X must have shape (m, ...).")
        if X.shape[0] == 0:
            raise ValueError("FinitePriorSupport must contain at least one candidate.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("FinitePriorSupport.X and .y must have matching length.")
        if X.shape[0] != len(self.source_ids):
            raise ValueError("source_ids must have length len(X).")
        if X.shape[0] != d.shape[0]:
            raise ValueError("distances_to_center must have length len(X).")
        if X.shape[0] != w.shape[0]:
            raise ValueError("weights must have length len(X).")
        if X.shape[1:] != center_x.shape:
            raise ValueError("support feature shape must match center_x shape.")
        if len(set(map(str, self.source_ids))) != len(self.source_ids):
            raise ValueError("FinitePriorSupport.source_ids must be unique.")
        if np.any(y != int(self.center_y)):
            raise ValueError("canonical finite-prior support must be same-label.")
        if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
            raise ValueError("support weights must be finite and strictly positive.")

        w = w / float(np.sum(w))
        radius = None if self.radius is None else float(self.radius)
        if radius is not None:
            if radius < 0.0:
                raise ValueError("radius must be nonnegative.")
            if np.any(d > radius + 1e-5):
                raise ValueError("support contains candidates outside the radius.")

        object.__setattr__(self, "center_source_id", str(self.center_source_id))
        object.__setattr__(self, "center_index", int(self.center_index))
        object.__setattr__(self, "center_x", center_x)
        object.__setattr__(self, "center_y", int(self.center_y))
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "source_ids", tuple(str(s) for s in self.source_ids))
        object.__setattr__(self, "distances_to_center", d)
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "support_hash", str(self.support_hash))
        object.__setattr__(self, "selection", str(self.selection))
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def m(self) -> int:
        return int(self.X.shape[0])

    @property
    def oblivious_kappa(self) -> float:
        return float(np.max(self.weights))


@dc.dataclass(frozen=True)
class FinitePriorTrial:
    """Canonical finite-prior replacement trial.

    D^- is fixed by removing center_index from the private train set.
    The hidden target is support[target_support_position].
    The trained dataset is D^- union {target}, with the target appended at the end.
    """

    D_minus_X: np.ndarray
    D_minus_y: np.ndarray
    X_full: np.ndarray
    y_full: np.ndarray
    target_index: int
    target_support_position: int
    target_source_id: str
    support: FinitePriorSupport

    def __post_init__(self) -> None:
        D_X = np.asarray(self.D_minus_X, dtype=np.float32)
        D_y = np.asarray(self.D_minus_y, dtype=np.int32).reshape(-1)
        X_full = np.asarray(self.X_full, dtype=np.float32)
        y_full = np.asarray(self.y_full, dtype=np.int32).reshape(-1)

        if D_X.shape[0] != D_y.shape[0]:
            raise ValueError("D_minus_X and D_minus_y must have matching length.")
        if X_full.shape[0] != y_full.shape[0]:
            raise ValueError("X_full and y_full must have matching length.")
        if X_full.shape[1:] != D_X.shape[1:]:
            raise ValueError("X_full and D_minus_X feature shapes differ.")
        if X_full.shape[0] != D_X.shape[0] + 1:
            raise ValueError("X_full must be D^- plus exactly one appended target.")

        target_index = int(self.target_index)
        target_pos = int(self.target_support_position)

        if target_index != X_full.shape[0] - 1:
            raise ValueError("canonical trial requires target appended at final index.")
        if target_pos < 0 or target_pos >= self.support.m:
            raise IndexError("target_support_position is out of range.")

        x_target = self.support.X[target_pos]
        y_target = int(self.support.y[target_pos])
        if not np.allclose(X_full[target_index], x_target, atol=1e-7, rtol=0.0):
            raise ValueError("X_full[target_index] does not equal support target.")
        if int(y_full[target_index]) != y_target:
            raise ValueError("y_full[target_index] does not equal support target label.")

        expected_sid = self.support.source_ids[target_pos]
        if str(self.target_source_id) != str(expected_sid):
            raise ValueError("target_source_id does not match support target position.")

        object.__setattr__(self, "D_minus_X", D_X)
        object.__setattr__(self, "D_minus_y", D_y)
        object.__setattr__(self, "X_full", X_full)
        object.__setattr__(self, "y_full", y_full)
        object.__setattr__(self, "target_index", target_index)
        object.__setattr__(self, "target_support_position", target_pos)
        object.__setattr__(self, "target_source_id", str(self.target_source_id))


def build_same_label_ball_bank(
    *,
    center_x: np.ndarray,
    center_y: int,
    center_source_id: str,
    center_index: int,
    sources: Sequence[CandidateSource],
    radius: Optional[float],
    exclude_source_ids: Optional[set[str]] = None,
    tol: float = 1e-7,
    dedup_decimals: int = 7,
    metadata: Optional[dict[str, Any]] = None,
) -> CandidateBank:
    center = np.asarray(center_x, dtype=np.float32)
    label = int(center_y)
    exclude = set() if exclude_source_ids is None else {str(s) for s in exclude_source_ids}

    if radius is not None and float(radius) < 0.0:
        raise ValueError("radius must be nonnegative or None.")

    best: dict[tuple[int, bytes], tuple[float, str, np.ndarray, int]] = {}

    for source in sources:
        src = source if isinstance(source, CandidateSource) else CandidateSource(**source)
        same = np.flatnonzero(np.asarray(src.y, dtype=np.int32) == label)

        for pos in same.tolist():
            src_index = int(np.asarray(src.indices, dtype=np.int64)[pos])
            sid = f"{src.name}:{src_index}"
            if sid in exclude:
                continue

            vec = np.asarray(src.X[pos], dtype=np.float32)
            if vec.shape != center.shape:
                raise ValueError(
                    f"candidate {sid} shape {vec.shape} does not match center shape {center.shape}."
                )

            dist = _feature_distance(vec, center)
            if radius is not None and dist > float(radius) + float(tol):
                continue

            key = _dedup_key(vec, label, decimals=int(dedup_decimals))
            prev = best.get(key)
            item = (float(dist), str(sid), vec, label)
            if prev is None or (item[0], item[1]) < (prev[0], prev[1]):
                best[key] = item

    items = sorted(best.values(), key=lambda item: (item[0], item[1]))

    if items:
        distances = np.asarray([item[0] for item in items], dtype=np.float32)
        source_ids = tuple(str(item[1]) for item in items)
        X = np.stack([item[2] for item in items], axis=0).astype(np.float32, copy=False)
        y = np.asarray([item[3] for item in items], dtype=np.int32)
    else:
        distances = np.zeros((0,), dtype=np.float32)
        source_ids = tuple()
        X = np.zeros((0,) + center.shape, dtype=np.float32)
        y = np.zeros((0,), dtype=np.int32)

    return CandidateBank(
        center_source_id=str(center_source_id),
        center_index=int(center_index),
        center_x=center,
        center_y=label,
        X=X,
        y=y,
        source_ids=source_ids,
        distances_to_center=distances,
        radius=None if radius is None else float(radius),
        metadata={} if metadata is None else dict(metadata),
    )


def _greedy_farthest_positions(
    X: np.ndarray,
    *,
    m: int,
    rng: np.random.Generator,
    anchor_distances: Optional[np.ndarray] = None,
) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float32)
    n = int(X_arr.shape[0])
    if n < int(m):
        raise ValueError("not enough candidates for farthest support selection.")

    X_flat = X_arr.reshape(n, -1)

    if anchor_distances is None:
        first = int(rng.integers(0, n))
    else:
        d_anchor = np.asarray(anchor_distances, dtype=np.float64).reshape(-1)
        top_k = min(n, max(int(m), 4 * int(m)))
        farthest_tail = np.argsort(-d_anchor)[:top_k]
        first = int(rng.choice(farthest_tail))

    selected = [first]
    min_d = np.linalg.norm(X_flat - X_flat[first][None, :], axis=1)
    min_d[first] = -np.inf

    while len(selected) < int(m):
        best_val = float(np.max(min_d))
        candidates = np.flatnonzero(np.isclose(min_d, best_val, rtol=1e-7, atol=1e-7))
        nxt = int(rng.choice(candidates)) if candidates.size else int(np.argmax(min_d))
        selected.append(nxt)
        d_new = np.linalg.norm(X_flat - X_flat[nxt][None, :], axis=1)
        min_d = np.minimum(min_d, d_new)
        min_d[selected] = -np.inf

    return np.asarray(selected, dtype=np.int64)


def select_support_from_bank(
    bank: CandidateBank,
    *,
    m: int,
    selection: SupportSelection = "random",
    seed: int = 0,
    draw_index: int = 0,
    weights: Optional[Sequence[float]] = None,
) -> FinitePriorSupport:
    m = int(m)
    if m < 2:
        raise ValueError("m must be at least 2.")

    available = int(bank.X.shape[0])
    if available < m:
        raise ValueError(
            f"center {bank.center_source_id} has bank size {available}, but m={m}."
        )

    rng = _rng_from_parts(seed, bank.center_index, draw_index, m, 991)
    mode = str(selection).strip().lower()

    if mode == "random":
        positions = rng.choice(available, size=m, replace=False)
    elif mode == "nearest":
        positions = np.argsort(np.asarray(bank.distances_to_center, dtype=np.float64))[:m]
    elif mode == "farthest":
        positions = _greedy_farthest_positions(
            bank.X,
            m=m,
            rng=rng,
            anchor_distances=bank.distances_to_center,
        )
    else:
        raise ValueError("selection must be one of {'random', 'nearest', 'farthest'}.")

    positions = np.asarray(positions, dtype=np.int64)

    Xs = np.asarray(bank.X[positions], dtype=np.float32)
    ys = np.asarray(bank.y[positions], dtype=np.int32)
    sids = tuple(bank.source_ids[int(i)] for i in positions.tolist())
    dists = np.asarray(bank.distances_to_center[positions], dtype=np.float32)

    if weights is None:
        w = np.full((m,), 1.0 / float(m), dtype=np.float64)
    else:
        w = np.asarray(tuple(float(v) for v in weights), dtype=np.float64).reshape(-1)
        if w.shape != (m,):
            raise ValueError(f"weights must have shape ({m},).")
        if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
            raise ValueError("weights must be finite and strictly positive.")
        w = w / float(np.sum(w))

    h = finite_support_hash(Xs, ys, sids, w)

    return FinitePriorSupport(
        center_source_id=bank.center_source_id,
        center_index=int(bank.center_index),
        center_x=np.asarray(bank.center_x, dtype=np.float32),
        center_y=int(bank.center_y),
        X=Xs,
        y=ys,
        source_ids=sids,
        distances_to_center=dists,
        weights=w,
        support_hash=h,
        selection=mode,
        radius=bank.radius,
        metadata={
            **dict(bank.metadata),
            "bank_size": int(available),
            "support_positions_in_bank": positions.astype(int).tolist(),
        },
    )


def find_feasible_replacement_banks(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    candidate_sources: Sequence[CandidateSource],
    radius: float,
    min_support_size: int,
    num_banks: int,
    seed: int = 0,
    max_search: Optional[int] = None,
    explicit_anchor_indices: Optional[Sequence[int]] = None,
    anchor_selection: AnchorSelection = "random",
    strict: bool = False,
    train_source_name: str = "train",
) -> list[CandidateBank]:
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).reshape(-1)

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have matching length.")
    if int(min_support_size) < 2:
        raise ValueError("min_support_size must be at least 2.")
    if int(num_banks) <= 0:
        raise ValueError("num_banks must be positive.")

    strategy = str(anchor_selection).strip().lower()
    if strategy not in {"random", "rare_class", "large_bank"}:
        raise ValueError(
            "anchor_selection must be one of {'random', 'rare_class', 'large_bank'}."
        )

    rng = np.random.default_rng(int(seed))

    if explicit_anchor_indices:
        candidate_indices = [int(i) for i in explicit_anchor_indices]
    else:
        if strategy == "rare_class":
            labels = np.asarray(y_train, dtype=np.int64)
            counts = np.bincount(labels, minlength=int(np.max(labels)) + 1)
            jitter = rng.random(len(X_train))
            candidate_indices = sorted(
                range(len(X_train)),
                key=lambda i: (int(counts[int(y_train[i])]), float(jitter[i])),
            )
        else:
            candidate_indices = rng.permutation(len(X_train)).astype(int).tolist()

        if max_search is not None and strategy != "large_bank":
            candidate_indices = candidate_indices[: int(max_search)]

    banks: list[CandidateBank] = []

    def build(idx: int) -> CandidateBank:
        center_source_id = f"{train_source_name}:{int(idx)}"
        return build_same_label_ball_bank(
            center_x=X_train[int(idx)],
            center_y=int(y_train[int(idx)]),
            center_source_id=center_source_id,
            center_index=int(idx),
            sources=candidate_sources,
            radius=float(radius),
            exclude_source_ids={center_source_id},
            metadata={
                "anchor_selection": strategy,
                "train_source_name": str(train_source_name),
            },
        )

    if strategy == "large_bank" and not explicit_anchor_indices:
        search_indices = candidate_indices
        if max_search is not None:
            search_indices = search_indices[: int(max_search)]

        scored: list[tuple[int, CandidateBank]] = []
        for idx in search_indices:
            bank = build(int(idx))
            if int(bank.X.shape[0]) >= int(min_support_size):
                scored.append((int(bank.X.shape[0]), bank))

        scored.sort(key=lambda item: item[0], reverse=True)
        banks = [bank for _, bank in scored[: int(num_banks)]]
    else:
        for idx in candidate_indices:
            bank = build(int(idx))
            if int(bank.X.shape[0]) >= int(min_support_size):
                banks.append(bank)
            if len(banks) >= int(num_banks):
                break

    if len(banks) < int(num_banks) and strict:
        raise RuntimeError(
            f"Requested {num_banks} feasible banks, but found only {len(banks)} "
            f"for radius={float(radius):.6g}, min_support_size={int(min_support_size)}, "
            f"anchor_selection={strategy!r}."
        )
    if not banks:
        raise RuntimeError(
            f"No feasible finite-prior banks found for radius={float(radius):.6g}, "
            f"min_support_size={int(min_support_size)}, anchor_selection={strategy!r}."
        )

    return banks


def target_positions_for_support(
    support: FinitePriorSupport,
    *,
    policy: TargetPolicy = "all",
    num_targets: Optional[int] = None,
    seed: int = 0,
) -> list[int]:
    m = int(support.m)
    key = str(policy).strip().lower()

    if key == "all":
        return list(range(m))

    if key != "sample":
        raise ValueError("policy must be one of {'all', 'sample'}.")

    k = 1 if num_targets is None else int(num_targets)
    if k <= 0:
        raise ValueError("num_targets must be positive.")
    if k > m:
        raise ValueError("num_targets cannot exceed support size for sample policy.")

    rng = _rng_from_parts(seed, int(support.center_index), m, 4243)
    return [int(i) for i in rng.choice(m, size=k, replace=False).tolist()]


def make_replacement_trial(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    support: FinitePriorSupport,
    target_support_position: int,
    anchor_index: Optional[int] = None,
) -> FinitePriorTrial:
    idx = support.center_index if anchor_index is None else int(anchor_index)
    target_pos = int(target_support_position)

    D_X, D_y = remove_index(X_train, y_train, idx)
    X_full, y_full, target_index = append_candidate_at_end(
        D_X,
        D_y,
        support.X[target_pos],
        int(support.y[target_pos]),
    )

    return FinitePriorTrial(
        D_minus_X=D_X,
        D_minus_y=D_y,
        X_full=X_full,
        y_full=y_full,
        target_index=target_index,
        target_support_position=target_pos,
        target_source_id=support.source_ids[target_pos],
        support=support,
    )


def make_replacement_trials_for_support(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    support: FinitePriorSupport,
    target_policy: TargetPolicy = "all",
    num_targets: Optional[int] = None,
    seed: int = 0,
) -> list[FinitePriorTrial]:
    positions = target_positions_for_support(
        support,
        policy=target_policy,
        num_targets=num_targets,
        seed=seed,
    )
    return [
        make_replacement_trial(
            X_train=X_train,
            y_train=y_train,
            support=support,
            target_support_position=pos,
        )
        for pos in positions
    ]


def support_to_records(support: FinitePriorSupport) -> list[Record]:
    return [
        Record(
            features=np.asarray(support.X[i], dtype=np.float32),
            label=int(support.y[i]),
        )
        for i in range(support.m)
    ]


def trial_true_record(trial: FinitePriorTrial) -> Record:
    return Record(
        features=np.asarray(trial.X_full[trial.target_index], dtype=np.float32),
        label=int(trial.y_full[trial.target_index]),
    )


def enrich_attack_result_with_trial(result: Any, trial: FinitePriorTrial) -> Any:
    """Attach source-ID bookkeeping to an AttackResult-like object.

    This keeps the algorithmic attack code independent of experiment setup while
    making exact-ID metrics robust to duplicate or near-duplicate feature vectors.
    """
    if getattr(result, "diagnostics", None) is None:
        result.diagnostics = {}
    if getattr(result, "metrics", None) is None:
        result.metrics = {}

    diagnostics = result.diagnostics
    metrics = result.metrics

    diagnostics["support_hash"] = str(trial.support.support_hash)
    diagnostics["support_source_ids"] = list(trial.support.source_ids)
    diagnostics["target_source_id"] = str(trial.target_source_id)
    diagnostics["target_support_position"] = int(trial.target_support_position)
    diagnostics["target_index"] = int(trial.target_index)
    diagnostics["center_source_id"] = str(trial.support.center_source_id)
    diagnostics["center_index"] = int(trial.support.center_index)
    diagnostics["center_label"] = int(trial.support.center_y)
    diagnostics["support_radius"] = (
        None if trial.support.radius is None else float(trial.support.radius)
    )
    diagnostics["support_selection"] = str(trial.support.selection)

    pred_idx_raw = diagnostics.get("predicted_prior_index", None)
    pred_sid = None

    try:
        pred_idx = int(pred_idx_raw)
    except Exception:
        pred_idx = None

    if pred_idx is not None and 0 <= pred_idx < trial.support.m:
        pred_sid = str(trial.support.source_ids[pred_idx])
        diagnostics["predicted_source_id"] = pred_sid
    else:
        diagnostics["predicted_source_id"] = None

    metrics["oblivious_kappa"] = float(trial.support.oblivious_kappa)
    metrics["source_exact_identification_success"] = (
        float(pred_sid == str(trial.target_source_id)) if pred_sid is not None else 0.0
    )

    return result
