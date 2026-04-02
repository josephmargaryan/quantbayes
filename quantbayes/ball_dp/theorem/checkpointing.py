from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Optional

import equinox as eqx
import jax.random as jr

from .registry import make_model, replace_dense_with_svd
from .specs import TheoremBounds, TheoremModelSpec


def save_model_checkpoint(
    model: Any,
    spec: TheoremModelSpec,
    path: str | Path,
    *,
    state: Any = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "spec.json").write_text(
        json.dumps(spec.as_dict(), indent=2, sort_keys=True)
    )
    (path / "metadata.json").write_text(
        json.dumps(dict(metadata or {}), indent=2, sort_keys=True)
    )
    eqx.tree_serialise_leaves(path / "model.eqx", model)
    if state is not None:
        with (path / "state.pkl").open("wb") as f:
            pickle.dump(state, f)
    return path


def load_model_checkpoint(
    path: str | Path,
    *,
    key: Any = None,
    dtype: Any = None,
) -> tuple[Any, TheoremModelSpec, Any, dict[str, Any]]:
    path = Path(path)
    spec = TheoremModelSpec(**json.loads((path / "spec.json").read_text()))
    if key is None:
        key = jr.PRNGKey(0)
    template = make_model(spec, key=key, dtype=dtype, init_project=False)
    model = eqx.tree_deserialise_leaves(path / "model.eqx", template)

    state = None
    if (path / "state.pkl").exists():
        with (path / "state.pkl").open("rb") as f:
            state = pickle.load(f)

    metadata = {}
    if (path / "metadata.json").exists():
        metadata = json.loads((path / "metadata.json").read_text())
    return model, spec, state, metadata


def load_dense_checkpoint_as_svd(
    path: str | Path,
    *,
    rank: Optional[int] = None,
    bounds: Optional[TheoremBounds] = None,
    init_project: bool = False,
    key: Any = None,
    dtype: Any = None,
) -> tuple[Any, TheoremModelSpec, Any, dict[str, Any]]:
    dense_model, dense_spec, state, metadata = load_model_checkpoint(
        path, key=key, dtype=dtype
    )
    if dense_spec.parameterization != "dense":
        raise ValueError(
            "load_dense_checkpoint_as_svd expects a checkpoint created from a dense theorem model."
        )
    svd_spec = dense_spec.to_svd(rank=rank)
    svd_model = replace_dense_with_svd(
        dense_model,
        svd_spec,
        init_project=bool(init_project),
        bounds=bounds,
    )
    return svd_model, svd_spec, state, metadata
