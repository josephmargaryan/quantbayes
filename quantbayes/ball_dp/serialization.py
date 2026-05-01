# quantbayes/ball_dp/serialization.py
from __future__ import annotations

import dataclasses as dc
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .config import to_dict
from .types import ReleaseArtifact, ShadowCorpus, ReconstructorArtifact


def _to_jsonable(obj: Any) -> Any:
    if dc.is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in dc.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes)):
        return {str(k): _to_jsonable(v) for k, v in obj.__dict__.items()}
    return obj


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_dataframe(
    df: pd.DataFrame, path: str | Path, *, save_parquet_if_possible: bool = True
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if save_parquet_if_possible:
        try:
            df.to_parquet(path.with_suffix(".parquet"), index=False)
        except Exception:
            pass


def save_release_artifact(
    release: ReleaseArtifact,
    out_dir: str | Path,
    *,
    allow_sensitive_extra: bool = False,
) -> Path:
    if (not allow_sensitive_extra) and ("nonprivate_reference" in release.extra):
        raise ValueError(
            'Refusing to serialize release.extra["nonprivate_reference"] by default. Pass allow_sensitive_extra=True only for controlled local debugging.'
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl = out_dir / "release.pkl"
    meta = out_dir / "release_metadata.json"
    save_pickle(release, pkl)
    meta.write_text(
        json.dumps(_to_jsonable(_release_metadata(release)), indent=2, sort_keys=True)
    )
    return pkl


def save_shadow_corpus(corpus: ShadowCorpus, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl = out_dir / "shadow_corpus.pkl"
    save_pickle(corpus, pkl)
    (out_dir / "shadow_corpus_metadata.json").write_text(
        json.dumps(
            _to_jsonable(
                {
                    "n_train": len(corpus.train_examples),
                    "n_val": len(corpus.val_examples),
                    "n_test": len(corpus.test_examples),
                    "codec_metadata": corpus.codec_metadata,
                    "feature_metadata": corpus.feature_metadata,
                    "config_snapshot": corpus.config_snapshot,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return pkl


def save_reconstructor_artifact(
    artifact: ReconstructorArtifact, out_dir: str | Path
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl = out_dir / "reconstructor.pkl"
    save_pickle(artifact, pkl)
    (out_dir / "reconstructor_metadata.json").write_text(
        json.dumps(
            _to_jsonable(
                {
                    "feature_dim": artifact.feature_dim,
                    "target_feature_dim": artifact.target_feature_dim,
                    "num_classes": artifact.num_classes,
                    "validation_metrics": artifact.validation_metrics,
                    "config_snapshot": artifact.config_snapshot,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return pkl


def _release_metadata(release: ReleaseArtifact) -> Dict[str, Any]:
    return {
        "release_kind": release.release_kind,
        "model_family": release.model_family,
        "architecture": release.architecture,
        "training_config": release.training_config,
        "sensitivity": dc.asdict(release.sensitivity),
        "optimization": (
            None if release.optimization is None else dc.asdict(release.optimization)
        ),
        "attack_metadata": release.attack_metadata,
        "dataset_metadata": release.dataset_metadata,
        "utility_metrics": release.utility_metrics,
        "extra": {
            k: ("<redacted>" if k == "nonprivate_reference" else v)
            for k, v in release.extra.items()
        },
    }
