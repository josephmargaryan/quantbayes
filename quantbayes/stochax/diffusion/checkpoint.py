# quantbayes/stochax/diffusion/checkpoint.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import equinox as eqx


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    ckpt_dir: str | Path,
    *,
    model,
    ema_model,
    opt_state,
    step: int,
    extras: dict[str, Any] | None = None,
    keep_last: int = 3,
) -> Path:
    """Save eqx model, ema_model, opt_state and metadata under ckpt_dir/step_{step}."""
    ckpt_dir = Path(ckpt_dir)
    step_dir = ckpt_dir / f"step_{step:08d}"
    _ensure_dir(step_dir)

    with open(step_dir / "model.eqx", "wb") as f:
        eqx.tree_serialise_leaves(f, model)  # <-- file first
    with open(step_dir / "ema_model.eqx", "wb") as f:
        eqx.tree_serialise_leaves(f, ema_model)  # <-- file first
    with open(step_dir / "opt_state.eqx", "wb") as f:
        eqx.tree_serialise_leaves(f, opt_state)  # <-- file first

    meta = {"step": step}
    if extras:
        meta.update(extras)
    with open(step_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    _ensure_dir(ckpt_dir)
    with open(ckpt_dir / "latest.txt", "w") as f:
        f.write(str(step))

    # GC old checkpoints
    all_steps = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
    )
    if len(all_steps) > keep_last:
        for d in all_steps[: len(all_steps) - keep_last]:
            for fp in d.glob("*"):
                try:
                    fp.unlink()
                except Exception:
                    pass
            try:
                d.rmdir()
            except Exception:
                pass

    return step_dir


def load_checkpoint(
    ckpt_dir: str | Path,
    model_template,
    ema_template,
    opt_state_template,
    *,
    step: int | None = None,
) -> Tuple[Any, Any, Any, int]:
    """Load the specified step (or latest) into the provided templates."""
    ckpt_dir = Path(ckpt_dir)
    if step is None:
        latest = ckpt_dir / "latest.txt"
        if not latest.exists():
            raise FileNotFoundError(f"No latest checkpoint in {ckpt_dir}")
        step = int(latest.read_text().strip())

    step_dir = ckpt_dir / f"step_{step:08d}"
    if not step_dir.exists():
        raise FileNotFoundError(f"Checkpoint {step_dir} not found")

    with open(step_dir / "model.eqx", "rb") as f:
        model = eqx.tree_deserialise_leaves(f, model_template)  # file first
    with open(step_dir / "ema_model.eqx", "rb") as f:
        ema_model = eqx.tree_deserialise_leaves(f, ema_template)  # file first
    with open(step_dir / "opt_state.eqx", "rb") as f:
        opt_state = eqx.tree_deserialise_leaves(f, opt_state_template)

    return model, ema_model, opt_state, step
