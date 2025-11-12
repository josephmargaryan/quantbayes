# quantbayes/stochax/utils/optim_util.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Tuple,
    Union,
    Sequence,
    Iterable,
)

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

__all__ = [
    "OptimizerConfig",
    "DecayMaskConfig",
    "LabelConfig",
    "build_optimizer",
]

"""

import optax
from quantbayes.stochax.utils.optim_util import (
    OptimizerConfig, DecayMaskConfig, build_optimizer
)
from quantbayes.stochax.utils.freeze_mask import make_freeze_mask

freeze_mask = make_freeze_mask(model, names=("U", "V"))  # optionally add "alpha_raw"

opt, opt_state, aux = build_optimizer(
    model,
    OptimizerConfig(
        algorithm="adamw",
        lr=3e-4,
        weight_decay=0.05,
        decay_mask=DecayMaskConfig(
            # keep defaults: no decay on spectral tokens; forbid SVD factors
            forbid_svd_factors=True
        ),
        labels=...,  # optional LR groups
    ),
    # zero the update for U,V BEFORE clipping/decay/groups
    prepend=optax.masked(optax.set_to_zero(), freeze_mask),
)

"""
# ---------------------------------------------------------------------------
# Public config objects
# ---------------------------------------------------------------------------


def _path_to_str(path_entries: Tuple[Any, ...]) -> str:
    parts = []
    for pe in path_entries:
        if hasattr(pe, "name"):  # GetAttrKey
            parts.append(str(pe.name))
        elif hasattr(pe, "key"):  # DictKey
            parts.append(str(pe.key))
        elif hasattr(pe, "idx"):  # SequenceKey
            parts.append(f"[{pe.idx}]")
        else:
            parts.append(str(pe))
    s = ".".join(parts)
    return s.replace(".[", "[")


def make_freeze_mask(
    model_or_params: Any,
    names: Iterable[str] = ("U", "V"),
    extra_paths: Iterable[str] = (),
):
    """Return a pytree of booleans: True means 'freeze' (zero out the update)."""
    # arrays-only tree
    if hasattr(model_or_params, "__class__") and not isinstance(
        model_or_params, (dict, list, tuple)
    ):
        params_tree, _ = eqx.partition(model_or_params, eqx.is_inexact_array)
    else:
        params_tree = model_or_params

    leaves_with_path, treedef = jtu.tree_flatten_with_path(params_tree)
    names = set(names)
    extra_paths = set(extra_paths)

    masks = []
    for path_entries, leaf in leaves_with_path:
        if not isinstance(leaf, jnp.ndarray):
            masks.append(False)
            continue
        p = _path_to_str(path_entries)
        toks = set(p.split("."))
        freeze = bool(names & toks) or (p in extra_paths)
        masks.append(freeze)
    return jtu.tree_unflatten(treedef, masks)


@dataclass
class DecayMaskConfig:
    """
    Controls which parameters receive decoupled weight decay.

    Notes
    -----
    - In modern "AdamW"-style setups, **biases and norm parameters do not get decay**.
    - For spectral layers, we default to *no decay* on spectral weights
      (e.g. w_real, w_imag, H_half, K_half, s, W) and on 'alpha'/'delta' scalars.

    Heuristics use leaf path names (see `path_to_str`) and shapes:
      - Bias-like: leaf.ndim <= 1.
      - Norm-like: path contains "norm", "layernorm", "batchnorm", "groupnorm", or
                   leaf.ndim <= 1 (covers scale/offset in most frameworks).
      - Embeddings: path contains "embed", "pos_embed", "token_embedding", etc.
      - Spectral weights: path contains one of {"w_real","w_imag","H_half","K_half","s","W"}.
      - SVD factors: if your model accidentally exposes learnable U/V, we exclude them
        from decay by default (set forbid_svd_factors=False to allow).
    """

    decay_bias: bool = False
    decay_norm: bool = False
    decay_embeddings: bool = False
    decay_spectral_weights: bool = False
    decay_spectral_alpha: bool = False  # alpha_raw, delta_z
    forbid_svd_factors: bool = True
    custom_no_decay_tokens: Tuple[str, ...] = ()


@dataclass
class LabelConfig:
    """
    Per-group learning-rate multipliers via `optax.multi_transform`.

    Typical use:
      - Head/classifier has larger LR: head_lr_mult = 10.0
      - (Optional) Give spectral weights a smaller LR: spectral_lr_mult = 0.5
    """

    head_lr_mult: Optional[float] = None
    spectral_lr_mult: Optional[float] = None

    # Optional custom matchers: Callable[[path_str, leaf], bool] -> True if in group.
    head_matcher: Optional[Callable[[str, jnp.ndarray], bool]] = None
    spectral_matcher: Optional[Callable[[str, jnp.ndarray], bool]] = None


@dataclass
class OptimizerConfig:
    """
    Master config for building an Optax optimizer with best-practice defaults.

    Parameters
    ----------
    algorithm:
        One of {"adamw","adam","sgd","sgdn","lion","adan"}.
        - "sgdn" is SGD with Nesterov momentum.
        - "adan" is optional; only works if your Optax version provides optax.adan.
    lr:
        Base learning-rate or a schedule (callable taking step -> lr).
        For multi-group setups, the base LR is multiplied by each group's multiplier.
    weight_decay:
        Decoupled weight decay factor (AdamW-style). Applied only where the decay mask
        is True (see `DecayMaskConfig`). Leave as 0.0 to disable.
    clip_global_norm:
        If set, applies **global gradient clipping** at this norm (after AGC).
    agc_clip:
        If set, uses **Adaptive Gradient Clipping (AGC)** with this clipping value and
        applies it **before** global clipping.
    """

    algorithm: Literal["adamw", "adam", "sgd", "sgdn", "lion", "adan"] = "adamw"
    lr: Union[float, Callable[[int], float]] = 1e-3
    weight_decay: float = 0.0

    # gradient conditioning
    clip_global_norm: Optional[float] = 1.0
    agc_clip: Optional[float] = None

    # Adam/AdamW
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8

    # SGD
    sgd_momentum: float = 0.9
    sgd_nesterov: bool = False  # ignored unless algorithm == "sgdn"

    # Lion
    lion_b1: float = 0.9
    lion_b2: float = 0.99

    # Adan (if available in your optax)
    adan_b1: float = 0.98
    adan_b2: float = 0.92
    adan_b3: float = 0.99
    adan_eps: float = 1e-8

    # masking/grouping
    decay_mask: DecayMaskConfig = field(default_factory=DecayMaskConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)


# ---------------------------------------------------------------------------
# Helpers: path utils, heuristics, schedules
# ---------------------------------------------------------------------------


def _summarize_labels_and_decay(params_tree, labels_tree, decay_mask_tree):
    leaves = jtu.tree_leaves(params_tree)
    labels = jtu.tree_leaves(labels_tree)
    masks = jtu.tree_leaves(decay_mask_tree)
    n = len(leaves)
    n_decay = sum(bool(m) for m in masks)
    by_group: Dict[str, int] = {}
    for lab in labels:
        by_group[lab] = by_group.get(lab, 0) + 1
    return {"n_leaves": n, "n_decay": n_decay, "groups": by_group}


def _safe_lr_schedule(
    lr: Union[float, Callable[[int], float]],
) -> Callable[[int], float]:
    if callable(lr):
        return lr
    else:
        return lambda step: float(lr)


def _mul_schedule(
    sched: Callable[[int], float], factor: float
) -> Callable[[int], float]:
    if factor == 1.0:
        return sched
    return lambda step: factor * float(sched(step))


def _path_to_str(path_entries: Tuple[Any, ...]) -> str:
    """Turn JAX path entries into a dotted string."""
    parts = []
    for pe in path_entries:
        if hasattr(pe, "name"):  # GetAttrKey
            parts.append(str(pe.name))
        elif hasattr(pe, "key"):  # DictKey
            parts.append(str(pe.key))
        elif hasattr(pe, "idx"):  # SequenceKey
            parts.append(f"[{pe.idx}]")
        else:
            parts.append(str(pe))
    s = ".".join(parts)
    s = s.replace(".[", "[")
    return s


# default matchers/tokens
_SPECTRAL_TOKENS = {"w_real", "w_imag", "H_half", "K_half", "s", "W"}
_ALPHA_TOKENS = {"alpha_raw", "delta_z"}
_EMBED_TOKENS = {
    "embed",
    "embedding",
    "pos_embed",
    "pos_embedding",
    "token_embedding",
    "position",
    "posenc",
}
_NORM_TOKENS = {"norm", "layernorm", "batchnorm", "groupnorm", "ln", "bn", "gn"}


def _default_head_matcher(path: str, leaf: jnp.ndarray) -> bool:
    p = path.lower()
    return any(tok in p for tok in ("head", "classifier", "logits", "out_proj", "fc"))


def _default_spectral_matcher(path: str, leaf: jnp.ndarray) -> bool:
    return any(tok in path.split(".") for tok in _SPECTRAL_TOKENS)


def _should_decay(path: str, leaf: jnp.ndarray, cfg: DecayMaskConfig) -> bool:
    # Explicit custom tokens → never decay
    if cfg.custom_no_decay_tokens and any(
        tok in path for tok in cfg.custom_no_decay_tokens
    ):
        return False

    # Embeddings: default no decay (can enable via cfg)
    if any(tok in path.lower() for tok in _EMBED_TOKENS):
        return cfg.decay_embeddings

    # Spectral weights & alpha/delta
    if any(tok in path.split(".") for tok in _SPECTRAL_TOKENS):
        return cfg.decay_spectral_weights
    if any(tok in path.split(".") for tok in _ALPHA_TOKENS):
        return cfg.decay_spectral_alpha

    # SVD factors (rare safeguard)
    if cfg.forbid_svd_factors and any(tok in path.split(".") for tok in ("U", "V")):
        return False

    # Norm params: often 1D "scale"/"bias" or named with *norm*
    if (any(tok in path.lower() for tok in _NORM_TOKENS)) or (leaf.ndim <= 1):
        # treat as norm/bias bucket → decay only if explicitly enabled
        return cfg.decay_norm if (leaf.ndim > 1) else cfg.decay_bias

    # Otherwise: decay matrices/kernels/etc.
    return True


def _label_for_leaf(path: str, leaf: jnp.ndarray, labels: LabelConfig) -> str:
    # spectral group first (optional)
    if labels.spectral_lr_mult and labels.spectral_lr_mult != 1.0:
        sm = labels.spectral_matcher or _default_spectral_matcher
        if sm(path, leaf):
            return "spectral"

    # head group (optional)
    if labels.head_lr_mult and labels.head_lr_mult != 1.0:
        hm = labels.head_matcher or _default_head_matcher
        if hm(path, leaf):
            return "head"

    # default group
    return "default"


def _labels_and_decay_mask(params_tree: Any, cfg: OptimizerConfig) -> Tuple[Any, Any]:
    """
    Returns:
      labels_tree: pytree of strings in {"default","head","spectral"} (as needed)
      decay_mask_tree: pytree of bools; True means apply decoupled weight decay
    """
    with_path = jtu.tree_flatten_with_path(params_tree)
    leaves_with_path, treedef = with_path

    labels_leaves: list[str] = []
    decay_mask_leaves: list[bool] = []

    for path_entries, leaf in leaves_with_path:
        path = _path_to_str(path_entries)
        labels_leaves.append(_label_for_leaf(path, leaf, cfg.labels))
        decay_mask_leaves.append(_should_decay(path, leaf, cfg.decay_mask))

    labels_tree = jtu.tree_unflatten(treedef, labels_leaves)
    decay_mask_tree = jtu.tree_unflatten(treedef, decay_mask_leaves)
    return labels_tree, decay_mask_tree


def _make_group_optimizer(
    algo: str,
    lr_sched: Callable[[int], float],
    cfg: OptimizerConfig,
) -> optax.GradientTransformation:
    """
    Create the per-group optimizer core (no weight decay here).
    Weight decay is added once outside via `optax.add_decayed_weights(mask=...)`.
    """
    if algo == "adamw":
        return optax.adamw(
            learning_rate=lr_sched,
            b1=cfg.adam_b1,
            b2=cfg.adam_b2,
            eps=cfg.adam_eps,
            weight_decay=0.0,  # decoupled decay applied separately with mask
        )
    elif algo == "adam":
        return optax.adam(
            learning_rate=lr_sched,
            b1=cfg.adam_b1,
            b2=cfg.adam_b2,
            eps=cfg.adam_eps,
        )
    elif algo in ("sgd", "sgdn"):
        return optax.sgd(
            learning_rate=lr_sched,
            momentum=cfg.sgd_momentum,
            nesterov=(algo == "sgdn"),
        )
    elif algo == "lion":
        return optax.lion(
            learning_rate=lr_sched,
            b1=cfg.lion_b1,
            b2=cfg.lion_b2,
            weight_decay=0.0,
        )
    elif algo == "adan":
        if not hasattr(optax, "adan"):
            raise ValueError(
                "Requested algorithm 'adan' but optax.adan is not available in this environment."
            )
        return optax.adan(
            learning_rate=lr_sched,
            b1=cfg.adan_b1,
            b2=cfg.adan_b2,
            b3=cfg.adan_b3,
            eps=cfg.adan_eps,
            weight_decay=0.0,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo!r}")


def _normalize_prepend(
    prepend: Optional[
        Union[optax.GradientTransformation, Sequence[optax.GradientTransformation]]
    ],
) -> list[optax.GradientTransformation]:
    """Accept a single GradientTransformation or a sequence; return a list."""
    if prepend is None:
        return []
    if isinstance(prepend, (list, tuple)):
        return list(prepend)
    # duck-typing for a GradientTransformation
    if hasattr(prepend, "init") and hasattr(prepend, "update"):
        return [prepend]  # type: ignore[return-value]
    raise TypeError(
        "prepend must be an optax.GradientTransformation or a sequence of them."
    )


# ---------------------------------------------------------------------------
# Core: assemble optimizer
# ---------------------------------------------------------------------------


def build_optimizer(
    model_or_params: Any,
    cfg: OptimizerConfig,
    *,
    return_params_tree: bool = False,
    prepend: Optional[
        Union[list[optax.GradientTransformation], optax.GradientTransformation]
    ] = None,
) -> Tuple[optax.GradientTransformation, optax.OptState, Dict[str, Any]]:
    """
    Build an Optax optimizer pipeline with best-practice defaults:

      [prepend transforms ...]  →  AGC  →  global clip  →  masked decoupled WD  →  multi_transform(groups)

    Notes
    -----
    - AGC precedes global clipping (preferred).
    - Weight decay is decoupled and masked (bias/norm/spectral off by default).
    - Per-group LRs via `multi_transform`, guarded by passing a labels **callable**
      to avoid the callable-pytree trap with Equinox Modules.
    """
    # 1) arrays-only params tree
    if hasattr(model_or_params, "__class__") and not isinstance(
        model_or_params, (dict, list, tuple)
    ):
        params_tree, _ = eqx.partition(model_or_params, eqx.is_inexact_array)
    else:
        params_tree = model_or_params

    # 2) labels + decay mask
    labels_tree, decay_mask_tree = _labels_and_decay_mask(params_tree, cfg)

    if jtu.tree_structure(params_tree) != jtu.tree_structure(labels_tree):
        raise ValueError(
            "[build_optimizer] labels tree structure does not match params tree."
        )

    # Friendly warnings if groups requested but empty
    groups_present = set(jtu.tree_leaves(labels_tree))
    if (
        cfg.labels.head_lr_mult
        and cfg.labels.head_lr_mult != 1.0
        and "head" not in groups_present
    ):
        print(
            "[build_optimizer] Warning: head_lr_mult set but no parameters matched the 'head' group."
        )
    if (
        cfg.labels.spectral_lr_mult
        and cfg.labels.spectral_lr_mult != 1.0
        and "spectral" not in groups_present
    ):
        print(
            "[build_optimizer] Warning: spectral_lr_mult set but no parameters matched the 'spectral' group."
        )

    # 3) compose optimizer
    chain: list[optax.GradientTransformation] = []
    chain.extend(_normalize_prepend(prepend))

    # **Preferred order:** AGC → global clip
    if cfg.agc_clip is not None:
        chain.append(optax.adaptive_grad_clip(cfg.agc_clip))
    if cfg.clip_global_norm is not None:
        chain.append(optax.clip_by_global_norm(cfg.clip_global_norm))

    if cfg.weight_decay and cfg.weight_decay > 0.0:
        chain.append(optax.add_decayed_weights(cfg.weight_decay, mask=decay_mask_tree))

    base_sched = _safe_lr_schedule(cfg.lr)
    groups: Dict[str, optax.GradientTransformation] = {
        "default": _make_group_optimizer(cfg.algorithm, base_sched, cfg)
    }
    if cfg.labels.spectral_lr_mult and cfg.labels.spectral_lr_mult != 1.0:
        groups["spectral"] = _make_group_optimizer(
            cfg.algorithm, _mul_schedule(base_sched, cfg.labels.spectral_lr_mult), cfg
        )
    if cfg.labels.head_lr_mult and cfg.labels.head_lr_mult != 1.0:
        groups["head"] = _make_group_optimizer(
            cfg.algorithm, _mul_schedule(base_sched, cfg.labels.head_lr_mult), cfg
        )

    # Guard multi_transform by passing a callable that RETURNS the labels pytree.
    chain.append(optax.multi_transform(groups, lambda _: labels_tree))

    opt = optax.chain(*chain)
    opt_state = opt.init(params_tree)

    aux = {
        "labels": labels_tree,
        "decay_mask": decay_mask_tree,
        "summary": _summarize_labels_and_decay(
            params_tree, labels_tree, decay_mask_tree
        ),
    }
    if return_params_tree:
        aux["params"] = params_tree
    return opt, opt_state, aux
