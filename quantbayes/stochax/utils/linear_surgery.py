from __future__ import annotations

"""Shared dense -> spectral retrofit utilities.

The vision package already exposed a successful dense -> ``SVDDense`` workflow.
This module lifts the generic parts into a reusable location so diffusion and VAE
code can use the same retrofit and freeze semantics without duplicating logic.

Design goals
------------
1. Operate on arbitrary Equinox pytrees.
2. Preserve behaviour exactly for SVD retrofits.
3. Support RFFT retrofits on square linear maps via the nearest-circulant
   projection used elsewhere in the library.
4. Produce explicit reports so experiment scripts can log which leaves were
   replaced, truncated, or skipped.
"""

from typing import Any, Callable, Iterable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from quantbayes.stochax.layers import RFFTCirculant1D, SVDDense
from quantbayes.stochax.utils.optim_util import make_freeze_mask

LinearPredicate = Callable[[str, Any], bool]
RetrofitVariant = Literal["dense", "svd", "rfft"]

__all__ = [
    "LinearPredicate",
    "RetrofitVariant",
    "replace_linears_with_svd",
    "replace_square_linears_with_rfft",
    "retrofit_linears",
    "make_svd_basis_freeze_mask",
    "make_s_only_freeze_mask",
    "path_to_str",
]


def path_to_str(path_entries: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for pe in path_entries:
        if hasattr(pe, "name"):
            parts.append(str(pe.name))
        elif hasattr(pe, "key"):
            parts.append(str(pe.key))
        elif hasattr(pe, "idx"):
            parts.append(f"[{pe.idx}]")
        else:
            parts.append(str(pe))
    return ".".join(parts).replace(".[", "[")


def _is_dense_like_linear(node: Any) -> bool:
    if isinstance(node, eqx.nn.Linear):
        return True
    weight = getattr(node, "weight", None)
    bias = getattr(node, "bias", None)
    if weight is None:
        return False
    if not isinstance(weight, jnp.ndarray) or weight.ndim != 2:
        return False
    if bias is not None and (not isinstance(bias, jnp.ndarray) or bias.ndim != 1):
        return False
    name = type(node).__name__.lower()
    return "linear" in name


def _dense_weight_bias(node: Any) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    weight = getattr(node, "weight", None)
    if weight is None or not isinstance(weight, jnp.ndarray) or weight.ndim != 2:
        raise TypeError(f"Expected a dense-like linear leaf; got {type(node)!r}.")
    bias = getattr(node, "bias", None)
    if bias is not None and (not isinstance(bias, jnp.ndarray) or bias.ndim != 1):
        raise TypeError(
            f"Expected bias=None or a rank-1 array; got {type(bias)!r} for {type(node)!r}."
        )
    return weight, bias


def _full_rank(weight: jnp.ndarray) -> int:
    return int(min(weight.shape[0], weight.shape[1]))


def _resolve_rank(
    weight: jnp.ndarray,
    *,
    rank: int | None,
    rank_ratio: float | None,
) -> int:
    if rank is not None and rank_ratio is not None:
        raise ValueError("Pass at most one of rank= or rank_ratio=.")

    full_rank = _full_rank(weight)
    if rank is None and rank_ratio is None:
        return full_rank
    if rank is not None:
        return max(1, min(int(rank), full_rank))

    ratio = float(rank_ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"rank_ratio must lie in (0, 1], got {rank_ratio!r}.")
    return max(1, min(int(round(ratio * full_rank)), full_rank))


def _svddense_from_weight_bias(
    weight: jnp.ndarray,
    bias: jnp.ndarray | None,
    *,
    rank: int | None,
    rank_ratio: float | None,
    alpha_init: float,
) -> SVDDense:
    U, s, Vh = jnp.linalg.svd(weight, full_matrices=False)
    chosen_rank = _resolve_rank(weight, rank=rank, rank_ratio=rank_ratio)
    U = U[:, :chosen_rank]
    s = s[:chosen_rank]
    V = Vh[:chosen_rank, :].T
    bias_out = (
        bias if bias is not None else jnp.zeros((weight.shape[0],), dtype=weight.dtype)
    )
    return SVDDense(U=U, V=V, s=s, bias=bias_out, alpha_init=alpha_init)


def _nearest_circulant_first_column(weight: jnp.ndarray) -> jnp.ndarray:
    """Least-squares projection of a square matrix onto the circulant subspace.

    For a circulant matrix ``C`` with first column ``c``, the Frobenius-optimal
    ``c`` is the average over each cyclic diagonal.
    """

    if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError(f"Expected a square matrix; got {tuple(weight.shape)}.")
    n = int(weight.shape[0])
    rows = jnp.arange(n)
    cols = jnp.arange(n)

    def mean_for_offset(k):
        return jnp.mean(weight[rows, (cols - k) % n])

    return jax.vmap(mean_for_offset)(jnp.arange(n)).astype(weight.dtype)


def _rfft_from_weight_bias(
    weight: jnp.ndarray,
    bias: jnp.ndarray | None,
    *,
    alpha_init: float,
    key: Any,
    warmstart: bool,
) -> RFFTCirculant1D:
    out_features, in_features = map(int, weight.shape)
    if out_features != in_features:
        raise ValueError(
            "RFFT1D retrofits require square dense weights; "
            f"got {(out_features, in_features)}."
        )

    vector_bias = bias is not None
    mod = RFFTCirculant1D(
        in_features=in_features,
        padded_dim=in_features,
        crop_output=True,
        alpha_init=alpha_init,
        vector_bias=vector_bias,
        key=key,
    )

    if warmstart:
        c = _nearest_circulant_first_column(weight)
        h_half = (jnp.fft.rfft(c, norm="ortho") * jnp.sqrt(float(in_features))).astype(
            mod.H_half.dtype
        )
        mod = eqx.tree_at(lambda m: m.H_half, mod, h_half)
        if bias is not None:
            mod = eqx.tree_at(lambda m: m.bias, mod, bias.astype(mod.bias.dtype))
        else:
            mod = eqx.tree_at(lambda m: m.bias, mod, jnp.zeros_like(mod.bias))
    return mod


def _replaceable_leaves(model):
    return jtu.tree_flatten_with_path(model, is_leaf=_is_dense_like_linear)


def replace_linears_with_svd(
    model,
    *,
    rank: int | None = None,
    rank_ratio: float | None = None,
    alpha_init: float = 1.0,
    predicate: LinearPredicate | None = None,
):
    """Replace selected dense-like linear leaves with ``SVDDense``.

    By default this uses full rank and therefore preserves the represented dense
    map exactly up to SVD numerical precision.
    """

    predicate = (lambda _path, _leaf: True) if predicate is None else predicate
    leaves_with_path, treedef = _replaceable_leaves(model)

    new_leaves = []
    report = {
        "variant": "svd",
        "n_linear_total": 0,
        "n_replaced": 0,
        "n_skipped_by_predicate": 0,
        "replaced": [],
        "skipped": [],
    }

    for path_entries, leaf in leaves_with_path:
        if not _is_dense_like_linear(leaf):
            new_leaves.append(leaf)
            continue

        path = path_to_str(path_entries)
        report["n_linear_total"] += 1
        if not predicate(path, leaf):
            report["n_skipped_by_predicate"] += 1
            report["skipped"].append({"path": path, "reason": "predicate"})
            new_leaves.append(leaf)
            continue

        weight, bias = _dense_weight_bias(leaf)
        new_leaf = _svddense_from_weight_bias(
            weight,
            bias,
            rank=rank,
            rank_ratio=rank_ratio,
            alpha_init=alpha_init,
        )
        chosen_rank = int(new_leaf.s.shape[0])
        new_leaves.append(new_leaf)
        report["n_replaced"] += 1
        report["replaced"].append(
            {
                "path": path,
                "shape": tuple(int(x) for x in weight.shape),
                "rank": chosen_rank,
                "full_rank": _full_rank(weight),
                "truncated": chosen_rank < _full_rank(weight),
            }
        )

    new_model = jtu.tree_unflatten(treedef, new_leaves)
    return new_model, report


def replace_square_linears_with_rfft(
    model,
    *,
    alpha_init: float = 1.0,
    predicate: LinearPredicate | None = None,
    warmstart: bool = True,
    key: Any | None = None,
):
    """Replace selected square dense-like linear leaves with ``RFFTCirculant1D``.

    Parameters
    ----------
    warmstart:
        If ``True``, project each square dense weight onto the nearest circulant
        matrix and initialise the RFFT layer from that projection. If ``False``,
        new RFFT layers use their own random initialisation and ``key`` is
        required.
    """

    predicate = (lambda _path, _leaf: True) if predicate is None else predicate
    leaves_with_path, treedef = _replaceable_leaves(model)

    candidate_count = 0
    report = {
        "variant": "rfft",
        "warmstart": bool(warmstart),
        "n_linear_total": 0,
        "n_square_total": 0,
        "n_replaced": 0,
        "n_skipped_by_predicate": 0,
        "n_skipped_non_square": 0,
        "replaced": [],
        "skipped": [],
    }

    for path_entries, leaf in leaves_with_path:
        if not _is_dense_like_linear(leaf):
            continue
        report["n_linear_total"] += 1
        path = path_to_str(path_entries)
        if not predicate(path, leaf):
            report["n_skipped_by_predicate"] += 1
            report["skipped"].append({"path": path, "reason": "predicate"})
            continue
        weight, _ = _dense_weight_bias(leaf)
        if int(weight.shape[0]) != int(weight.shape[1]):
            report["n_skipped_non_square"] += 1
            report["skipped"].append(
                {
                    "path": path,
                    "reason": "non_square",
                    "shape": tuple(int(x) for x in weight.shape),
                }
            )
            continue
        report["n_square_total"] += 1
        candidate_count += 1

    if not warmstart and candidate_count > 0 and key is None:
        raise ValueError(
            "replace_square_linears_with_rfft(..., warmstart=False) requires key=."
        )

    replace_keys = (
        iter(jr.split(key, candidate_count))
        if candidate_count > 0 and key is not None
        else iter(())
    )
    new_leaves = []

    for path_entries, leaf in leaves_with_path:
        if not _is_dense_like_linear(leaf):
            new_leaves.append(leaf)
            continue

        path = path_to_str(path_entries)
        if not predicate(path, leaf):
            new_leaves.append(leaf)
            continue

        weight, bias = _dense_weight_bias(leaf)
        if int(weight.shape[0]) != int(weight.shape[1]):
            new_leaves.append(leaf)
            continue

        subkey = next(replace_keys, jr.PRNGKey(0))
        new_leaf = _rfft_from_weight_bias(
            weight,
            bias,
            alpha_init=alpha_init,
            key=subkey,
            warmstart=warmstart,
        )
        new_leaves.append(new_leaf)
        report["n_replaced"] += 1
        report["replaced"].append(
            {
                "path": path,
                "shape": tuple(int(x) for x in weight.shape),
                "bias": "vector" if bias is not None else "scalar_zero",
                "warmstart": bool(warmstart),
            }
        )

    new_model = jtu.tree_unflatten(treedef, new_leaves)
    return new_model, report


def retrofit_linears(
    model,
    *,
    variant: RetrofitVariant,
    rank: int | None = None,
    rank_ratio: float | None = None,
    alpha_init: float = 1.0,
    predicate: LinearPredicate | None = None,
    warmstart: bool = True,
    key: Any | None = None,
):
    """Variant-dispatch helper for dense/SVD/RFFT linears."""

    if variant == "dense":
        return model, {"variant": "dense", "n_replaced": 0, "replaced": []}
    if variant == "svd":
        return replace_linears_with_svd(
            model,
            rank=rank,
            rank_ratio=rank_ratio,
            alpha_init=alpha_init,
            predicate=predicate,
        )
    if variant == "rfft":
        return replace_square_linears_with_rfft(
            model,
            alpha_init=alpha_init,
            predicate=predicate,
            warmstart=warmstart,
            key=key,
        )
    raise ValueError(f"Unsupported retrofit variant: {variant!r}.")


def make_svd_basis_freeze_mask(model, *, freeze_alpha: bool = True):
    """Return an Optax-style freeze mask that zeros updates to ``U`` and ``V``.

    Set ``freeze_alpha=True`` to also freeze ``alpha_raw`` so the strict trainable
    set inside each ``SVDDense`` block is ``{s, bias}``.
    """

    names = ["U", "V"]
    if freeze_alpha:
        names.append("alpha_raw")
    return make_freeze_mask(model, names=tuple(names))


def make_s_only_freeze_mask(
    model,
    *,
    train_bias: bool = False,
    train_alpha: bool = False,
):
    """Return a freeze mask for the strict "train only ``s``" regime."""

    params_tree, _ = eqx.partition(model, eqx.is_inexact_array)
    leaves_with_path, treedef = jtu.tree_flatten_with_path(params_tree)

    masks = []
    for path_entries, leaf in leaves_with_path:
        if leaf is None:
            masks.append(False)
            continue

        tokens = set(path_to_str(path_entries).split("."))
        train_leaf = "s" in tokens
        if train_bias and "bias" in tokens:
            train_leaf = True
        if train_alpha and "alpha_raw" in tokens:
            train_leaf = True
        masks.append(not train_leaf)

    return jtu.tree_unflatten(treedef, masks)
