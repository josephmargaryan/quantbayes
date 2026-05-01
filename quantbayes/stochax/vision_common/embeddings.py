from __future__ import annotations

"""Feature-extraction helpers for vision models.

This module keeps the main classifier API unchanged while making it easy to
obtain image embeddings from pretrained models.

The common pattern is:

1. instantiate the final model you want,
2. optionally load pretrained weights with :func:`load_pretrained`,
3. call :func:`extract_embeddings` to obtain the penultimate representation.

For most models, embeddings are defined as the activations immediately before
`fc`, `head`, or `fc3`:

- ResNet / ConvNeXt / EfficientNet / Inception: vector before `fc`
- ViT / Swin / DINO / ViT-ResNet hybrid: vector before `head`
- VGG: vector before `fc3`

The helper can also produce a feature-extractor model whose `__call__` returns
embeddings with the same `(x, key, state) -> (z, state)` contract as the normal
classifier.
"""

from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .pretrained_api import AutoFamily, Family, infer_pretrained_family


Array = jnp.ndarray


class IdentityHead(eqx.Module):
    """Simple identity replacement for a classifier head."""

    def __call__(self, x, *args, **kwargs):
        return x


class FeatureExtractor(eqx.Module):
    """Wrap a model exposing `forward_features` as the main callable."""

    model: Any

    def __call__(self, x, key, state):
        return self.model.forward_features(x, key, state)


def _head_attr_for_family(family: Family) -> str:
    mapping = {
        "resnet": "fc",
        "convnext": "fc",
        "efficientnet": "fc",
        "inception": "fc",
        "vit": "head",
        "swin": "head",
        "dino": "head",
        "vit_resnet_backbone": "head",
        "vgg": "fc3",
    }
    try:
        return mapping[family]
    except KeyError as exc:
        raise ValueError(
            f"Embeddings are not defined for family={family!r} by the generic helper."
        ) from exc


def _iter_tree(obj):
    yield obj
    if isinstance(obj, eqx.Module):
        for child in vars(obj).values():
            yield from _iter_tree(child)
    elif isinstance(obj, (list, tuple)):
        for child in obj:
            yield from _iter_tree(child)
    elif isinstance(obj, dict):
        for child in obj.values():
            yield from _iter_tree(child)


def _first_linear_in_tree(obj) -> Optional[eqx.nn.Linear]:
    for node in _iter_tree(obj):
        if isinstance(node, eqx.nn.Linear):
            return node
    return None


def _l2_normalize(z: Array, eps: float = 1e-12) -> Array:
    denom = jnp.linalg.norm(z, axis=-1, keepdims=True)
    return z / jnp.maximum(denom, eps)


@eqx.filter_jit
def _extract_batch(feature_model, state, X: Array, key: Array) -> Array:
    inference_model = eqx.nn.inference_mode(feature_model)

    def single(x, k):
        z, _ = inference_model(x, k, state)
        return z

    keys = jr.split(key, X.shape[0])
    return jax.vmap(single, in_axes=(0, 0))(X, keys)


def infer_embedding_dim(model, *, family: AutoFamily = "auto") -> int:
    """Infer the output embedding dimension of a classifier's penultimate layer."""
    fam = infer_pretrained_family(model) if family == "auto" else family
    if fam == "unet_encoder":
        raise ValueError(
            "Generic image embeddings are not defined for family='unet_encoder'."
        )

    attr = _head_attr_for_family(fam)
    head = getattr(model, attr, None)
    if head is None:
        raise ValueError(
            f"Model of type {type(model).__name__} has no expected classifier attribute {attr!r}."
        )

    if isinstance(head, eqx.nn.Linear):
        return int(head.weight.shape[1])

    linear = _first_linear_in_tree(head)
    if linear is not None:
        return int(linear.weight.shape[1])

    for name in ("embed_dim", "embedding_dim"):
        value = getattr(model, name, None)
        if value is not None:
            return int(value)

    raise ValueError(
        f"Could not infer embedding dimension for {type(model).__module__}.{type(model).__name__}."
    )


def as_feature_extractor(model, *, family: AutoFamily = "auto"):
    """Return a model whose call returns embeddings instead of logits.

    The returned model preserves the usual call contract:

        (x, key, state) -> (embedding, state)

    If the input model already implements `forward_features`, a lightweight
    wrapper is returned. Otherwise the classifier head is replaced with an
    identity map in a copied pytree.
    """
    forward_features = getattr(model, "forward_features", None)
    if callable(forward_features):
        return FeatureExtractor(model)

    fam = infer_pretrained_family(model) if family == "auto" else family
    if fam == "unet_encoder":
        raise ValueError(
            "Generic image embeddings are not defined for family='unet_encoder'."
        )

    attr = _head_attr_for_family(fam)
    if not hasattr(model, attr):
        raise ValueError(
            f"Model of type {type(model).__name__} does not expose the expected classifier attribute {attr!r}."
        )
    return eqx.tree_at(lambda m: getattr(m, attr), model, IdentityHead())


def extract_embeddings(
    model,
    state,
    X: Array,
    key: Optional[Array] = None,
    *,
    family: AutoFamily = "auto",
    batch_size: Optional[int] = None,
    l2_normalize: bool = False,
) -> Array:
    """Extract image embeddings from a model.

    Parameters
    ----------
    model, state:
        Standard Equinox model and mutable state.
    X:
        Either a single sample `[C, H, W]` (or more generally one unbatched
        model input) or a batch `[B, ...]`.
    key:
        PRNG key used for the underlying model call. If omitted, a deterministic
        zero key is used.
    family:
        Explicit family override when model-family inference is ambiguous.
    batch_size:
        Optional chunk size for large batches.
    l2_normalize:
        If `True`, return unit-norm embeddings along the last dimension.
    """
    key = jr.PRNGKey(0) if key is None else key
    feature_model = as_feature_extractor(model, family=family)

    if X.ndim == 0:
        raise ValueError(
            "X must be an array representing one sample or a batch of samples."
        )

    if batch_size is None:
        if X.ndim == 1 or X.ndim == 3:
            z, _ = eqx.nn.inference_mode(feature_model)(X, key, state)
        else:
            z = _extract_batch(feature_model, state, X, key)
    else:
        if X.ndim == 1 or X.ndim == 3:
            z, _ = eqx.nn.inference_mode(feature_model)(X, key, state)
        else:
            z = extract_embeddings_batched(
                feature_model,
                state,
                X,
                key,
                batch_size=batch_size,
                l2_normalize=False,
                _already_feature_extractor=True,
            )

    return _l2_normalize(z) if l2_normalize else z


def extract_embeddings_batched(
    model,
    state,
    X: Array,
    key: Array,
    *,
    batch_size: int = 256,
    family: AutoFamily = "auto",
    l2_normalize: bool = False,
    _already_feature_extractor: bool = False,
) -> Array:
    """Chunked embedding extraction for large batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if X.ndim < 2:
        raise ValueError(
            "extract_embeddings_batched expects a batched input with leading batch dimension."
        )

    feature_model = (
        model
        if _already_feature_extractor
        else as_feature_extractor(model, family=family)
    )

    N = X.shape[0]
    num_batches = (N + batch_size - 1) // batch_size
    batch_keys = jr.split(key, num_batches)
    parts = []
    start = 0
    for bk in batch_keys:
        end = min(start + batch_size, N)
        xb = X[start:end]
        parts.append(_extract_batch(feature_model, state, xb, bk))
        start = end

    z = jnp.concatenate(parts, axis=0)
    return _l2_normalize(z) if l2_normalize else z


__all__ = [
    "IdentityHead",
    "FeatureExtractor",
    "as_feature_extractor",
    "extract_embeddings",
    "extract_embeddings_batched",
    "infer_embedding_dim",
]
