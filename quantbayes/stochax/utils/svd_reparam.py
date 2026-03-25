# quantbayes/stochax/utils/svd_reparam.py
from __future__ import annotations
from typing import Optional, Any

import jax
import equinox as eqx
import jax.numpy as jnp

from quantbayes.stochax.layers.spectral_layers import SVDDense  # your class
from quantbayes.stochax.vision_common.spectral_surgery import replace_modules


"""
# 1) build vanilla model and load pretrained weights (your existing loader)
model = build_model(...)
model, report = load_imagenet_resnet(model, "resnet18_imagenet.npz", ...)

# 2) retrofit linears to SVDDense from the loaded weights
from quantbayes.stochax.utils.svd_reparam import retrofit_linear_to_svddense
model = retrofit_linear_to_svddense(
    model,
    default_rank=None,     # or an int to truncate
    freeze_UV=True,        # keep orthonormal bases fixed
    alpha_init=1.0,
)

# 3) build optimizer with freeze/decay masks as above

"""


def retrofit_linear_to_svddense(
    model: eqx.Module,
    *,
    default_rank: Optional[int] = None,  # None = full rank
    freeze_UV: bool = True,
    alpha_init: float = 1.0,
) -> eqx.Module:
    """Replace all eqx.nn.Linear with SVDDense.from_linear(lin, rank=..., alpha_init=...)."""

    def pred(x, _path):
        import equinox.nn as nn

        return isinstance(x, getattr(nn, "Linear", ()))

    def build(lin, _path):
        svd = SVDDense.from_linear(lin, rank=default_rank, alpha_init=alpha_init)
        if freeze_UV:
            svd = eqx.tree_at(lambda m: m.U, svd, jax.lax.stop_gradient(svd.U))
            svd = eqx.tree_at(lambda m: m.V, svd, jax.lax.stop_gradient(svd.V))
        return svd

    return replace_modules(model, pred, build)
