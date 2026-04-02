from __future__ import annotations

from typing import Any, Literal, Optional

import equinox as eqx
import jax
import optax

from .registry import certified_lz, check_input_bound, make_projector
from .specs import TheoremBounds, TheoremModelSpec, TrainConfig


TrainableMode = Literal["default", "all", "s_only"]


def _freeze_basis_mask(model: Any) -> Any:
    mask = jax.tree_util.tree_map(lambda _: False, model)
    mask = eqx.tree_at(lambda m: m.hidden.U, mask, True)
    mask = eqx.tree_at(lambda m: m.hidden.V, mask, True)
    return mask


def _s_only_mask(model: Any) -> Any:
    mask = jax.tree_util.tree_map(lambda _: True, model)
    mask = eqx.tree_at(lambda m: m.hidden.s, mask, False)
    return mask


def make_optimizer(
    model: Any,
    spec: TheoremModelSpec,
    *,
    learning_rate: float,
    trainable: TrainableMode = "default",
) -> optax.GradientTransformation:
    if trainable not in {"default", "all", "s_only"}:
        raise ValueError("trainable must be one of {'default', 'all', 's_only'}.")

    lr = float(learning_rate)
    if spec.parameterization != "svd":
        if trainable not in {"default", "all"}:
            raise ValueError(
                "s_only is only meaningful for the fixed-basis SVD models."
            )
        return optax.adam(lr)

    if trainable == "all":
        return optax.adam(lr)
    if trainable == "s_only":
        return optax.chain(
            optax.masked(optax.set_to_zero(), _s_only_mask(model)),
            optax.adam(lr),
        )
    return optax.chain(
        optax.masked(optax.set_to_zero(), _freeze_basis_mask(model)),
        optax.adam(lr),
    )


def fit_release(
    model: Any,
    spec: TheoremModelSpec,
    bounds: TheoremBounds,
    X_train: Any,
    y_train: Any,
    *,
    train_cfg: TrainConfig,
    X_eval: Optional[Any] = None,
    y_eval: Optional[Any] = None,
    optimizer: Optional[optax.GradientTransformation] = None,
    trainable: TrainableMode = "default",
    state: Any = None,
    param_projector: Any = "default",
    key: Any = None,
    return_debug_history: bool = False,
    **extra_fit_kwargs: Any,
):
    from quantbayes.ball_dp.api import fit_ball_sgd

    check_input_bound(X_train, bounds)
    if X_eval is not None:
        check_input_bound(X_eval, bounds)

    if optimizer is None:
        optimizer = make_optimizer(
            model,
            spec,
            learning_rate=float(train_cfg.learning_rate),
            trainable=trainable,
        )

    if param_projector == "default":
        param_projector = make_projector(spec, bounds)
    elif param_projector is False:
        param_projector = None

    lz = certified_lz(spec, bounds)

    fit_kwargs = train_cfg.as_fit_kwargs()
    fit_kwargs.update(extra_fit_kwargs)
    fit_kwargs.update(
        {
            "radius": float(train_cfg.radius),
            "lz": float(lz),
            "loss_name": spec.loss_name,
            "state": state,
            "param_projector": param_projector,
            "key": key,
            "return_debug_history": bool(return_debug_history),
        }
    )

    return fit_ball_sgd(
        model,
        optimizer,
        X_train,
        y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        **fit_kwargs,
    )
