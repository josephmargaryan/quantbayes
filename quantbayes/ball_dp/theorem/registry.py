from __future__ import annotations

import dataclasses as dc
import importlib
from typing import Any, Optional

from .specs import TheoremBounds, TheoremModelSpec


@dc.dataclass(frozen=True)
class RegistryEntry:
    module: str
    make_name: str
    constants_name: str
    projector_name: str
    check_name: str
    input_check_name: str = "check_input_bound"
    from_dense_name: Optional[str] = None


_REGISTRY = {
    ("binary", "dense", "fro"): RegistryEntry(
        module="quantbayes.ball_dp.nonconvex.models.ball_net",
        make_name="make_ball_tanh_net",
        constants_name="certified_tanh_mlp_constants",
        projector_name="make_ball_tanh_projector",
        check_name="check_ball_tanh_constraints",
    ),
    ("binary", "dense", "op"): RegistryEntry(
        module="quantbayes.ball_dp.nonconvex.models.ball_net_op",
        make_name="make_ball_tanh_op_net",
        constants_name="certified_tanh_mlp_op_constants",
        projector_name="make_ball_tanh_op_projector",
        check_name="check_ball_tanh_op_constraints",
    ),
    ("binary", "svd", "op"): RegistryEntry(
        module="quantbayes.ball_dp.nonconvex.models.ball_net_svd",
        make_name="make_ball_svd_tanh_net",
        constants_name="certified_tanh_svd_mlp_constants",
        projector_name="make_ball_svd_projector",
        check_name="check_ball_svd_constraints",
        from_dense_name="make_ball_svd_tanh_net_from_dense",
    ),
    ("multiclass", "dense", "fro"): RegistryEntry(
        module="quantbayes.ball_dp.nonconvex.models.ball_net_multiclass",
        make_name="make_ball_tanh_multiclass_net",
        constants_name="certified_tanh_mlp_multiclass_constants",
        projector_name="make_ball_tanh_multiclass_projector",
        check_name="check_ball_tanh_multiclass_constraints",
    ),
    ("multiclass", "dense", "op"): RegistryEntry(
        module="quantbayes.ball_dp.nonconvex.models.ball_net_op_multiclass",
        make_name="make_ball_tanh_op_multiclass_net",
        constants_name="certified_tanh_mlp_op_multiclass_constants",
        projector_name="make_ball_tanh_op_multiclass_projector",
        check_name="check_ball_tanh_op_multiclass_constraints",
    ),
    ("multiclass", "svd", "op"): RegistryEntry(
        module="quantbayes.ball_dp.nonconvex.models.ball_net_svd_multiclass",
        make_name="make_ball_svd_tanh_multiclass_net",
        constants_name="certified_tanh_svd_mlp_multiclass_constants",
        projector_name="make_ball_svd_multiclass_projector",
        check_name="check_ball_svd_multiclass_constraints",
        from_dense_name="make_ball_svd_tanh_multiclass_net_from_dense",
    ),
}


def _entry(spec: TheoremModelSpec) -> RegistryEntry:
    key = (spec.task, spec.parameterization, spec.constraint)
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported theorem family {key!r}.") from exc


def _module(spec: TheoremModelSpec):
    entry = _entry(spec)
    return importlib.import_module(entry.module), entry


def _make_kwargs(
    spec: TheoremModelSpec,
    *,
    key: Any,
    dtype: Any,
    init_project: bool,
    bounds: Optional[TheoremBounds],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "d_in": int(spec.d_in),
        "hidden_dim": int(spec.hidden_dim),
        "key": key,
        "dtype": dtype,
        "init_project": bool(init_project),
    }
    if spec.task == "multiclass":
        kwargs["num_classes"] = int(spec.num_classes)
    if spec.parameterization == "svd":
        kwargs["rank"] = None if spec.rank is None else int(spec.rank)
    if init_project:
        if bounds is None:
            raise ValueError("init_project=True requires theorem bounds.")
        kwargs.update(bounds.constraint_kwargs(spec))
    return kwargs


def make_model(
    spec: TheoremModelSpec,
    *,
    key: Any,
    dtype: Any = None,
    init_project: bool = False,
    bounds: Optional[TheoremBounds] = None,
) -> Any:
    module, entry = _module(spec)
    make_fn = getattr(module, entry.make_name)
    return make_fn(
        **_make_kwargs(
            spec, key=key, dtype=dtype, init_project=init_project, bounds=bounds
        )
    )


def replace_dense_with_svd(
    dense_model: Any,
    svd_spec: TheoremModelSpec,
    *,
    init_project: bool = False,
    bounds: Optional[TheoremBounds] = None,
) -> Any:
    if svd_spec.parameterization != "svd":
        raise ValueError("replace_dense_with_svd requires an SVD target spec.")
    module, entry = _module(svd_spec)
    if entry.from_dense_name is None:
        raise ValueError(
            "This theorem family does not expose a dense->SVD conversion helper."
        )
    from_dense = getattr(module, entry.from_dense_name)
    kwargs = {
        "rank": None if svd_spec.rank is None else int(svd_spec.rank),
        "init_project": bool(init_project),
    }
    if init_project:
        if bounds is None:
            raise ValueError("init_project=True requires theorem bounds.")
        kwargs.update(bounds.constraint_kwargs(svd_spec))
    return from_dense(dense_model, **kwargs)


def certified_constants(
    spec: TheoremModelSpec, bounds: TheoremBounds
) -> dict[str, float]:
    module, entry = _module(spec)
    bounds.validate_for(spec)
    const_fn = getattr(module, entry.constants_name)
    return const_fn(**bounds.certificate_kwargs(spec))


def certified_lz(spec: TheoremModelSpec, bounds: TheoremBounds) -> float:
    return float(certified_constants(spec, bounds)["L_z"])


def make_projector(spec: TheoremModelSpec, bounds: TheoremBounds):
    module, entry = _module(spec)
    proj_fn = getattr(module, entry.projector_name)
    return proj_fn(**bounds.constraint_kwargs(spec))


def check_constraints(
    model: Any, spec: TheoremModelSpec, bounds: TheoremBounds, *, atol: float = 1e-6
) -> None:
    module, entry = _module(spec)
    check_fn = getattr(module, entry.check_name)
    kwargs = bounds.constraint_kwargs(spec)
    kwargs["atol"] = float(atol)
    check_fn(model, **kwargs)


def check_input_bound(
    X: Any,
    bounds: TheoremBounds,
    spec: TheoremModelSpec | None = None,
    *,
    atol: float = 1e-6,
) -> None:
    del spec
    # Any theorem family can validate the public input norm bound.
    module = importlib.import_module("quantbayes.ball_dp.nonconvex.models.ball_net")
    getattr(module, "check_input_bound")(X, B=float(bounds.B), atol=float(atol))
