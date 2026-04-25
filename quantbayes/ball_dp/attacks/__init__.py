"""Attack helpers.

Heavy learned-reconstructor and gradient attacks require optional ML dependencies.
The lightweight prior objects in ``ball_priors`` are still importable when those
optional dependencies are absent.
"""

from __future__ import annotations

_HEAVY_NAMES = [
    "FlatRecordCodec",
    "ParametersOnlyFeatureMap",
    "build_shadow_corpus",
    "make_attack_feature_map",
    "run_model_based_attack",
    "train_shadow_reconstructor",
    "DPSGDTrace",
    "DPSGDTraceRecorder",
    "DPSGDTraceStep",
    "TraceOptimizationAttackConfig",
    "run_prior_aware_trace_attack",
    "run_trace_optimization_attack",
    "subtract_known_batch_gradients",
]

try:  # pragma: no cover - exercised with full ML dependencies installed.
    from .model_based import (
        FlatRecordCodec,
        ParametersOnlyFeatureMap,
        build_shadow_corpus,
        make_attack_feature_map,
        run_model_based_attack,
        train_shadow_reconstructor,
    )
    from .gradient_based import (
        DPSGDTrace,
        DPSGDTraceRecorder,
        DPSGDTraceStep,
        TraceOptimizationAttackConfig,
        run_prior_aware_trace_attack,
        run_trace_optimization_attack,
        subtract_known_batch_gradients,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    _exc = exc

    def _missing_heavy_attack(*args, _exc=_exc, **kwargs):
        del args, kwargs
        raise ImportError(
            "This attack helper requires optional ML dependencies that are not installed: "
            f"{_exc.name!r}. The finite-prior and decentralized Gaussian MAP attacks remain available."
        ) from _exc

    for _name in _HEAVY_NAMES:
        globals()[_name] = _missing_heavy_attack

__all__ = list(_HEAVY_NAMES)
