from __future__ import annotations

"""Public surface for :mod:`quantbayes.ball_dp.attacks`.

The finite-prior support construction helpers are deliberately NumPy-only and are
used by lightweight demos.  The model-based and gradient-based attacks require
optional JAX/Equinox/Optax training dependencies.  Importing this package should
therefore not fail merely because those optional dependencies are unavailable;
heavy APIs are replaced by clear stubs in that case.
"""

from .finite_prior_setup import (
    CandidateSource,
    CandidateBank,
    FinitePriorSupport,
    FinitePriorTrial,
    build_same_label_ball_bank,
    find_feasible_replacement_banks,
    select_support_from_bank,
    target_positions_for_support,
    make_replacement_trial,
    make_replacement_trials_for_support,
    support_to_records,
    trial_true_record,
    enrich_attack_result_with_trial,
    remove_index,
    append_candidate_at_end,
    support_source_hash,
    finite_support_hash,
)


def _make_missing_optional_api(name: str, exc: BaseException):
    def _missing_optional_api(*args, **kwargs):
        del args, kwargs
        raise ImportError(
            f"quantbayes.ball_dp.attacks.{name} requires optional training "
            f"dependencies that are unavailable: {exc}"
        ) from exc

    _missing_optional_api.__name__ = name
    _missing_optional_api.__qualname__ = name
    _missing_optional_api.__doc__ = (
        f"Placeholder for {name}; install the optional training dependency stack "
        "to use this API."
    )
    return _missing_optional_api


_MODEL_BASED_NAMES = (
    "FlatRecordCodec",
    "ParametersOnlyFeatureMap",
    "build_shadow_corpus",
    "make_attack_feature_map",
    "run_model_based_attack",
    "train_shadow_reconstructor",
)

try:  # pragma: no cover - depends on optional ML stack.
    from .model_based import (
        FlatRecordCodec,
        ParametersOnlyFeatureMap,
        build_shadow_corpus,
        make_attack_feature_map,
        run_model_based_attack,
        train_shadow_reconstructor,
    )
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    for _name in _MODEL_BASED_NAMES:
        globals()[_name] = _make_missing_optional_api(_name, exc)


_GRADIENT_BASED_NAMES = (
    "DPSGDTrace",
    "DPSGDTraceRecorder",
    "DPSGDTraceStep",
    "TraceOptimizationAttackConfig",
    "run_prior_aware_trace_attack",
    "run_trace_optimization_attack",
    "subtract_known_batch_gradients",
)

try:  # pragma: no cover - depends on optional ML stack.
    from .gradient_based import (
        DPSGDTrace,
        DPSGDTraceRecorder,
        DPSGDTraceStep,
        TraceOptimizationAttackConfig,
        run_prior_aware_trace_attack,
        run_trace_optimization_attack,
        subtract_known_batch_gradients,
    )
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    for _name in _GRADIENT_BASED_NAMES:
        globals()[_name] = _make_missing_optional_api(_name, exc)


__all__ = [
    "CandidateSource",
    "CandidateBank",
    "FinitePriorSupport",
    "FinitePriorTrial",
    "build_same_label_ball_bank",
    "find_feasible_replacement_banks",
    "select_support_from_bank",
    "target_positions_for_support",
    "make_replacement_trial",
    "make_replacement_trials_for_support",
    "support_to_records",
    "trial_true_record",
    "enrich_attack_result_with_trial",
    "remove_index",
    "append_candidate_at_end",
    "support_source_hash",
    "finite_support_hash",
    *_MODEL_BASED_NAMES,
    *_GRADIENT_BASED_NAMES,
]
