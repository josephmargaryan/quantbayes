"""Public package surface for :mod:`quantbayes.ball_dp`.

The centralized/nonconvex API uses optional JAX ecosystem dependencies such as
``equinox`` and ``optax``.  The decentralized accounting path is intentionally much
lighter, so package import should not fail merely because those optional training
dependencies are absent.  When the heavy API cannot be imported, its public names are
replaced by stubs that raise a clear ImportError only if called.
"""

from __future__ import annotations

_API_NAMES = [
    "fit_convex",
    "attack_convex",
    "attack_convex_ball_output",
    "attack_convex_ball_output_finite_prior",
    "attack_convex_finite_prior_trial",
    "diagnose_convex_ball_output_finite_prior",
    "ball_rero",
    "fit_ball_sgd",
    "attack_nonconvex_ball_trace_finite_prior",
    "attack_nonconvex_finite_prior_trial",
    "account_ball_sgd_noise_multiplier",
    "calibrate_ball_sgd_noise_multiplier",
    "extract_privacy_epsilon",
    "calibrate_privacy_parameter",
    "evaluate_release_classifier",
    "make_trace_metadata_from_release",
    "summarize_attack_trials",
    "make_uniform_ball_prior",
    "make_uniform_ball_attack_prior",
    "make_truncated_gaussian_ball_attack_prior",
    "make_finite_identification_prior",
    "make_empirical_ball_prior",
    "summarize_embedding_ball_radii",
    "select_ball_radius",
    "get_public_curve_history",
    "get_operator_norm_history",
    "get_release_step_table",
]

_API_IMPORT_ERROR = None
try:  # pragma: no cover - exercised in environments with the full ML stack.
    from .api import (
        fit_convex,
        attack_convex,
        attack_convex_ball_output,
        attack_convex_ball_output_finite_prior,
        attack_convex_finite_prior_trial,
        diagnose_convex_ball_output_finite_prior,
        ball_rero,
        fit_ball_sgd,
        attack_nonconvex_ball_trace_finite_prior,
        attack_nonconvex_finite_prior_trial,
        account_ball_sgd_noise_multiplier,
        calibrate_ball_sgd_noise_multiplier,
        extract_privacy_epsilon,
        calibrate_privacy_parameter,
        evaluate_release_classifier,
        make_trace_metadata_from_release,
        summarize_attack_trials,
        make_uniform_ball_prior,
        make_uniform_ball_attack_prior,
        make_truncated_gaussian_ball_attack_prior,
        make_finite_identification_prior,
        make_empirical_ball_prior,
        summarize_embedding_ball_radii,
        select_ball_radius,
        get_public_curve_history,
        get_operator_norm_history,
        get_release_step_table,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard.
    _API_IMPORT_ERROR = exc
    _exc = exc

    def _missing_optional_api(*args, _exc=_exc, **kwargs):
        del args, kwargs
        raise ImportError(
            "The requested ball_dp API function requires optional training dependencies "
            f"that are not installed: {_exc.name!r}. Install the full ML dependency set "
            "or import a lighter submodule such as quantbayes.ball_dp.decentralized."
        ) from _exc

    for _name in _API_NAMES:
        globals()[_name] = _missing_optional_api

try:  # Plot helpers are optional for headless/accounting-only use.
    from .plots import (
        plot_convex_attack_result,
        plot_convex_model_parameters,
        plot_linear_model_weights,
        plot_operator_norm_history,
        plot_reconstruction_pair,
        plot_release_curves,
        plot_rero_report,
        plot_ridge_prototypes,
    )
except Exception:  # pragma: no cover

    def _missing_plot(*args, **kwargs):
        del args, kwargs
        raise ImportError(
            "Plot helpers require matplotlib and the plotting dependency stack."
        )

    plot_convex_attack_result = _missing_plot
    plot_convex_model_parameters = _missing_plot
    plot_linear_model_weights = _missing_plot
    plot_operator_norm_history = _missing_plot
    plot_reconstruction_pair = _missing_plot
    plot_release_curves = _missing_plot
    plot_rero_report = _missing_plot
    plot_ridge_prototypes = _missing_plot

from .types import (
    ArrayDataset,
    AttackResult,
    Record,
    ReRoPoint,
    ReRoReport,
    ReleaseArtifact,
)

__all__ = [
    "fit_convex",
    "attack_convex",
    "attack_convex_ball_output",
    "attack_convex_ball_output_finite_prior",
    "attack_convex_finite_prior_trial",
    "diagnose_convex_ball_output_finite_prior",
    "ball_rero",
    "fit_ball_sgd",
    "attack_nonconvex_ball_trace_finite_prior",
    "attack_nonconvex_finite_prior_trial",
    "account_ball_sgd_noise_multiplier",
    "calibrate_ball_sgd_noise_multiplier",
    "extract_privacy_epsilon",
    "calibrate_privacy_parameter",
    "evaluate_release_classifier",
    "make_trace_metadata_from_release",
    "summarize_attack_trials",
    "make_uniform_ball_prior",
    "make_uniform_ball_attack_prior",
    "make_truncated_gaussian_ball_attack_prior",
    "make_finite_identification_prior",
    "make_empirical_ball_prior",
    "summarize_embedding_ball_radii",
    "select_ball_radius",
    "get_public_curve_history",
    "get_operator_norm_history",
    "get_release_step_table",
    "plot_convex_attack_result",
    "plot_reconstruction_pair",
    "plot_release_curves",
    "plot_operator_norm_history",
    "plot_rero_report",
    "plot_ridge_prototypes",
    "plot_linear_model_weights",
    "plot_convex_model_parameters",
    "ArrayDataset",
    "Record",
    "ReleaseArtifact",
    "AttackResult",
    "ReRoPoint",
    "ReRoReport",
]
