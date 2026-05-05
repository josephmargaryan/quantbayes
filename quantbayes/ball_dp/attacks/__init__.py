from __future__ import annotations


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

from .finite_prior_eta import (
    FiniteSupportEtaGeometry,
    arbitrary_eta_bayes_decision,
    build_finite_support_eta_geometry,
    eta_grid_from_geometry,
    evaluate_eta_decoders,
    finite_support_kappa_rows,
    posterior_probabilities_from_attack_result,
    support_eta_bayes_decision,
)
