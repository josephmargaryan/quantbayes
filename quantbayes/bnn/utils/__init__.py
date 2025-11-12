from .entropy_analysis import EntropyAndMutualInformation
from .generalization_bound import BayesianAnalysis
from .gp_utils import (
    predict_gp,
    predict_gp_binary,
    sample_gp_prior,
    visualize_gp_kernel,
    visualize_predictions,
    visualize_predictions_binary,
)
from .hdi_plot import plot_hdi
from .mcmc_metrics import evaluate_mcmc
from .model_calibration import (
    CalibratedBNN,
    expected_calibration_error,
    plot_calibration_curve,
    plot_roc_curve,
    maximum_calibration_error,
    multiclass_brier_score,
    multiclass_nll,
    binary_nll,
)
from .vis_classification import (
    visualize_uncertainty_binary,
    visualize_uncertainty_multiclass,
)
from .log_density import NLL

from .uncertainty import (
    avg_nll_from_proba,
    auroc_from_proba,
    brier_from_proba,
    ece_mce_from_proba,
    fpr95_from_proba,
    one_hot,
    predictive_entropy_from_proba,
    plot_id_vs_ood_entropy_kde,
    plot_id_vs_ood_entropy_kde_full_and_zoom,
    save_entropy_kde_pair,
    save_pred_entropy_from_arrays_trained_minimal,
)

__all__ = [
    "NLL",
    "CalibratedBNN",
    "EntropyAndMutualInformation",
    "BayesianAnalysis",
    "plot_roc_curve",
    "maximum_calibration_error",
    "multiclass_brier_score",
    "plot_calibration_curve",
    "multiclass_nll",
    "binary_nll",
    "expected_calibration_error",
    "plot_hdi",
    "evaluate_mcmc",
    "visualize_uncertainty_multiclass",
    "visualize_uncertainty_binary",
    "visualize_gp_kernel",
    "sample_gp_prior",
    "predict_gp",
    "predict_gp_binary",
    "visualize_predictions",
    "visualize_predictions_binary",
    "one_hot",
    "predictive_entropy_from_proba",
    "avg_nll_from_proba",
    "brier_from_proba",
    "ece_mce_from_proba",
    "auroc_from_proba",
    "fpr95_from_proba",
    "save_pred_entropy_from_arrays_trained_minimal",
    "plot_id_vs_ood_entropy_kde",
    "plot_id_vs_ood_entropy_kde_full_and_zoom",
    "save_entropy_kde_pair",
]
