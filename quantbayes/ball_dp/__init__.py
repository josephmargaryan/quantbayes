# quantbayes/ball_dp/__init__.py
"""
Ball-Adjacency Differential Privacy (Ball-DP)

This subpackage provides:
- The Ball adjacency relation and radius selection utilities.
- Sensitivity bounds for strongly convex ERM under Ball adjacency.
- Gaussian output perturbation calibration utilities.
- L_z bounds and empirical estimators.
- Threat-model-aligned auditing/attack utilities.
- Reproducible experiment scripts under quantbayes/ball_dp/experiments/.

Design goal: clean, reusable primitives with production-grade defaults.
"""

from .adjacency import BallAdjacency
from .mechanisms import gaussian_sigma, add_gaussian_noise
from .sensitivity import erm_sensitivity_l2
from .lz import (
    lz_prototypes_exact,
    lz_logistic_binary_bound,
    lz_softmax_linear_bound,
)
from .radius import (
    within_class_nn_distances,
    radii_from_percentiles,
    coverage_curve,
    dp_quantile_exponential_mechanism,
)
from .api import (
    EmbeddingBundle,
    RadiusPolicy,
    embed_torch_dataloaders,
    save_embeddings_npz,
    load_embeddings_npz,
    compute_radius_policy,
    dp_release_erm_params_gaussian,
)


__all__ = [
    "BallAdjacency",
    "gaussian_sigma",
    "add_gaussian_noise",
    "erm_sensitivity_l2",
    "lz_prototypes_exact",
    "lz_logistic_binary_bound",
    "lz_softmax_linear_bound",
    "within_class_nn_distances",
    "radii_from_percentiles",
    "coverage_curve",
    "dp_quantile_exponential_mechanism",
    "EmbeddingBundle",
    "RadiusPolicy",
    "embed_torch_dataloaders",
    "save_embeddings_npz",
    "load_embeddings_npz",
    "compute_radius_policy",
    "dp_release_erm_params_gaussian",
]
