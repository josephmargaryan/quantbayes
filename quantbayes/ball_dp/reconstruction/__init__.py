# quantbayes/ball_dp/reconstruction/__init__.py
from __future__ import annotations

from .splits import InformedSplit, make_informed_split

from .convex_prototypes import (
    class_sums_counts_np,
    reconstruct_missing_from_prototypes_given_label,
)

from .convex_logreg import (
    to_pm1,
    logreg_grad_sum_np,
    reconstruct_missing_binary_logreg_from_release,
)

from .convex_softmax import (
    softmax_grad_sum_np,
    reconstruct_missing_softmax_from_release,
)

from .nonconvex_head_eqx import EmbeddingMLPClassifier, init_embedding_mlp_classifier

from .reconn_eqx import (
    WeightNormalizer,
    ReconstructorMLP,
    train_shadow_models_eqx,
    train_reconstructor_eqx,
    reconn_reconstruct_targets_eqx,
    nearest_neighbor_oracle_l2,
)

# from .audit_identification import (
#    candidate_set_within_radius,
#    proto_ml_identification_audit,
# )
