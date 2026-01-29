# quantbayes/ball_dp/heads/__init__.py
from .prototypes import (
    fit_ridge_prototypes,
    predict_nearest_prototype,
    prototypes_sensitivity_l2,
)
from .logreg_torch import (
    train_softmax_logreg_torch,
    predict_softmax_logreg_torch,
)

__all__ = [
    "fit_ridge_prototypes",
    "predict_nearest_prototype",
    "prototypes_sensitivity_l2",
    "train_softmax_logreg_torch",
    "predict_softmax_logreg_torch",
]
