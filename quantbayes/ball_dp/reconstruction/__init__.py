# quantbayes/ball_dp/reconstruction/__init__.py
from .types import Candidate, ReconstructionResult, IdentificationResult
from .priors import PoolBallPrior, l2_metric_batch
from .vectorizers import (
    vectorize_prototypes,
    vectorize_softmax,
    vectorize_binary_linear,
    vectorize_eqx_model,
    l2_distance,
)

from .convex import (
    RidgePrototypesEquationSolver,
    SoftmaxEquationSolver,
    BinaryLogisticEquationSolver,
    SquaredHingeEquationSolver,
    GaussianOutputIdentifier,
)

from .nonconvex import (
    ShadowModelIdentifier,
    EqxTrainerConfig,
    EqxNonPrivateTrainer,
    EqxDPSGDTrainer,
    EqxBallDPSGDTrainer,
)
