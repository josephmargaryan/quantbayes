# quantbayes/ball_dp/reconstruction/convex/__init__.py
from .equation_solvers import (
    RidgePrototypesEquationSolver,
    SoftmaxEquationSolver,
    BinaryLogisticEquationSolver,
    SquaredHingeEquationSolver,
)
from .gaussian_identifier import GaussianOutputIdentifier
