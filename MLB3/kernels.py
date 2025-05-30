"""
Extra kernels for optional experiments.
"""

from sklearn.metrics.pairwise import polynomial_kernel


def gaussian_gamma(a: float) -> float:
    """Convert assignment bandwidth a to scikit‑learn gamma."""
    return 1.0 / (2.0 * a**2)


def poly_kernel(X, Y=None, degree=3, coef0=1.0):
    """Thin wrapper around scikit‑learn's polynomial_kernel, for completeness."""
    return polynomial_kernel(X, Y, degree=degree, coef0=coef0)
