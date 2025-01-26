# quantbayes/bnn/__init__.py

# Import specific classes from layers.py to expose them clearly
from .layers import *
from .layers.base import Module

# Import everything from AutoML (if needed)
from .AutoML import *

# Explicitly define what this module exposes
__all__ = [
    "Module",
    "Linear",  # Add your other classes from layers.py
    "FFTLinear",
    "ParticleLinear",
    "FFTParticleLinear",
    "Conv1d",
    "Conv2d",
    "SelfAttention",
    "TransposedConv2d",
    "FFTTransposedConv2d",
    "MaxPool2d",
    "FFTConv2d"
    # You can list other classes or functions from layers.py as needed
]