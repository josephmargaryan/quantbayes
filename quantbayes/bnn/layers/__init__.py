# Import Module from base
from .base import Module

# Import all classes and the __all__ list from layers.py
from .layers import *  # This imports everything listed in layers.py's __all__


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
