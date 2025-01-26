# quantbayes/bnn/__init__.py

# Import specific classes from layers.py to expose them clearly
from .layers import (
    Linear,
    FFTLinear,
    ParticleLinear,
    FFTParticleLinear,
    Conv1d,
    Conv2d,
    SelfAttention,
    TransposedConv2d,
    FFTTransposedConv2d

)
from .layers.base import Module

# Import everything from AutoML (if needed)
from .AutoML import *

# Explicitly define what this module exposes
__all__ = [
    "Linear",
    "FFTLinear",
    "ParticleLinear",
    "FFTParticleLinear",
    "Conv1d",
    "Conv2d",
    "SelfAttention",
    "Module",
    "TransposedConv2d",
    "FFTTransposedConv2d"
    # You can add more classes from AutoML as needed
]
