# quantbayes/bnn/__init__.py

# Import specific classes from layers.py to expose them clearly
from .layers import Linear, FFTLinear, ParticleLinear, FFTParticleLinear, Conv1d, Conv2d, SelfAttention

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
    # You can add more classes from AutoML as needed
]
