# quantbayes/bnn/layers/__init__.py

# Import Module from base
from .base import Module  # Expose the Module class

# Import all classes from layers.py
from .layers import *  # Optionally, you could list specific imports here as well

# Explicitly define what this module exposes
__all__ = [
    "Module",  # Expose the Module class
    "Linear",  # Add your other classes from layers.py
    "FFTLinear",
    "ParticleLinear",
    "FFTParticleLinear",
    "Conv1d",
    "Conv2d",
    "SelfAttention"
    # You can list other classes or functions from layers.py as needed
]
