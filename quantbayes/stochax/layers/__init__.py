"""Public re-exports for spectral layers used across stochax.

Historically several modules imported symbols directly from
``quantbayes.stochax.layers`` even though only ``spectral_layers.py`` existed.
This ``__init__`` file restores that import surface and keeps the public API
stable for vision, diffusion, and VAE workflows.
"""

from .spectral_layers import (
    AdaptiveSpectralCirculantLayer,
    AdaptiveSpectralDense,
    GraphChebSobolev,
    RFFTCirculant1D,
    RFFTCirculant2D,
    SpectralCirculantLayer,
    SpectralCirculantLayer2d,
    SpectralConv2d,
    SpectralDense,
    SpectralTokenMixer,
    SVDDense,
)

__all__ = [
    "AdaptiveSpectralCirculantLayer",
    "AdaptiveSpectralDense",
    "GraphChebSobolev",
    "RFFTCirculant1D",
    "RFFTCirculant2D",
    "SpectralCirculantLayer",
    "SpectralCirculantLayer2d",
    "SpectralConv2d",
    "SpectralDense",
    "SpectralTokenMixer",
    "SVDDense",
]
