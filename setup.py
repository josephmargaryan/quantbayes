from setuptools import find_packages, setup

setup(
    name="quantbayes",
    version="0.1.1",
    description="A library for probabilistic and Bayesian machine learning models",
    author="Joseph Margaryan",
    author_email="josephmargaryan@gmail.com",
    url="https://github.com/josephmargaryan/quantbayes",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.1.0,<2.2",  # Compatible with numpy 1.26.x
        "numpy>=1.26.2,<1.27",  # Ensures compatibility with pandas and other libraries
        "numpyro>=0.17.2,<0.19",  # Aligns with compatible jax versions
        "jax>=0.4.20,<0.5",  # Matches numpyro requirements
        "jaxlib>=0.4.20,<0.5",  # Ensures compatibility with jax
        "torch>=2.1.0,<2.2",  # Compatible with numpy versions below 2.0
        "scikit-learn>=1.5.0,<1.7",  # General compatibility range
        "matplotlib>=3.8.0,<3.11",  # Stable versions with necessary features
        "seaborn>=0.13.0,<0.14",  # Compatible with matplotlib versions
        "funsor>=0.4.0,<0.5",  # Aligns with numpyro requirements
        "diffrax>=0.6.0,<0.8",  # Compatible with jax versions
        "optax>=0.2.3,<0.3",  # Aligns with jax versions
        "einops>=0.6.0,<0.9",  # General compatibility range
        "arviz>=0.20.0,<0.22",  # Compatible with numpyro and jax
        "optuna>=4.0.0,<4.3",  # General compatibility range
        "xgboost>=2.0.0,<3.1",  # Ensures compatibility with numpy versions
        "catboost>=1.2.0,<1.3",  # General compatibility range
        "lightgbm>=4.0.0,<4.7",  # General compatibility range
        "equinox>=0.11.0,<0.12",  # Aligns with jax versions
    ],
    python_requires=">=3.8",
)
