from setuptools import setup, find_packages

setup(
    name="quantbayes",
    version="0.1.0",
    description="A library for probabilistic and Bayesian machine learning models",
    author="Joseph Margaryan",
    author_email="josephmargaryan@gmail.com",
    url="https://github.com/josephmargaryan/quantbayes",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "numpyro",
        "torch",
        "matplotlib",
        "seaborn",
        "flax",
        "pandas",
        "optax",
        "diffrax",
        "funsor",
        "einops",
    ],
    python_requires=">=3.7",
)
