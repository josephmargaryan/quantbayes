from setuptools import setup, find_packages

setup(
    name="ChronosForge",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "jax",
        "numpyro",
        "torch",
        "matplotlib",
        "pandas",
        "pyro-ppl",
        "scikit-learn"
    ],
)

