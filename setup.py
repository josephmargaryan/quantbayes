from setuptools import setup, find_packages

setup(
    name="ChronosForge",  # Package name
    version="0.1.0",  # Version
    description="A library for probabilistic and Bayesian machine learning models",
    author="Joseph Margaryan",
    author_email="your-email@example.com",
    url="https://github.com/josephmargaryan/ChronosForge",  # GitHub URL
    packages=find_packages(),  # Automatically find all subpackages
    install_requires=[
        "numpy",
        "jax",
        "numpyro",
        "torch",
        "matplotlib",
        "pyro-ppl",
        "pandas",
    ],
    python_requires=">=3.7",  # Python version compatibility
)
