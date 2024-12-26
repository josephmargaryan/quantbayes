from setuptools import setup, find_packages

setup(
    name="ChronosForge",  
    version="0.1.0",  
    description="A library for probabilistic and Bayesian machine learning models",
    author="Joseph Margaryan",
    author_email="josephmargaryan@gmail.com",
    url="https://github.com/josephmargaryan/ChronosForge",  
    packages=find_packages(),  
    install_requires=[
        "numpy",
        "jax",
        "numpyro",
        "torch",
        "matplotlib",
        "pyro-ppl",
        "pandas",
    ],
    python_requires=">=3.7", 
)
