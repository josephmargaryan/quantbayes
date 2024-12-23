import numpyro
from dual_moon_distribution import DualMoonDistribution


def dual_moon_model():
    numpyro.sample("x", DualMoonDistribution())
