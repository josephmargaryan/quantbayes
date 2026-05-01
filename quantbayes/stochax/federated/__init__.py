# quantbayes/stochax/federated/__init__.py

from .fedprox import FedProxTrainer
from .optimizers import FedAdamServer

__all__ = ["FedProxTrainer", "FedAdamServer"]
