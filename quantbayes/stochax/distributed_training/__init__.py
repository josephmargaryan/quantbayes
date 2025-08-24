from .centralized import CentralizedTrainer
from .decentralized import DecentralizedTrainer
from .augmented_decentralized import AugmentedDecentralizedTrainer
from .fedavg import FederatedTrainer

__all__ = [
    "CentralizedTrainer",
    "DecentralizedTrainer",
    "FederatedTrainer",
    "AugmentedDecentralizedTrainer",
]
