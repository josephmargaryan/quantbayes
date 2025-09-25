from .centralized import CentralizedTrainer
from .decentralized import DecentralizedTrainer
from .augmented_decentralized import AugmentedDecentralizedTrainer
from .fedavg import FederatedTrainer
from .dgd import (
    DGDTrainerEqx,
    centralized_gd_eqx,
    plot_global_loss_q3,
    plot_consensus_q3,
)

__all__ = [
    "CentralizedTrainer",
    "DecentralizedTrainer",
    "FederatedTrainer",
    "AugmentedDecentralizedTrainer",
    "DGDTrainerEqx",
    "centralized_gd_eqx",
    "plot_global_loss_q3",
    "plot_consensus_q3",
]
