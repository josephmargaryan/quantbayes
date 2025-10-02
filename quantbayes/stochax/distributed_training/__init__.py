from .centralized import CentralizedTrainer
from .decentralized import DecentralizedTrainer
from .augmented_decentralized import AugmentedDecentralizedTrainer
from .fedavg import FederatedTrainer
from .dgd import (
    DGDTrainerEqx,
    centralized_gd_eqx,
    plot_global_loss_q3,
    plot_consensus_q3,
    safe_alpha,
    plot_q4_cases,
    plot_link_replacement,
)
from .dsgd_trainer_eqx import (
    DSGDTrainerEqx,
    DGDTrainerSwitchingEqx,
    make_batch_schedule_powerlaw,
    make_batch_schedule_piecewise,
    plot_dsgd_global_losses,
    plot_consensus,
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
    "safe_alpha",
    "plot_q4_cases",
    "plot_link_replacement",
    "DSGDTrainerEqx",
    "DGDTrainerSwitchingEqx",
    "make_batch_schedule_powerlaw",
    "make_batch_schedule_piecewise",
    "plot_dsgd_global_losses",
    "plot_consensus",
]
