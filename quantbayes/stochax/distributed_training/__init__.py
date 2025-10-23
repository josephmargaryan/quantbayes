# quantbayes/stochax/distributed_training/__init__.py

# Centralized / Federated
from .centralized import CentralizedTrainer
from .fedavg import FederatedTrainer

# Decentralized (primal–dual family)
from .decentralized import DecentralizedTrainer
from .augmented_decentralized import AugmentedDecentralizedTrainer
from .mixing_policies import repeat_mix, chebyshev_mix, disagreement_interval_from_L

# DGD (full gradient) — includes plotting helpers
from .dgd import (
    DGDTrainerEqx,
    centralized_gd_eqx,
    plot_global_loss_q3,
    plot_consensus_q3,
    plot_q4_cases,
    plot_link_replacement,
    safe_alpha as dgd_safe_alpha,  # alias to avoid clashing with DSGD's safe_alpha
)

# DSGD + Switching DGD — includes plotting helpers
from .dsgd_trainer_eqx import (
    DSGDTrainerEqx,
    make_batch_schedule_powerlaw,
    make_batch_schedule_piecewise,
    plot_dsgd_global_losses,
    plot_consensus,
    safe_alpha,  # re-export; name kept as 'safe_alpha'
)

from .peer_gossip_trainer_eqx import PeerGossipTrainerEqx

# Switching topologies DGD/DSGD
from .dgd_trainer_switching_eqx import DGDTrainerSwitchingEqx

# Periodic Local-GD + one-step P2P DGD gossip (assignment-style)
from .local_gd_server_eqx import (
    LocalGDServerEqx,
    make_star_with_server_edges,
    make_mixing_with_per_node_alphas,
    plot_server_loss,
    plot_consensus_localgd,
    make_constant_lr,
    make_polynomial_decay,
)

# Asynchronous Parameter Server
from .async_ps_trainer_eqx import AsyncParameterServerEqx

__all__ = [
    # Core trainers
    "CentralizedTrainer",
    "FederatedTrainer",
    "DecentralizedTrainer",
    "AugmentedDecentralizedTrainer",
    "DGDTrainerEqx",
    "DSGDTrainerEqx",
    "PeerGossipTrainerEqx",
    "DGDTrainerSwitchingEqx",
    "LocalGDServerEqx",
    "AsyncParameterServerEqx",
    # DGD helpers + plots
    "centralized_gd_eqx",
    "plot_global_loss_q3",
    "plot_consensus_q3",
    "plot_q4_cases",
    "plot_link_replacement",
    "dgd_safe_alpha",
    # DSGD helpers + plots
    "make_batch_schedule_powerlaw",
    "make_batch_schedule_piecewise",
    "plot_dsgd_global_losses",
    "plot_consensus",
    "safe_alpha",
    # Local-GD helpers + plots
    "make_star_with_server_edges",
    "make_mixing_with_per_node_alphas",
    "plot_server_loss",
    "plot_consensus_localgd",
    "make_constant_lr",
    "make_polynomial_decay",
]
