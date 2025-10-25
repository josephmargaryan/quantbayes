# quantbayes/stochax/distributed_training/__init__.py

# Centralized / Federated
from .centralized import CentralizedTrainer
from .fedavg import FederatedTrainer

# Decentralized (primal–dual family)
from .decentralized import DecentralizedTrainer
from .augmented_decentralized import AugmentedDecentralizedTrainer

# Switching topologies DGD/DSGD
from .dgd_trainer_switching_eqx import DGDTrainerSwitchingEqx

from .p2p_theory_trainer_eqx import P2PTheoryTrainerEqx
from .star_theory_trainer_eqx import StarTheoryTrainerEqx

# Asynchronous Parameter Server
from .async_ps_trainer_eqx import AsyncParameterServerEqx

__all__ = [
    # Core trainers
    "CentralizedTrainer",
    "FederatedTrainer",
    "DecentralizedTrainer",
    "AugmentedDecentralizedTrainer",
    "DGDTrainerSwitchingEqx",
    "AsyncParameterServerEqx",
    "P2PTheoryTrainerEqx",
    "StarTheoryTrainerEqx",
]
