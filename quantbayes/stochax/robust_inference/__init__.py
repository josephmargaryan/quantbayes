# quantbayes/stochax/robust_inference/__init__.py
"""
Public API for quantbayes.stochax.robust_inference
(Re-exports commonly used symbols so users can write
`from quantbayes.stochax.robust_inference import ...`)
"""

# Data
from .data import (
    load_mnist,
    load_cifar10,
    load_cifar100,
    load_synthetic,
    dirichlet_label_split,
    collect_probits_dataset,
)

# Clients
from .clients import train_clients

# Aggregators
from .aggregators import make_aggregator, DeepSetTM

# Aggregator training
from .agg_trainer import train_aggregator_erm, train_aggregator_robust

from quantbayes.stochax.robust_inference.eval import (
    aggregator_clean_acc,
    quick_attack_bench,
    aggregator_pgd_cw_acc,
    pgd_cw_vs_f,
)


__all__ = [
    # data
    "load_mnist",
    "load_cifar10",
    "load_cifar100",
    "load_synthetic",
    "dirichlet_label_split",
    "collect_probits_dataset",
    # clients
    "train_clients",
    # aggregators
    "make_aggregator",
    "DeepSetTM",
    # training
    "train_aggregator_erm",
    "train_aggregator_robust",
    # eval
    "aggregator_clean_acc",
    "quick_attack_bench",
    "aggregator_pgd_cw_acc",
    "pgd_cw_vs_f",
]
