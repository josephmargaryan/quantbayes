import torch, numpy as np
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ..utils import set_seed
from .base import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def __init__(self, model, *, lr=1e-4, weight_decay=1e-4, epochs=30, seed=42, **kw):
        set_seed(seed)
        opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        super().__init__(
            model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=opt,
            scheduler=sched,
            monitor="val_loss",
            mode="min",
            **kw,
        )
