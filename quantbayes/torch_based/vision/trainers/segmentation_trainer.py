import torch, numpy as np
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Callable
from .base import BaseTrainer
from ..utils import set_seed


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.a, self.b, self.g, self.s = alpha, beta, gamma, smooth

    def forward(self, logits, y):
        p = torch.sigmoid(logits)
        dims = (1, 2, 3)
        tp = (p * y).sum(dims)
        fp = ((1 - y) * p).sum(dims)
        fn = (y * (1 - p)).sum(dims)
        t = (tp + self.s) / (tp + self.a * fp + self.b * fn + self.s)
        return ((1 - t) ** self.g).mean()


def dice_score(logits, y, eps=1e-6):
    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p * y).sum((1, 2, 3))
    denom = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
    return ((2 * inter) / (denom + eps)).mean().item()


class SegmentationTrainer(BaseTrainer):
    """Thin wrapper providing sensible defaults."""

    def __init__(self, model, *, lr=1e-4, weight_decay=1e-5, epochs=40, seed=42, **kw):
        set_seed(seed)
        opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        super().__init__(
            model,
            loss_fn=FocalTverskyLoss(),
            optimizer=opt,
            scheduler=sched,
            monitor="val_loss",
            mode="min",
            **kw,
        )
