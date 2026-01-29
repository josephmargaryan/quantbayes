# quantbayes/ball_dp/heads/logreg_torch.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class LogRegTorchConfig:
    """
    Softmax linear classifier trained on embeddings.

    We keep it intentionally simple and robust:
      - single Linear layer
      - cross-entropy
      - L2 regularization via weight_decay (AdamW) or explicit penalty

    For DP output perturbation experiments, the training is non-private.
    """

    num_classes: int = 10
    lr: float = 1e-2
    weight_decay: float = 1e-3
    epochs: int = 50
    batch_size: int = 2048
    device: str = "cuda"
    seed: int = 0


class SoftmaxLogReg(nn.Module):
    def __init__(self, d_in: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_in, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_softmax_logreg_torch(
    Ztr: np.ndarray,
    ytr: np.ndarray,
    Zva: Optional[np.ndarray],
    yva: Optional[np.ndarray],
    *,
    cfg: LogRegTorchConfig,
) -> Tuple[SoftmaxLogReg, Dict[str, float]]:
    """
    Train softmax logistic regression on embeddings.

    Returns:
      model (on CPU),
      metrics dict (train_acc, val_acc if val provided).
    """
    torch.manual_seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    Ztr_t = torch.from_numpy(Ztr).float()
    ytr_t = torch.from_numpy(ytr).long()

    d_in = int(Ztr.shape[1])
    model = SoftmaxLogReg(d_in=d_in, num_classes=int(cfg.num_classes)).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
    )
    loss_fn = nn.CrossEntropyLoss()

    # simple minibatch loader
    N = Ztr_t.shape[0]
    bs = int(cfg.batch_size)

    for ep in range(int(cfg.epochs)):
        model.train()
        # shuffle
        perm = torch.randperm(N)
        Ztr_ep = Ztr_t[perm]
        ytr_ep = ytr_t[perm]

        for s in range(0, N, bs):
            e = min(s + bs, N)
            xb = Ztr_ep[s:e].to(device)
            yb = ytr_ep[s:e].to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    # compute metrics
    model.eval()
    with torch.no_grad():
        train_logits = model(Ztr_t.to(device)).cpu()
        train_pred = train_logits.argmax(dim=1).numpy()
        train_acc = float((train_pred == ytr).mean())

        val_acc = float("nan")
        if Zva is not None and yva is not None:
            Zva_t = torch.from_numpy(Zva).float().to(device)
            val_logits = model(Zva_t).cpu()
            val_pred = val_logits.argmax(dim=1).numpy()
            val_acc = float((val_pred == yva).mean())

    # return model on cpu for easier noise addition
    model_cpu = model.to("cpu")
    return model_cpu, {"train_acc": train_acc, "val_acc": val_acc}


def predict_softmax_logreg_torch(model: SoftmaxLogReg, Z: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Z).float()).cpu().numpy()
    return logits.argmax(axis=1)
