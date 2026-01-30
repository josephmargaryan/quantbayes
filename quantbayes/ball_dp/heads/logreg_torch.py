# quantbayes/ball_dp/heads/logreg_torch.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class LogRegTorchConfig:
    """
    Softmax linear classifier trained on embeddings as a convex, strongly-convex ERM:

      F(W,b) = (1/n) sum_i CE(softmax(W e_i + b), y_i)
               + (lam/2) (||W||_F^2 + ||b||_2^2)    [if regularize_bias=True]

    Key point: we include the L2 penalty explicitly in the objective so that:
      - the objective matches the paper,
      - 'lam' is the actual strong-convexity coefficient used for DP sensitivity.

    Optimizers:
      - 'lbfgs' (default) is the most faithful for convex ERM.
      - 'sgd' / 'adam' are fine for speed but are approximate solvers.
    """

    num_classes: int = 10

    lam: float = 1e-2
    regularize_bias: bool = True

    optimizer: Literal["lbfgs", "sgd", "adam"] = "lbfgs"

    # Used for LBFGS:
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 500

    # Used for SGD/Adam:
    lr: float = 1e-2
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


def _l2_penalty(model: SoftmaxLogReg, *, regularize_bias: bool) -> torch.Tensor:
    W = model.linear.weight
    b = model.linear.bias
    if regularize_bias:
        return (W * W).sum() + (b * b).sum()
    return (W * W).sum()


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
    if float(cfg.lam) <= 0.0:
        raise ValueError("cfg.lam must be > 0 for strong convexity.")

    torch.manual_seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    Ztr_t = torch.from_numpy(np.asarray(Ztr, dtype=np.float32)).to(device)
    ytr_t = torch.from_numpy(np.asarray(ytr, dtype=np.int64)).to(device)

    d_in = int(Ztr_t.shape[1])
    model = SoftmaxLogReg(d_in=d_in, num_classes=int(cfg.num_classes)).to(device)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    lam = float(cfg.lam)

    if cfg.optimizer == "lbfgs":
        opt = torch.optim.LBFGS(
            model.parameters(),
            lr=float(cfg.lbfgs_lr),
            max_iter=int(cfg.lbfgs_max_iter),
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            logits = model(Ztr_t)
            ce = loss_fn(logits, ytr_t)
            reg = (
                0.5
                * lam
                * _l2_penalty(model, regularize_bias=bool(cfg.regularize_bias))
            )
            loss = ce + reg
            loss.backward()
            return loss

        opt.step(closure)

    else:
        if cfg.optimizer == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=float(cfg.lr), momentum=0.9)
        elif cfg.optimizer == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
        else:
            raise ValueError(f"Unknown optimizer={cfg.optimizer!r}")

        N = int(Ztr_t.shape[0])
        bs = int(cfg.batch_size)

        for _ in range(int(cfg.epochs)):
            model.train()
            perm = torch.randperm(N, device=device)
            Zep = Ztr_t[perm]
            yep = ytr_t[perm]

            for s in range(0, N, bs):
                e = min(s + bs, N)
                xb = Zep[s:e]
                yb = yep[s:e]
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                ce = loss_fn(logits, yb)
                reg = (
                    0.5
                    * lam
                    * _l2_penalty(model, regularize_bias=bool(cfg.regularize_bias))
                )
                loss = ce + reg
                loss.backward()
                opt.step()

    # compute metrics
    model.eval()
    with torch.no_grad():
        train_logits = model(Ztr_t).detach().cpu()
        train_pred = train_logits.argmax(dim=1).numpy()
        train_acc = float((train_pred == np.asarray(ytr)).mean())

        val_acc = float("nan")
        if Zva is not None and yva is not None:
            Zva_t = torch.from_numpy(np.asarray(Zva, dtype=np.float32)).to(device)
            val_logits = model(Zva_t).detach().cpu()
            val_pred = val_logits.argmax(dim=1).numpy()
            val_acc = float((val_pred == np.asarray(yva)).mean())

    # return model on cpu for easier noise addition
    model_cpu = model.to("cpu")
    return model_cpu, {"train_acc": train_acc, "val_acc": val_acc}


def predict_softmax_logreg_torch(model: SoftmaxLogReg, Z: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(Z, dtype=np.float32))).cpu().numpy()
    return logits.argmax(axis=1)
