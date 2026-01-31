from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class SquaredHingeTorchConfig:
    """
    Binary squared-hinge head trained on embeddings as a convex, strongly-convex ERM:

      F(w,b) = (1/n) sum_i max(0, 1 - y_i (w^T e_i + b))^2
               + (lam/2)(||w||^2 + ||b||^2)   if regularize_bias=True

    NOTES:
      - y must be in {-1, +1}
      - lam must be > 0 for strong convexity (including bias if regularize_bias=True)
      - LBFGS is the most faithful for convex ERM.
    """

    lam: float = 1e-2
    regularize_bias: bool = True

    optimizer: Literal["lbfgs", "sgd", "adam"] = "lbfgs"

    # LBFGS:
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 500

    # SGD/Adam:
    lr: float = 1e-2
    epochs: int = 50
    batch_size: int = 2048

    device: str = "cuda"
    seed: int = 0


class SquaredHingeSVM(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.linear = nn.Linear(d_in, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)  # (B,)


def _l2_penalty(model: SquaredHingeSVM, *, regularize_bias: bool) -> torch.Tensor:
    w = model.linear.weight  # (1,d)
    b = model.linear.bias  # (1,)
    if regularize_bias:
        return (w * w).sum() + (b * b).sum()
    return (w * w).sum()


def _squared_hinge_loss(scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    scores: (B,), y: (B,) with y in {-1,+1}
    loss = mean( relu(1 - y*scores)^2 )
    """
    margin = 1.0 - y * scores
    return torch.mean(torch.relu(margin) ** 2)


def train_squared_hinge_svm_torch(
    Ztr: np.ndarray,
    ytr: np.ndarray,
    Zva: Optional[np.ndarray],
    yva: Optional[np.ndarray],
    *,
    cfg: SquaredHingeTorchConfig,
) -> Tuple[SquaredHingeSVM, Dict[str, float]]:
    if float(cfg.lam) <= 0.0:
        raise ValueError("cfg.lam must be > 0 for strong convexity.")

    ytr = np.asarray(ytr, dtype=np.int64).reshape(-1)
    if not np.all(np.isin(ytr, [-1, 1])):
        raise ValueError("ytr must be in {-1, +1} for squared hinge SVM.")

    torch.manual_seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    Ztr_t = torch.from_numpy(np.asarray(Ztr, dtype=np.float32)).to(device)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(device)

    d_in = int(Ztr_t.shape[1])
    model = SquaredHingeSVM(d_in=d_in).to(device)

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
            scores = model(Ztr_t)
            loss = _squared_hinge_loss(scores, ytr_t)
            reg = (
                0.5
                * lam
                * _l2_penalty(model, regularize_bias=bool(cfg.regularize_bias))
            )
            obj = loss + reg
            obj.backward()
            return obj

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
                scores = model(xb)
                loss = _squared_hinge_loss(scores, yb)
                reg = (
                    0.5
                    * lam
                    * _l2_penalty(model, regularize_bias=bool(cfg.regularize_bias))
                )
                obj = loss + reg
                obj.backward()
                opt.step()

    # metrics
    model.eval()
    with torch.no_grad():
        scores_tr = model(Ztr_t).detach().cpu().numpy()
        pred_tr = np.where(scores_tr >= 0.0, 1, -1)
        train_acc = float(np.mean(pred_tr == ytr))

        val_acc = float("nan")
        if Zva is not None and yva is not None:
            yva = np.asarray(yva, dtype=np.int64).reshape(-1)
            if not np.all(np.isin(yva, [-1, 1])):
                raise ValueError("yva must be in {-1, +1}.")
            Zva_t = torch.from_numpy(np.asarray(Zva, dtype=np.float32)).to(device)
            scores_va = model(Zva_t).detach().cpu().numpy()
            pred_va = np.where(scores_va >= 0.0, 1, -1)
            val_acc = float(np.mean(pred_va == yva))

    model_cpu = model.to("cpu")
    return model_cpu, {"train_acc": train_acc, "val_acc": val_acc}


def predict_squared_hinge_svm_torch(
    model: SquaredHingeSVM, Z: np.ndarray
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(np.asarray(Z, dtype=np.float32))).cpu().numpy()
    return np.where(scores >= 0.0, 1, -1).astype(np.int64)


def pack_squared_hinge_params(model: SquaredHingeSVM) -> np.ndarray:
    """
    Flatten params as [w (d,), b (1,)] to feed into dp_release_erm_params_gaussian.
    """
    w = model.linear.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
    b = model.linear.bias.detach().cpu().numpy().reshape(-1).astype(np.float32)
    return np.concatenate([w, b], axis=0)
