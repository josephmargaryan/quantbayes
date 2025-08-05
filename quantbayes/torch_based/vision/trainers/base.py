from __future__ import annotations
import torch, time, math
from torch.utils.data import DataLoader
from typing import Callable, Dict, Any, List, Optional
from ..utils import save_checkpoint


class History(dict):
    """dict(epoch → metrics) with helper .log()."""

    def log(self, epoch: int, **metrics):
        self[epoch] = metrics


class BaseTrainer:
    """
    Minimal yet extensible training loop (no Lightning dependency).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device | str = "cuda",
        *,
        clip_grad: float | None = 1.0,
        patience: int = 10,
        save_dir: str = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        self.model, self.loss_fn = model, loss_fn
        self.opt, self.sched = optimizer, scheduler
        self.device = torch.device(device)
        self.clip_grad, self.patience = clip_grad, patience
        self.save_dir = save_dir
        self.monitor, self.mode = monitor, mode
        self.history = History()
        self.best_metric = math.inf if mode == "min" else -math.inf
        self._not_improved = 0
        self.model.to(self.device)

    # --------------------------------------------------------------------- priv
    def _step(self, batch, train=True):
        x, y = (t.to(self.device, non_blocking=True) for t in batch)
        with torch.cuda.amp.autocast(enabled=train):
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
        if train:
            self.opt.zero_grad()
            loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.opt.step()
        return logits.detach(), loss.detach()

    # --------------------------------------------------------------------- api
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
    ):
        for ep in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()
            tr_loss = 0.0
            for batch in train_loader:
                _, loss = self._step(batch, train=True)
                tr_loss += float(loss)
            tr_loss /= len(train_loader)

            val_loss = float("nan")
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    val_loss = sum(float(self._step(b, False)[1]) for b in val_loader)
                val_loss /= len(val_loader)

            # Scheduler step *after* validation (cosine, reduce‑lr‑on‑plateau, etc.)
            if self.sched:
                if hasattr(self.sched, "step"):
                    self.sched.step(
                        val_loss
                        if "plateau" in self.sched.__class__.__name__.lower()
                        else None
                    )

            self.history.log(
                ep,
                train_loss=tr_loss,
                val_loss=val_loss,
                lr=self.opt.param_groups[0]["lr"],
                time=time.time() - t0,
            )

            metric = val_loss if self.monitor == "val_loss" else tr_loss
            improved = (
                (metric < self.best_metric)
                if self.mode == "min"
                else (metric > self.best_metric)
            )
            if improved:
                self.best_metric, self._not_improved = metric, 0
                save_checkpoint(
                    {"model": self.model.state_dict()}, f"{self.save_dir}/best.pt"
                )
            else:
                self._not_improved += 1
                if self._not_improved >= self.patience:
                    print(f"Early stop @ epoch {ep}")
                    break

        # always save final
        save_checkpoint({"model": self.model.state_dict()}, f"{self.save_dir}/last.pt")
        return self.history
