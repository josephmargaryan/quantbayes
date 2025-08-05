import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

from torch.amp import GradScaler, autocast


@dataclass
class TrainerConfig:
    output_dir: str
    save_best_k: int = 3
    grad_accum_steps: int = 1
    max_grad_norm: Optional[float] = None
    amp: bool = False
    early_stop_patience: Optional[int] = None
    restore_best: bool = True
    log_interval: int = 50
    deterministic: bool = True
    seed: int = 42
    scheduler_step: str = "epoch"  # "epoch" or "batch"
    min_delta: float = 0.0


class Callback:
    def on_train_start(self, state: Dict[str, Any]): ...
    def on_epoch_start(self, state: Dict[str, Any]): ...
    def on_batch_end(self, state: Dict[str, Any]): ...
    def on_epoch_end(self, state: Dict[str, Any]): ...
    def on_train_end(self, state: Dict[str, Any]): ...


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scheduler: Optional[_LRScheduler] = None,
        device: Union[str, torch.device] = "cpu",
        config: Optional[TrainerConfig] = None,
        callbacks: Optional[List[Callback]] = None,
        metrics: Optional[Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]] = None,
    ):
        self.cfg = config or TrainerConfig(output_dir="./outputs")
        self.device = torch.device(device)
        if self.cfg.deterministic:
            torch.manual_seed(self.cfg.seed)
            torch.cuda.manual_seed_all(self.cfg.seed)

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.scaler = GradScaler(device_type="cuda") if (self.cfg.amp and self.device.type == "cuda") else None
        self.callbacks = callbacks or []
        self.metrics = metrics or {}

        self.state = {
            "epoch": 0,
            "global_step": 0,
            "best_metric": float("inf"),
            "epochs_no_improve": 0,
        }
        self.best_checkpoints: List[str] = []

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        resume_from: Optional[Union[str, os.PathLike]] = None,
    ) -> nn.Module:
        if resume_from:
            ckpt_state = self.load_checkpoint(resume_from)
            self.state.update(ckpt_state)

        for cb in self.callbacks:
            cb.on_train_start(self.state)

        for epoch in range(self.state["epoch"], num_epochs):
            self.state["epoch"] = epoch
            for cb in self.callbacks:
                cb.on_epoch_start(self.state)

            train_logs = self._run_epoch(train_loader, training=True)
            val_logs = self._run_epoch(val_loader, training=False)

            train_loss = train_logs["loss"]
            val_loss = val_logs["loss"]

            print(f"[Epoch {epoch+1}/{num_epochs}] "
                  f"Train Loss={train_loss:.4f}  Val Loss={val_loss:.4f}")

            # save checkpoint & update early stopping state
            self._maybe_save_checkpoint(val_loss)

            # scheduler step
            if isinstance(self.scheduler, ReduceLROnPlateau) and self.cfg.scheduler_step == "epoch":
                self.scheduler.step(val_loss)

            for cb in self.callbacks:
                cb.on_epoch_end(self.state)

            if (self.cfg.early_stop_patience is not None
               and self.state["epochs_no_improve"] >= self.cfg.early_stop_patience):
                print(f"Early stopping at epoch {epoch+1}")
                break

        for cb in self.callbacks:
            cb.on_train_end(self.state)

        if self.cfg.restore_best and self.best_checkpoints:
            self.load_checkpoint(self.best_checkpoints[0])
        return self.model

    def _run_epoch(self, loader: DataLoader, training: bool) -> Dict[str, float]:
        epoch_loss = 0.0
        n_batches = len(loader)
        self.model.train() if training else self.model.eval()

        for batch_idx, batch in enumerate(loader):
            x, y = [b.to(self.device) for b in batch]
            with torch.set_grad_enabled(training):
                if self.scaler:
                    with autocast(device_type=self.device.type):
                        preds = self.model(x)
                        loss = self.loss_fn(preds, y)
                    self.scaler.scale(loss).backward()
                else:
                    preds = self.model(x)
                    loss = self.loss_fn(preds, y)
                    if training:
                        loss.backward()

                if training and (batch_idx + 1) % self.cfg.grad_accum_steps == 0:
                    if self.cfg.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.state["global_step"] += 1

                if training and self.scheduler and self.cfg.scheduler_step == "batch":
                    # For schedulers like OneCycleLR, etc.
                    self.scheduler.step()

            epoch_loss += loss.item()
            for cb in self.callbacks:
                cb.on_batch_end(self.state)

        avg_loss = epoch_loss / n_batches
        return {"loss": avg_loss}

    def save_checkpoint(self, path: Union[str, os.PathLike], **extra):
        tmp = f"{path}.tmp"
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            **extra
        }
        torch.save(state, tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path: Union[str, os.PathLike]) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        return {k: v for k, v in ckpt.items() if k not in ("model", "optimizer", "scheduler")}

    def _maybe_save_checkpoint(self, current_metric: float):
        improved = current_metric < (self.state["best_metric"] - self.cfg.min_delta)
        if improved:
            self.state["best_metric"] = current_metric
            self.state["epochs_no_improve"] = 0
        else:
            self.state["epochs_no_improve"] += 1

        if improved:
            fname = os.path.join(
                self.cfg.output_dir,
                f"best_epoch{self.state['epoch']+1:03d}_metric{current_metric:.4f}.pt"
            )
            self.save_checkpoint(fname, epoch=self.state["epoch"], best_metric=current_metric)
            self.best_checkpoints.insert(0, fname)

        # prune old checkpoints
        for old in self.best_checkpoints[self.cfg.save_best_k:]:
            if os.path.exists(old):
                os.remove(old)
        self.best_checkpoints = self.best_checkpoints[: self.cfg.save_best_k]

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torch.optim import SGD
    from pathlib import Path

    # A small subclass that forces validation losses
    class DummyTrainer(Trainer):
        def __init__(self, *args, val_losses: list, **kwargs):
            super().__init__(*args, **kwargs)
            self._forced_val_losses = val_losses
            self._val_call = 0

        def _run_epoch(self, loader, training: bool):
            if training:
                return {"loss": 1.0}
            loss = self._forced_val_losses[self._val_call]
            self._val_call += 1
            return {"loss": loss}

    # Test parameters
    val_losses = [0.5, 0.4, 0.45, 0.47, 0.50]  # improves at epoch 2, then plateaus
    patience = 2

    # Build model, optimizer, loss
    model = nn.Linear(10, 1)
    optimizer = SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    # Configure trainer for early stopping
    cfg = TrainerConfig(
        output_dir=str(Path("/tmp/test_outputs")),
        early_stop_patience=patience,
        restore_best=True,
        save_best_k=1,
    )

    # Instantiate and run
    trainer = DummyTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=None,
        device="cpu",
        config=cfg,
        val_losses=val_losses
    )
    trained_model = trainer.train(train_loader=None, val_loader=None, num_epochs=10)

    # Report results
    stopped_epoch = trainer.state["epoch"] + 1
    print(f"Training stopped at epoch {stopped_epoch}.")
    print("Best checkpoint saved at:", trainer.best_checkpoints[0])
