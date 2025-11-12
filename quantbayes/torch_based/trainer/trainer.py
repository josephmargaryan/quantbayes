import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

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
    amp: bool = False  # Enable mixed precision (CUDA only)
    amp_dtype: str = "bf16"  # "bf16" or "fp16" (CUDA only)
    early_stop_patience: Optional[int] = None
    restore_best: bool = True
    log_interval: int = 50
    deterministic: bool = True
    seed: int = 42
    scheduler_step: str = "epoch"  # "epoch" or "batch"
    min_delta: float = 0.0  # improvement threshold
    monitor: str = "val_loss"  # metric to monitor for early stopping/checkpointing
    mode: str = "min"  # "min" or "max"
    zero_grad_set_to_none: bool = True
    nonfinite_skip: bool = True  # skip batches with NaN/Inf loss
    compile: bool = False  # optional torch.compile() for the model


class Callback:
    def on_train_start(self, state: Dict[str, Any]): ...
    def on_epoch_start(self, state: Dict[str, Any]): ...
    def on_batch_end(self, state: Dict[str, Any]): ...
    def on_epoch_end(self, state: Dict[str, Any]): ...
    def on_train_end(self, state: Dict[str, Any]): ...


class Trainer:
    """
    Production-grade PyTorch trainer with:
      - AMP (CUDA) with correct grad unscale + clipping
      - Gradient accumulation with proper scaling and end-of-epoch flush
      - Deterministic seeding (warn_only to avoid hard errors)
      - Early stopping + top-k best checkpoints with monitor/mode
      - Robust scheduler stepping (batch or epoch, including ReduceLROnPlateau)
      - Non-finite loss handling
      - Optional torch.compile
      - Callback hooks and structured logging
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scheduler: Optional[_LRScheduler] = None,
        device: Union[str, torch.device] = "cpu",
        config: Optional[TrainerConfig] = None,
        callbacks: Optional[List[Callback]] = None,
        metrics: Optional[
            Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]
        ] = None,
    ):
        import logging

        self.cfg = config or TrainerConfig(output_dir="./outputs")
        self.device = torch.device(device)

        # Logger (stream only; integrate with your infra as needed)
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        if self.cfg.deterministic:
            self._set_deterministic(self.cfg.seed)

        os.makedirs(self.cfg.output_dir, exist_ok=True)

        # Model + optional compilation
        self.model = model.to(self.device)
        if self.cfg.compile:
            try:
                self.model = torch.compile(self.model)  # type: ignore[attr-defined]
                self.logger.info("Model compiled with torch.compile().")
            except Exception as e:
                self.logger.warning(
                    "torch.compile() failed (%s); proceeding without.", str(e)
                )

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn

        # AMP setup (only meaningful on CUDA)
        self.autocast_enabled = bool(self.cfg.amp and self.device.type == "cuda")
        self.scaler = GradScaler(device_type="cuda") if self.autocast_enabled else None
        self.amp_dtype = (
            torch.bfloat16 if self.cfg.amp_dtype.lower() == "bf16" else torch.float16
        )

        self.callbacks = callbacks or []
        self.metrics = metrics or {}

        best_init = -float("inf") if self.cfg.mode.lower() == "max" else float("inf")
        self.state: Dict[str, Any] = {
            "epoch": 0,
            "global_step": 0,
            "best_metric": best_init,
            "epochs_no_improve": 0,
            "monitor": self.cfg.monitor,
            "mode": self.cfg.mode.lower(),
            "last_train_logs": {},
            "last_val_logs": {},
        }
        self.best_checkpoints: List[str] = []

    # ---------------------------- Public API ---------------------------- #

    def train(
        self,
        train_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader],
        num_epochs: int,
        resume_from: Optional[Union[str, os.PathLike]] = None,
    ) -> nn.Module:
        if resume_from:
            ckpt_state = self.load_checkpoint(resume_from)
            self.state.update(ckpt_state)

        for cb in self.callbacks:
            cb.on_train_start(self.state)

        start_epoch = int(self.state.get("epoch", 0))
        for epoch in range(start_epoch, num_epochs):
            self.state["epoch"] = epoch
            for cb in self.callbacks:
                cb.on_epoch_start(self.state)

            train_logs = self._run_epoch(train_loader, training=True)
            self.state["last_train_logs"] = train_logs

            val_logs = (
                self._run_epoch(val_loader, training=False)
                if val_loader is not None
                else {}
            )
            self.state["last_val_logs"] = val_logs

            # Choose monitored value
            current_metric = self._resolve_monitor_value(train_logs, val_logs)

            # Logging
            lr = self.optimizer.param_groups[0].get("lr", None)
            msg = f"[Epoch {epoch+1}/{num_epochs}]"
            msg += f" train_loss={train_logs.get('loss', float('nan')):.4f}"
            if val_loader is not None:
                msg += f"  val_loss={val_logs.get('loss', float('nan')):.4f}"
            if lr is not None:
                msg += f"  lr={lr:.6f}"
            self.logger.info(msg)

            # Checkpoint + early stopping updates
            self._maybe_save_checkpoint(current_metric)

            # Scheduler stepping
            self._scheduler_step_after_epoch(current_metric)

            for cb in self.callbacks:
                cb.on_epoch_end(self.state)

            # Early stopping
            if (
                self.cfg.early_stop_patience is not None
                and self.state["epochs_no_improve"] >= self.cfg.early_stop_patience
            ):
                self.logger.info("Early stopping at epoch %d", epoch + 1)
                break

        for cb in self.callbacks:
            cb.on_train_end(self.state)

        if self.cfg.restore_best and self.best_checkpoints:
            self.logger.info("Restoring best checkpoint: %s", self.best_checkpoints[0])
            self.load_checkpoint(self.best_checkpoints[0])

        return self.model

    def save_checkpoint(self, path: Union[str, os.PathLike], **extra):
        tmp = f"{path}.tmp"
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "trainer_state": self.state,
            **extra,
        }
        torch.save(state, tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path: Union[str, os.PathLike]) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt and ckpt["optimizer"]:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                self.logger.warning("Could not load optimizer state: %s", str(e))
        if self.scheduler and ckpt.get("scheduler"):
            try:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                self.logger.warning("Could not load scheduler state: %s", str(e))
        ts = ckpt.get("trainer_state", {})
        return ts if isinstance(ts, dict) else {}

    # -------------------------- Internal Helpers ------------------------ #

    def _set_deterministic(self, seed: int):
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Help deterministic cuBLAS GEMM; harmless if set already.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # Avoid hard failures on non-deterministic ops but warn.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)  # type: ignore[attr-defined]
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _resolve_monitor_value(
        self, train_logs: Dict[str, float], val_logs: Dict[str, float]
    ) -> float:
        name = self.cfg.monitor
        mode = self.cfg.mode.lower()

        # Prefer val metric if requested and available; otherwise fall back to train
        if name in val_logs:
            val = val_logs[name]
        elif name in train_logs:
            val = train_logs[name]
        else:
            # Fallback to primary losses
            val = val_logs.get("loss", train_logs.get("loss", float("inf")))
        # For 'max' mode, we will invert comparison logic later; here we just return the value.
        return float(val)

    def _improved(self, current: float, best: float) -> bool:
        delta = self.cfg.min_delta
        if self.state["mode"] == "max":
            return current > (best + delta)
        return current < (best - delta)

    def _autocast_cm(self):
        # Enable autocast only when configured and on CUDA
        return autocast(
            device_type=(
                self.device.type if self.device.type in ("cuda", "cpu") else "cuda"
            ),
            dtype=self.amp_dtype,
            enabled=self.autocast_enabled,
        )

    def _to_device(self, x):
        if torch.is_tensor(x):
            return x.to(self.device, non_blocking=True)
        if isinstance(x, (list, tuple)):
            return type(x)(self._to_device(xx) for xx in x)
        if isinstance(x, dict):
            return {k: self._to_device(v) for k, v in x.items()}
        return x

    def _extract_xy(self, batch) -> Tuple[Any, Any]:
        # Common patterns: (x, y), {"inputs":..., "targets":...}, {"x":..., "y":...}, etc.
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        if isinstance(batch, dict):
            x = None
            y = None
            for k in ("inputs", "input", "x", "features"):
                if k in batch:
                    x = batch[k]
                    break
            for k in ("targets", "target", "y", "labels"):
                if k in batch:
                    y = batch[k]
                    break
            if x is not None and y is not None:
                return x, y
        raise ValueError(
            "Batch format not understood. Expected (x, y) or dict with inputs/targets."
        )

    def _compute_metrics(
        self, preds: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, fn in self.metrics.items():
            try:
                val = fn(preds, y)
                out[name] = (
                    float(val)
                    if isinstance(val, (int, float))
                    else float(val.detach().cpu().item())
                )
            except Exception as e:
                self.logger.warning("Metric '%s' failed: %s", name, str(e))
        return out

    def _scheduler_step_after_epoch(self, metric_value: float):
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            # Plateau schedulers expect the monitored value
            self.scheduler.step(metric_value)
        elif self.cfg.scheduler_step == "epoch":
            # Generic epoch-level schedulers
            self.scheduler.step()

    def _maybe_save_checkpoint(self, current_metric: float):
        best_before = self.state["best_metric"]
        improved = self._improved(current_metric, best_before)

        if improved:
            self.state["best_metric"] = current_metric
            self.state["epochs_no_improve"] = 0

            fname = os.path.join(
                self.cfg.output_dir,
                f"best_epoch{self.state['epoch']+1:03d}_{self.state['monitor']}{current_metric:.6f}.pt",
            )
            self.save_checkpoint(
                fname,
                epoch=self.state["epoch"],
                best_metric=current_metric,
                monitor=self.state["monitor"],
                mode=self.state["mode"],
                global_step=self.state["global_step"],
            )
            self.best_checkpoints.insert(0, fname)
        else:
            self.state["epochs_no_improve"] += 1

        # Prune old checkpoints
        for old in self.best_checkpoints[self.cfg.save_best_k :]:
            if os.path.exists(old):
                try:
                    os.remove(old)
                except Exception as e:
                    self.logger.warning(
                        "Failed removing old checkpoint %s: %s", old, str(e)
                    )
        self.best_checkpoints = self.best_checkpoints[: self.cfg.save_best_k]

    # --------------------------- Epoch Runner --------------------------- #

    def _run_epoch(
        self, loader: Optional[DataLoader], training: bool
    ) -> Dict[str, float]:
        if loader is None:
            # Allow subclasses (e.g., tests) to override and return logs without a loader.
            return {}

        if training:
            self.model.train()
        else:
            self.model.eval()

        epoch_loss_sum = 0.0
        seen_batches = 0

        # Simple metric averaging across batches
        metric_sums: Dict[str, float] = {}

        # Important: zero grads once before first backward of the epoch
        if training:
            self.optimizer.zero_grad(set_to_none=self.cfg.zero_grad_set_to_none)

        for batch_idx, batch in enumerate(loader):
            try:
                x, y = self._extract_xy(batch)
                x = self._to_device(x)
                y = self._to_device(y)

                # Forward (with autocast if enabled)
                with torch.set_grad_enabled(training), self._autocast_cm():
                    preds = self.model(x)
                    loss = self.loss_fn(preds, y)

                loss_value = float(loss.detach().cpu().item())

                # Non-finite loss handling
                if not (
                    loss_value == loss_value
                    and (loss_value != float("inf") and loss_value != -float("inf"))
                ):  # NaN/Inf check
                    if self.cfg.nonfinite_skip:
                        self.logger.warning(
                            "Non-finite loss at batch %d; skipping.", batch_idx
                        )
                        if training:
                            self.optimizer.zero_grad(
                                set_to_none=self.cfg.zero_grad_set_to_none
                            )
                        continue
                    else:
                        raise FloatingPointError("Non-finite loss encountered.")

                # Backward / step
                if training:
                    # Scale loss for accumulation so gradient magnitude matches non-accum case
                    loss_for_backward = loss / max(1, self.cfg.grad_accum_steps)

                    if self.scaler:
                        self.scaler.scale(loss_for_backward).backward()
                    else:
                        loss_for_backward.backward()

                    step_boundary = (batch_idx + 1) % self.cfg.grad_accum_steps == 0

                    if step_boundary:
                        # Unscale before clipping when using AMP
                        if self.cfg.max_grad_norm is not None:
                            if self.scaler:
                                self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.cfg.max_grad_norm
                            )

                        if self.scaler:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                        self.optimizer.zero_grad(
                            set_to_none=self.cfg.zero_grad_set_to_none
                        )
                        self.state["global_step"] += 1

                        if (
                            self.scheduler
                            and self.cfg.scheduler_step == "batch"
                            and not isinstance(self.scheduler, ReduceLROnPlateau)
                        ):
                            self.scheduler.step()

                # Metrics accumulation (no grad dependency)
                epoch_loss_sum += loss_value
                seen_batches += 1

                # Compute metrics in no-grad region
                with torch.no_grad():
                    batch_metrics = (
                        self._compute_metrics(preds, y) if self.metrics else {}
                    )
                for k, v in batch_metrics.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v)

                # Callback hook
                for cb in self.callbacks:
                    cb.on_batch_end(self.state)

                # Periodic logging
                if (
                    training
                    and self.cfg.log_interval
                    and (batch_idx + 1) % self.cfg.log_interval == 0
                ):
                    self.logger.info(
                        "Epoch %d | batch %d | loss=%.4f",
                        self.state["epoch"] + 1,
                        batch_idx + 1,
                        loss_value,
                    )

            except Exception as e:
                self.logger.exception("Exception in batch %d: %s", batch_idx, str(e))
                if training:
                    self.optimizer.zero_grad(set_to_none=self.cfg.zero_grad_set_to_none)
                # In production you may choose to re-raise; here we continue to keep training resilient.
                continue

        # End-of-epoch gradient flush for leftover microbatches
        if (
            training
            and seen_batches > 0
            and (seen_batches % self.cfg.grad_accum_steps != 0)
        ):
            if self.cfg.max_grad_norm is not None:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )

            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=self.cfg.zero_grad_set_to_none)
            self.state["global_step"] += 1

            if (
                self.scheduler
                and self.cfg.scheduler_step == "batch"
                and not isinstance(self.scheduler, ReduceLROnPlateau)
            ):
                self.scheduler.step()

        avg_loss = (
            (epoch_loss_sum / max(1, seen_batches))
            if seen_batches > 0
            else float("inf")
        )

        logs: Dict[str, float] = {"loss": float(avg_loss)}
        # Average metrics across batches
        for k, v_sum in metric_sums.items():
            logs[k] = float(v_sum / max(1, seen_batches))

        # Prefix val_ for validation metrics to avoid name collisions if desired
        if not training:
            # mirror "loss" into "val_loss" for convenience/monitoring
            logs = {
                (f"val_{k}" if not k.startswith("val_") else k): v
                for k, v in logs.items()
            }

        return logs


# ------------------------------- Example -------------------------------- #
if __name__ == "__main__":
    # Minimal smoke test demonstrating early stopping + best-k saving behavior.
    # Replace with your real loaders/model/metrics.
    import torch
    import torch.nn as nn
    from torch.optim import SGD
    from pathlib import Path

    class DummyTrainer(Trainer):
        def __init__(self, *args, val_losses: List[float], **kwargs):
            super().__init__(*args, **kwargs)
            self._forced_val_losses = val_losses
            self._val_call = 0

        def _run_epoch(self, loader, training: bool) -> Dict[str, float]:
            # Training returns constant loss; validation consumes scripted losses
            if training:
                return {"loss": 1.0}
            loss = self._forced_val_losses[self._val_call]
            self._val_call += 1
            return {"val_loss": loss, "loss": loss}  # include both keys

    val_losses = [0.5, 0.4, 0.45, 0.47, 0.50]  # improves at epoch 2, then plateaus
    patience = 2

    model = nn.Linear(10, 1)
    optimizer = SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    cfg = TrainerConfig(
        output_dir=str(Path("/tmp/test_outputs")),
        early_stop_patience=patience,
        restore_best=True,
        save_best_k=1,
        monitor="val_loss",
        mode="min",
        deterministic=True,
        amp=False,  # set True + CUDA for AMP test
    )

    trainer = DummyTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=None,
        device="cpu",
        config=cfg,
        val_losses=val_losses,
    )

    trained_model = trainer.train(train_loader=None, val_loader=None, num_epochs=10)
    stopped_epoch = trainer.state["epoch"] + 1
    print(f"Training stopped at epoch {stopped_epoch}.")
    print(
        "Best checkpoint saved at:",
        trainer.best_checkpoints[0] if trainer.best_checkpoints else "None",
    )
