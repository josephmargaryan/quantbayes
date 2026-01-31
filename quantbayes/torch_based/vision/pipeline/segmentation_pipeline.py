"""
High-level interface wrapping:
  - model building (single or list → ensemble)
  - Trainer orchestration (single split or fine-tune)
  - Prediction with optional TTA + ensembling
  - Save / load state so you can resume days later
"""

from __future__ import annotations
import os
import json
import numpy as np
import torch
import cv2
from typing import Sequence, Dict, List, Callable, Optional, Union
from torch.utils.data import DataLoader

from ..data import GenericDataset, build_transforms
from ..trainers.segmentation_trainer import SegmentationTrainer
from ..utils import load_checkpoint, save_checkpoint


class SegmentationPipeline:
    def __init__(
        self,
        model_builders: Dict[str, Callable[[], torch.nn.Module]],
        *,
        device: Union[str, torch.device] = "cuda",
    ):
        """
        Args:
            model_builders: mapping from architecture name to a zero-arg callable
                            that returns a fresh nn.Module instance.
            device: "cuda", "cpu", or torch.device.
        """
        # choose CPU if no GPU available
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # store builders for later fine-tune or load
        self.model_builders: Dict[str, Callable[[], torch.nn.Module]] = (
            model_builders.copy()
        )

        # instantiate one model per architecture
        self.models: Dict[str, List[torch.nn.Module]] = {}
        for name, builder in self.model_builders.items():
            m = builder().to(self.device)
            m.eval()
            self.models[name] = [m]

        # uniform ensemble weights and default thresholds
        n = len(self.models)
        self.weights: Dict[str, float] = {name: 1.0 / n for name in self.models}
        self.thresholds: Dict[str, float] = {name: 0.5 for name in self.models}

    # ---------------------------------------------------------------- fit / fine-tune
    def fit(
        self,
        train_ds: GenericDataset,
        val_ds: Optional[GenericDataset] = None,
        *,
        epochs: int = 40,
        save_dir: str = "checkpoints",
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        # prepare transforms and data loaders
        train_tf, valid_tf = build_transforms("segmentation")
        train_ds.transform = train_tf  # type: ignore
        val_loader = None
        if val_ds is not None:
            val_ds.transform = valid_tf  # type: ignore
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        # train each architecture
        for name, model_list in self.models.items():
            model = model_list[-1]
            ckpt_dir = os.path.join(save_dir, name)
            trainer = SegmentationTrainer(
                model,
                save_dir=ckpt_dir,
                epochs=epochs,
                device=self.device,
            )
            trainer.fit(train_loader, val_loader, epochs)

            # reload best weights
            ckpt = load_checkpoint(os.path.join(ckpt_dir, "best.pt"))
            model.load_state_dict(ckpt["model"])
            model.eval()

    def fine_tune(
        self,
        train_ds: GenericDataset,
        val_ds: Optional[GenericDataset] = None,
        *,
        epochs: int = 40,
        save_dir: str = "checkpoints",
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        """
        Append new instances of each architecture and retrain on new data.
        """
        for name, builder in self.model_builders.items():
            new_model = builder().to(self.device)
            new_model.eval()
            self.models[name].append(new_model)

        # reuse fit logic (will train the newly appended models)
        self.fit(
            train_ds,
            val_ds,
            epochs=epochs,
            save_dir=save_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def _tta_predict(self, model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
        preds: List[np.ndarray] = []
        for flip_dims in ([], [2], [3], [2, 3]):
            x_aug = torch.flip(x, dims=flip_dims) if flip_dims else x
            p = torch.sigmoid(model(x_aug))[0, 0]
            if flip_dims:
                inv = [d - 2 for d in flip_dims]
                p = torch.flip(p, dims=inv)
            preds.append(p.cpu().numpy())
        return np.mean(preds, axis=0)

    def predict(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Run TTA + ensemble voting on a single BGR image array.
        Returns a HxWx3 uint8 mask (0 or 255) in RGB order.
        """
        H, W = img_bgr.shape[:2]

        # preprocess → 512×512 RGB [0,1]
        inp = cv2.resize(img_bgr, (512, 512))[..., ::-1].astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None]
        inp_t = torch.from_numpy(inp).to(self.device)

        vote = np.zeros((512, 512), dtype=np.float32)
        for name, models in self.models.items():
            thr = self.thresholds[name]
            w = self.weights[name]
            prob = np.mean([self._tta_predict(m, inp_t) for m in models], axis=0)
            vote += (prob > thr).astype(np.float32) * w

        mask512 = (vote > 0.5).astype(np.uint8) * 255
        mask_full = cv2.resize(mask512, (W, H), interpolation=cv2.INTER_NEAREST)
        return np.stack([mask_full] * 3, axis=-1)

    # ---------------------------------------------------------------- persistence
    def save(self, path: str):
        """
        Save ensemble metadata and all model weights.
        """
        os.makedirs(path, exist_ok=True)
        meta = {
            "archs": list(self.models.keys()),
            "weights": self.weights,
            "thresholds": self.thresholds,
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        for name, model_list in self.models.items():
            for idx, model in enumerate(model_list):
                torch.save(
                    model.state_dict(),
                    os.path.join(path, f"{name}_{idx}.pt"),
                )

    @classmethod
    def load(
        cls,
        path: str,
        model_builders: Dict[str, Callable[[], torch.nn.Module]],
        *,
        device: Union[str, torch.device] = "cuda",
    ) -> SegmentationPipeline:
        """
        Load a previously saved pipeline.
        Requires the same model_builders mapping used originally.
        """
        # read metadata
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)

        # instantiate empty object
        self = object.__new__(cls)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_builders = model_builders.copy()
        self.weights = meta["weights"]
        self.thresholds = meta["thresholds"]
        self.models = {}

        # reload each saved version
        for name in meta["archs"]:
            builder = self.model_builders[name]
            # find available checkpoint files
            files = sorted(
                fn
                for fn in os.listdir(path)
                if fn.startswith(f"{name}_") and fn.endswith(".pt")
            )
            self.models[name] = []
            for fn in files:
                idx = int(fn.split("_")[-1].split(".")[0])
                m = builder().to(self.device)
                state = torch.load(os.path.join(path, fn), map_location=self.device)
                m.load_state_dict(state)
                m.eval()
                self.models[name].append(m)

        return self

    # ---------------------------------------------------------------- editing ensemble
    def add_model(
        self,
        arch_name: str,
        model: torch.nn.Module,
        *,
        weight: float = 1.0,
        threshold: float = 0.5,
    ):
        """
        Add an externally built/trained model to the ensemble.
        """
        model = model.to(self.device)
        model.eval()
        self.models.setdefault(arch_name, []).append(model)
        self.weights[arch_name] = weight
        self.thresholds[arch_name] = threshold

    def remove_architecture(self, arch_name: str):
        """
        Remove all models of a given architecture from the ensemble.
        """
        self.models.pop(arch_name, None)
        self.weights.pop(arch_name, None)
        self.thresholds.pop(arch_name, None)
