"""
High-level interface wrapping:
  - model building (single or list → ensemble)
  - Trainer orchestration (single split or fine-tune)
  - Prediction with ensembling
  - Save / load state so you can resume days later
"""

from __future__ import annotations
import os
import json
import torch
import numpy as np
from typing import Dict, List, Callable, Optional, Union
from torch.utils.data import DataLoader

from ..data import GenericDataset, build_transforms
from ..trainers.classification_trainer import ClassificationTrainer
from ..utils import load_checkpoint


class ClassificationPipeline:
    def __init__(
        self,
        model_builders: Dict[str, Callable[[], torch.nn.Module]],
        *,
        device: Union[str, torch.device] = "cuda",
    ):
        """
        Args:
            model_builders: mapping from architecture name to a zero-argument callable
                            that returns a new nn.Module (configured for correct num_classes).
            device: "cuda", "cpu", or torch.device.
        """
        # choose CPU if no GPU available
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # store builders for future fine-tune or load
        self.model_builders = model_builders.copy()
        # instantiate one model per architecture
        self.models: Dict[str, List[torch.nn.Module]] = {}
        for name, builder in self.model_builders.items():
            m = builder().to(self.device)
            m.eval()
            self.models[name] = [m]
        # uniform ensemble weights
        n = len(self.models)
        self.weights: Dict[str, float] = {name: 1.0 / n for name in self.models}

    def fit(
        self,
        train_ds: GenericDataset,
        val_ds: Optional[GenericDataset] = None,
        *,
        epochs: int = 30,
        save_dir: str = "checkpoints",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        # prepare transforms and loaders
        train_tf, valid_tf = build_transforms("classification")
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
        # train each architecture in turn
        for name, model_list in self.models.items():
            model = model_list[-1]
            ckpt_dir = os.path.join(save_dir, name)
            trainer = ClassificationTrainer(
                model,
                save_dir=ckpt_dir,
                epochs=epochs,
                device=self.device,
            )
            trainer.fit(train_loader, val_loader, epochs)
            # reload best checkpoint
            ckpt = load_checkpoint(os.path.join(ckpt_dir, "best.pt"))
            model.load_state_dict(ckpt["model"])
            model.eval()

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        x: BxCxHxW tensor → returns Bxnum_classes numpy array of probabilities
        (softmax averaged over ensemble members).
        """
        x = x.to(self.device)
        all_probs: List[np.ndarray] = []
        for name, models in self.models.items():
            w = self.weights[name]
            # stack predictions: [n_models, B, C]
            probs = torch.stack([torch.softmax(m(x), dim=1) for m in models], dim=0)
            mean_probs = probs.mean(dim=0)  # B x C
            all_probs.append(mean_probs.cpu().numpy() * w)
        return np.sum(all_probs, axis=0)

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Returns the argmax class index for each sample in the batch.
        """
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)

    def save(self, path: str):
        """
        Save ensemble metadata and all model weights.
        """
        os.makedirs(path, exist_ok=True)
        meta = {
            "archs": list(self.models.keys()),
            "weights": self.weights,
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
    ) -> ClassificationPipeline:
        """
        Load a previously saved pipeline.
        Requires the same model_builders mapping that was used to construct and save it.
        """
        # read metadata
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)
        # create instance without __init__
        self = object.__new__(cls)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_builders = model_builders.copy()
        self.weights = meta["weights"]
        self.models = {}
        # reload each saved model version
        for name in meta["archs"]:
            builder = self.model_builders[name]
            files = sorted(
                fn
                for fn in os.listdir(path)
                if fn.startswith(f"{name}_") and fn.endswith(".pt")
            )
            self.models[name] = []
            for fn in files:
                m = builder().to(self.device)
                state = torch.load(os.path.join(path, fn), map_location=self.device)
                m.load_state_dict(state)
                m.eval()
                self.models[name].append(m)
        return self

    def add_model(
        self,
        arch_name: str,
        model: torch.nn.Module,
        *,
        weight: float = 1.0,
    ):
        """
        Add an externally built/trained model to the ensemble.
        """
        m = model.to(self.device)
        m.eval()
        self.models.setdefault(arch_name, []).append(m)
        self.weights[arch_name] = weight

    def remove_architecture(self, arch_name: str):
        """
        Remove all models of a given architecture from the ensemble.
        """
        self.models.pop(arch_name, None)
        self.weights.pop(arch_name, None)
