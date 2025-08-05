import os
import json
import torch
import numpy as np
from typing import Sequence, Dict, List, Optional
from torch.utils.data import DataLoader

from ..registry import get_cls_model
from ..data import GenericDataset, build_transforms
from ..trainers.classification_trainer import ClassificationTrainer
from ..utils import load_checkpoint


class ClassificationPipeline:
    def __init__(
        self,
        model_names: Sequence[str],
        *,
        num_classes: int,
        pretrained: bool = True,
        device: str | torch.device = "cuda",
    ):
        # choose CPU if no GPU available
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # config for model builders
        self.model_cfg = dict(num_classes=num_classes, pretrained=pretrained)

        # build one instance per architecture
        self.models: Dict[str, List[torch.nn.Module]] = {}
        for name in model_names:
            model = get_cls_model(name)(**self.model_cfg).to(self.device)
            model.eval()
            self.models[name] = [model]

        # ensemble weights
        n = len(model_names)
        self.weights = {name: 1.0 / n for name in model_names}

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
                device=self.device,    # <–– ensure CPU-only won't try CUDA
            )
            trainer.fit(train_loader, val_loader, epochs)

            # reload best
            best_state = load_checkpoint(os.path.join(ckpt_dir, "best.pt"))["model"]
            model.load_state_dict(best_state)
            model.eval()

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        x: BxCxHxW tensor → returns Bxnum_classes numpy array of probabilities
        (softmax averaged over ensemble members).
        """
        x = x.to(self.device)
        all_probs = []
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
            "archs": list(self.models),
            "model_cfg": self.model_cfg,
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
        device: str | torch.device = "cuda",
    ) -> "ClassificationPipeline":
        """
        Load a previously saved pipeline (including all ensemble members).
        """
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)

        cfg = meta["model_cfg"]
        pipe = cls(
            meta["archs"],
            num_classes=cfg["num_classes"],
            pretrained=cfg["pretrained"],
            device=device,
        )
        pipe.weights = meta["weights"]

        # reload each saved model
        pipe.models = {}
        for name in meta["archs"]:
            versions = sorted(
                fn for fn in os.listdir(path) if fn.startswith(name + "_")
            )
            pipe.models[name] = []
            for fn in versions:
                idx = int(fn.split("_")[-1].split(".")[0])
                model = get_cls_model(name)(**pipe.model_cfg).to(pipe.device)
                state = torch.load(os.path.join(path, fn), map_location=pipe.device)
                model.load_state_dict(state)
                model.eval()
                pipe.models[name].append(model)

        return pipe

    def add_model(
        self,
        arch_name: str,
        weights_path: str,
        weight: float = 1.0,
    ):
        """
        Add an externally trained model to the ensemble.
        """
        model = get_cls_model(arch_name)(**self.model_cfg).to(self.device)
        state = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()

        self.models.setdefault(arch_name, []).append(model)
        self.weights[arch_name] = weight

    def remove_architecture(self, arch_name: str):
        """
        Remove all models of a given architecture from the ensemble.
        """
        self.models.pop(arch_name, None)
        self.weights.pop(arch_name, None)
