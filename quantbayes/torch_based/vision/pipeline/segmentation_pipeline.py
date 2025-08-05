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
from typing import Sequence, Dict, List, Optional
from torch.utils.data import DataLoader
from ..registry import get_seg_model
from ..data import GenericDataset, build_transforms
from ..trainers.segmentation_trainer import SegmentationTrainer
from ..utils import load_checkpoint, save_checkpoint


class SegmentationPipeline:
    def __init__(
        self,
        model_names: Sequence[str],
        *,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_weights: Optional[str] = "imagenet",
        device: str | torch.device = "cuda",
    ):
        # choose CPU if no GPU available
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # config for model builders
        self.model_cfg = dict(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_weights=encoder_weights,
        )

        # build one instance per architecture
        self.models: Dict[str, List[torch.nn.Module]] = {}
        for name in model_names:
            model = get_seg_model(name)(
                **{
                    "in_channels": in_channels,
                    "classes": num_classes,
                    "encoder_weights": encoder_weights,
                }
            ).to(self.device)
            model.eval()
            self.models[name] = [model]

        # ensemble voting weights and thresholds
        n = len(model_names)
        self.weights = {name: 1.0 / n for name in model_names}
        self.thresholds = {name: 0.5 for name in model_names}

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

        # one trainer per architecture
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

            # load the best checkpoint back into the model
            best_state = load_checkpoint(os.path.join(ckpt_dir, "best.pt"))["model"]
            model.load_state_dict(best_state)
            model.eval()

    def fine_tune(self, *args, **kwargs):
        """
        Re-instantiate each architecture and run fit again,
        appending new model instances to the ensemble lists.
        """
        for name in list(self.models):
            new_model = get_seg_model(name)(
                **{
                    "in_channels": self.model_cfg["in_channels"],
                    "classes": self.model_cfg["num_classes"],
                    "encoder_weights": self.model_cfg["encoder_weights"],
                }
            ).to(self.device)
            new_model.eval()
            self.models[name].append(new_model)
        self.fit(*args, **kwargs)

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def _tta_predict(self, model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
        preds: List[np.ndarray] = []
        for flip_dims in ([], [2], [3], [2, 3]):
            x_aug = torch.flip(x, dims=flip_dims) if flip_dims else x
            p = torch.sigmoid(model(x_aug))[0, 0]
            if flip_dims:
                # un-flip spatial dims
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

        # preprocess to 512×512 RGB float32 in [0,1]
        inp = cv2.resize(img_bgr, (512, 512))[..., ::-1].astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None]  # 1xCxHxW
        inp_t = torch.from_numpy(inp).to(self.device)

        # accumulate weighted votes
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
            "archs": list(self.models),
            "model_cfg": self.model_cfg,
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
        cls, path: str, device: str | torch.device = "cuda"
    ) -> SegmentationPipeline:
        """
        Load a previously saved pipeline (including all ensemble members).
        """
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)

        # adapt keys: model_cfg uses num_classes, but builders expect 'classes'
        cfg = meta["model_cfg"].copy()
        in_ch = cfg["in_channels"]
        num_cls = cfg["num_classes"]
        enc_w = cfg["encoder_weights"]

        pipe = cls(
            meta["archs"],
            in_channels=in_ch,
            num_classes=num_cls,
            encoder_weights=enc_w,
            device=device,
        )
        pipe.weights = meta["weights"]
        pipe.thresholds = meta["thresholds"]

        # reload each saved model
        pipe.models = {}
        for name in meta["archs"]:
            # count how many versions exist
            versions = sorted(
                fn for fn in os.listdir(path) if fn.startswith(name + "_")
            )
            pipe.models[name] = []
            for fn in versions:
                idx = int(fn.split("_")[-1].split(".")[0])
                model = get_seg_model(name)(
                    **{
                        "in_channels": in_ch,
                        "classes": num_cls,
                        "encoder_weights": enc_w,
                    }
                ).to(pipe.device)
                state = torch.load(os.path.join(path, fn), map_location=pipe.device)
                model.load_state_dict(state)
                model.eval()
                pipe.models[name].append(model)

        return pipe

    # ---------------------------------------------------------------- editing ensemble
    def add_model(
        self,
        arch_name: str,
        weights_path: str,
        weight: float = 1.0,
        threshold: float = 0.5,
    ):
        """
        Add an externally trained model to the ensemble.
        """
        model = get_seg_model(arch_name)(
            **{
                "in_channels": self.model_cfg["in_channels"],
                "classes": self.model_cfg["num_classes"],
                "encoder_weights": self.model_cfg["encoder_weights"],
            }
        ).to(self.device)
        state = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state)
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
