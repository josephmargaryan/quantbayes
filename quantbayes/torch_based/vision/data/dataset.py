import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Callable


class GenericDataset(Dataset):
    """
    A minimal, flexible dataset supporting:
      1)  X, y segmentation  (mask images)
      2)  X, y classification (label idx in .txt)
      3)  X only             (inference / pseudo-labelling)

    Returns (image_tensor, target) where:
      - image_tensor: torch.FloatTensor, shape [C,H,W], values in [0,1]
      - target (seg): torch.FloatTensor, shape [1,H,W] or [C,H,W]
      - target (cls): torch.LongTensor scalar
      - target (inf): torch.FloatTensor empty tensor
    """

    def __init__(
        self,
        img_dir: str,
        target_dir: Optional[str] = None,
        *,
        transform: Optional[Callable] = None,
        task: str = "segmentation",
        size: Tuple[int, int] = (512, 512),
        classes: int = 1,
        num_channels: int = 3,
    ):
        self.img_paths = sorted(
            os.path.join(img_dir, fn)
            for fn in os.listdir(img_dir)
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp"))
        )
        self.target_dir = target_dir
        self.transform = transform
        self.task = task
        self.size = size
        self.classes = classes
        self.num_channels = num_channels

    def __len__(self):
        return len(self.img_paths)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image {path}")
        if img.ndim == 2:
            img = np.stack([img] * self.num_channels, axis=-1)
        img = cv2.resize(img, self.size).astype(np.float32) / 255.0
        if img.shape[2] != self.num_channels:
            img = img[:, :, : self.num_channels]
        return img.transpose(2, 0, 1)  # CHW

    def _load_mask(self, img_path: str) -> np.ndarray:
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.target_dir, f"{base}.png")
        if not os.path.exists(mask_path):
            return np.zeros((*self.size, self.classes), dtype=np.float32)
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, self.size, interpolation=cv2.INTER_NEAREST)
        m = (m > 127).astype(np.float32)
        return m[..., None] if self.classes == 1 else m

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        img_np = self._load_image(img_path)  # CHW float32

        # ── SEGMENTATION ──────────────────────────────────────────────
        if self.task == "segmentation":
            mask_np = self._load_mask(img_path) if self.target_dir else None

            if self.transform:
                # Albumentations expects HWC
                inp = {"image": img_np.transpose(1, 2, 0)}
                if mask_np is not None:
                    inp["mask"] = mask_np
                out = self.transform(**inp)

                # image Tensor CHW
                x = out["image"]

                # unify mask to numpy array
                m = out.get("mask", None)
                if m is None:
                    return x, torch.zeros(0, dtype=torch.float32)

                # to numpy
                if isinstance(m, torch.Tensor):
                    m_arr = m.cpu().numpy()
                else:
                    m_arr = m

                # m_arr shape: HxW or HxWxC or CxHxW
                if m_arr.ndim == 2:
                    m_arr = m_arr[..., None]
                elif m_arr.ndim == 3:
                    # detect channel-first vs last
                    c0, h0, w0 = m_arr.shape[0], m_arr.shape[1], m_arr.shape[2]
                    if c0 == self.classes:
                        # already CxHxW
                        m_arr = m_arr
                    else:
                        # HxWxC → CxHxW
                        m_arr = m_arr.transpose(2, 0, 1)
                else:
                    raise ValueError(f"Invalid mask ndim {m_arr.ndim}")

                # ensure CxHxW
                if m_arr.shape[0] != self.classes:
                    raise ValueError(
                        f"Expected {self.classes} channels, got {m_arr.shape[0]}"
                    )

                y = torch.from_numpy(m_arr.astype(np.float32))
                return x, y

            else:
                # no-augmentation path
                x = torch.from_numpy(img_np)  # CHW
                if mask_np is None:
                    return x, torch.zeros(0, dtype=torch.float32)

                m_arr = mask_np  # HxW or HxWxC
                if m_arr.ndim == 2:
                    m_arr = m_arr[None]
                else:
                    m_arr = m_arr.transpose(2, 0, 1)
                y = torch.from_numpy(m_arr.astype(np.float32))
                return x, y

        # ── CLASSIFICATION ────────────────────────────────────────────
        else:
            if not self.target_dir:
                y = torch.tensor(-1, dtype=torch.long)
            else:
                label_path = os.path.join(self.target_dir, f"{base}.txt")
                if not os.path.exists(label_path):
                    raise FileNotFoundError(f"Label file not found: {label_path}")
                lbl = int(open(label_path, "r").read().strip())
                y = torch.tensor(lbl, dtype=torch.long)

            if self.transform:
                out = self.transform(image=img_np.transpose(1, 2, 0))
                x = out["image"]
            else:
                x = torch.from_numpy(img_np)

            return x, y
