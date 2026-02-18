# quantbayes/ball_dp/reconstruction/vision/encoders_torch.py
from __future__ import annotations

from typing import Any


def build_resnet18_embedder(device: str = "cuda") -> Any:
    """
    Torchvision ResNet18 with fc replaced by Identity -> outputs 512-d embedding.
    """
    try:
        import torch
        from torchvision import models
    except Exception as e:
        raise ImportError("This requires torch + torchvision installed.") from e

    dev = torch.device(
        device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )

    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    model.fc = torch.nn.Identity()
    model.eval().to(dev)
    return model


if __name__ == "__main__":
    m = build_resnet18_embedder(device="cpu")
    print(m)
    print("[OK] resnet18 embedder built.")
