# scripts/save_imagenet_resnets.py
"""
Download torchvision's ResNet ImageNet weights once,
and save each as a NumPy .npz for JAX/Equinox loaders.

Usage:
    $ python scripts/save_imagenet_resnets.py
"""
from pathlib import Path
import numpy as np
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

CHECKPOINTS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}


def main():
    for name, builder in CHECKPOINTS.items():
        print(f"⇢ downloading {name} …")
        model = builder(weights="IMAGENET1K_V1")
        ckpt_path = Path(f"{name}_imagenet.npz")
        print(f"↳ saving → {ckpt_path}")
        np.savez(
            ckpt_path, **{k: v.cpu().numpy() for k, v in model.state_dict().items()}
        )
        print(f"✓ done {ckpt_path}\n")


if __name__ == "__main__":
    main()
