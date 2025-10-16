# scripts/save_imagenet_vggs.py
"""
Download torchvision's VGG ImageNet weights once,
and save each as a NumPy .npz for JAX/Equinox loaders.

Usage:
    $ python scripts/save_imagenet_vggs.py
"""
from pathlib import Path
import numpy as np
from torchvision.models import (
    vgg11,
    vgg13,
    vgg16,
    vgg19,
    vgg11_bn,
    vgg13_bn,
    vgg16_bn,
    vgg19_bn,
)

CHECKPOINTS = {
    "vgg11": vgg11,
    "vgg13": vgg13,
    "vgg16": vgg16,
    "vgg19": vgg19,
    "vgg11_bn": vgg11_bn,
    "vgg13_bn": vgg13_bn,
    "vgg16_bn": vgg16_bn,
    "vgg19_bn": vgg19_bn,
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
