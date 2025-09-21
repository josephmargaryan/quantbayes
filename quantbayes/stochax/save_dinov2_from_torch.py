"""
Download DINOv2 backbones from PyTorch Hub and save as NumPy .npz for Equinox loaders.

Usage examples:
  $ python quantbayes/stochax/save_dinov2_from_torch.py --arch vits14 --registers
  $ python quantbayes/stochax/save_dinov2_from_torch.py --arch vitb14
  $ python quantbayes/stochax/save_dinov2_from_torch.py --arch vitl14 --registers

Files will be named like: dinov2_vitb14_reg.npz
"""

from pathlib import Path
import argparse
import numpy as np
import torch


ALIASES = {
    "vits14": ("facebookresearch/dinov2", "dinov2_vits14"),
    "vitb14": ("facebookresearch/dinov2", "dinov2_vitb14"),
    "vitl14": ("facebookresearch/dinov2", "dinov2_vitl14"),
    "vitg14": ("facebookresearch/dinov2", "dinov2_vitg14"),
}
ALIASES_REG = {
    "vits14": ("facebookresearch/dinov2", "dinov2_vits14_reg"),
    "vitb14": ("facebookresearch/dinov2", "dinov2_vitb14_reg"),
    "vitl14": ("facebookresearch/dinov2", "dinov2_vitl14_reg"),
    "vitg14": ("facebookresearch/dinov2", "dinov2_vitg14_reg"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(ALIASES.keys()))
    ap.add_argument(
        "--registers", action="store_true", help="Use *with registers* variant"
    )
    ap.add_argument("--out", type=Path, default=Path("."))
    args = ap.parse_args()

    repo, name = (ALIASES_REG if args.registers else ALIASES)[args.arch]
    print(f"⇢ loading torch.hub {repo}:{name}")
    model = torch.hub.load(repo, name, trust_repo=True)
    sd = model.state_dict()

    suffix = "reg" if args.registers else "noreg"
    fname = args.out / f"dinov2_{args.arch}_{suffix}.npz"
    print(f"↳ saving → {fname}")
    np.savez(fname, **{k: v.cpu().numpy() for k, v in sd.items()})
    print("✓ done")


if __name__ == "__main__":
    main()
