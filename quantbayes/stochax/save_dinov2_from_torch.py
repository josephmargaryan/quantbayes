# save_dinov2_from_torch.py
"""
Export a PyTorch DINOv2/timm/FAIR checkpoint to .npz.

- Point to a local .pth/.pt or programmatically load from timm/facebookresearch.
- Saves raw state_dict() tensors with original keys; the Equinox loader handles renaming.
"""
from pathlib import Path
import numpy as np
import torch

# Example: local path or torch.hub load
ckpt_path = "dinov2_base_pretrain.pth"  # change to your file


def main():
    sd = torch.load(ckpt_path, map_location="cpu")
    # If the file contains a dict with "model" or "state_dict", pick it:
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]

    out = Path(Path(ckpt_path).stem + ".npz")
    np.savez(
        out, **{k: (v.cpu().numpy() if hasattr(v, "cpu") else v) for k, v in sd.items()}
    )
    print(f"✓ saved: {out}")


if __name__ == "__main__":
    main()
