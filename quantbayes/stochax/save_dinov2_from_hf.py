# save_dinov2_from_hf.py
"""
Export Hugging Face DINOv2 weights to a flat .npz for Equinox loading.

Examples of repo_id:
  "facebook/dinov2-small"   # ViT-S/14
  "facebook/dinov2-base"    # ViT-B/14
  "facebook/dinov2-large"   # ViT-L/14
  "facebook/dinov2-giant"   # ViT-G/14
"""
from pathlib import Path
import numpy as np
from transformers import AutoModel

repo_id = "facebook/dinov2-base"  # change as needed


def main():
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    sd = model.state_dict()
    out = Path(repo_id.split("/")[-1] + ".npz")
    np.savez(out, **{k: v.cpu().numpy() for k, v in sd.items()})
    print(f"✓ saved: {out}")


if __name__ == "__main__":
    main()
