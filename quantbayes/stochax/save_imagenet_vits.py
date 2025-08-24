# scripts/save_imagenet_vits.py
from pathlib import Path
import numpy as np
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14

CHECKPOINTS = {
    "vit_b_16": (vit_b_16, "IMAGENET1K_V1"),
    "vit_b_32": (vit_b_32, "IMAGENET1K_V1"),
    "vit_l_16": (vit_l_16, "IMAGENET1K_V1"),
    "vit_l_32": (vit_l_32, "IMAGENET1K_V1"),
    "vit_h_14": (vit_h_14, "IMAGENET1K_SWAG_E2E_V1"),
}


def main():
    for name, (builder, weights_name) in CHECKPOINTS.items():
        print(f"⇢ downloading {name} …")
        model = builder(weights=weights_name)
        ckpt_path = Path(f"{name}_imagenet.npz")
        print(f"↳ saving → {ckpt_path}")
        np.savez(
            ckpt_path, **{k: v.cpu().numpy() for k, v in model.state_dict().items()}
        )
        print(f"✓ done {ckpt_path}\n")


if __name__ == "__main__":
    main()
