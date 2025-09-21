from pathlib import Path
import argparse, numpy as np
from torchvision.models import swin_t, swin_s, swin_b

CHECKPOINTS = {
    "swin_t": swin_t,
    "swin_s": swin_s,
    "swin_b": swin_b,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(CHECKPOINTS.keys()))
    ap.add_argument("--out", type=Path, default=Path("."))
    args = ap.parse_args()

    print(f"⇢ downloading {args.arch} …")
    m = CHECKPOINTS[args.arch](weights="IMAGENET1K_V1")
    sd = {k.replace("module.", ""): v.cpu().numpy() for k, v in m.state_dict().items()}

    fp = args.out / f"{args.arch}_imagenet.npz"
    print(f"↳ saving → {fp}")
    np.savez(fp, **sd)
    print("✓ done")


if __name__ == "__main__":
    main()
