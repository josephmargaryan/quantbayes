# quantbayes/ball_dp/reconstruction/vision/inversion_torch.py
from __future__ import annotations

from typing import Tuple, Optional
import numpy as np


def tv_loss(x: "torch.Tensor") -> "torch.Tensor":
    # x: (1,C,H,W)
    import torch

    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw


def invert_embedding_to_image(
    *,
    encoder,
    e_target: np.ndarray,
    out_shape: Tuple[int, int, int],  # (C,H,W) original resolution to optimize in
    device: str = "cuda",
    steps: int = 300,
    lr: float = 0.05,
    tv_weight: float = 1e-3,
    l2_weight: float = 1e-4,
    l2_normalize_embedding: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """
    Feature inversion: optimize an image x so that encoder(preprocess(x)) matches e_target.

    Returns:
      x_opt: (C,H,W) float32 in [0,1]
    """
    import torch
    import torch.nn.functional as F
    from torchvision.transforms.functional import resize

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    dev = torch.device(
        device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    encoder = encoder.to(dev).eval()

    e_t = torch.tensor(np.asarray(e_target, dtype=np.float32), device=dev).view(1, -1)
    if l2_normalize_embedding:
        e_t = F.normalize(e_t, p=2, dim=1)

    C, H, W = map(int, out_shape)
    x = torch.rand((1, C, H, W), device=dev, requires_grad=True)

    opt = torch.optim.Adam([x], lr=float(lr))

    # ImageNet normalization constants
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    for t in range(int(steps)):
        opt.zero_grad(set_to_none=True)

        x01 = x.clamp(0.0, 1.0)

        # encoder expects 3ch 224 + ImageNet norm
        if C == 1:
            x3 = x01.repeat(1, 3, 1, 1)
        else:
            x3 = x01

        x224 = F.interpolate(x3, size=(224, 224), mode="bilinear", align_corners=False)
        x224 = (x224 - mean) / std

        z = encoder(x224)
        if isinstance(z, (tuple, list)):
            z = z[0]
        if l2_normalize_embedding:
            z = F.normalize(z, p=2, dim=1)

        loss_embed = (z - e_t).pow(2).mean()
        loss_tv = tv_loss(x01)
        loss_l2 = (x01.pow(2)).mean()
        loss = loss_embed + float(tv_weight) * loss_tv + float(l2_weight) * loss_l2

        loss.backward()
        opt.step()

        if t % 50 == 0 or t == steps - 1:
            print(
                f"[invert] step={t:04d} loss={float(loss):.6f} embed={float(loss_embed):.6f}",
                flush=True,
            )

    x_out = x.detach().clamp(0.0, 1.0).cpu().numpy().astype(np.float32)[0]
    return x_out


if __name__ == "__main__":
    print(
        "[NOTE] inversion_torch is meant to be called from scripts with a real encoder+embedding."
    )
