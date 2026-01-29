# quantbayes/ball_dp/utils/torch_utils.py
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


@torch.no_grad()
def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def get_device(device: str) -> torch.device:
    if device.lower().startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")
