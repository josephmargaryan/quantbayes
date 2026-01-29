# quantbayes/ball_dp/utils/seeding.py
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int, *, deterministic_torch: bool = False) -> None:
    """
    Set seeds for Python, NumPy, (optionally) torch if installed.

    deterministic_torch=True can slow things down but improves reproducibility.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
