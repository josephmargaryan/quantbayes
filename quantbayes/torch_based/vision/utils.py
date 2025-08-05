import random, os, numpy as np, torch


def set_seed(seed: int = 42):
    """Deterministic-ish behaviour (CuDNN still non‑det by default)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)


def load_checkpoint(path: str, map_location=None):
    return torch.load(path, map_location=map_location or "cpu")
