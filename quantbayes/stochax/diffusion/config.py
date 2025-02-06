# score_diffusion/config.py

from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    # Common config for both image and time-series
    lr: float = 3e-4
    batch_size: int = 64
    num_steps: int = 100000
    print_every: int = 1000
    t1: float = 10.0
    dt0: float = 0.1
    seed: int = 42


@dataclass
class ImageConfig(DiffusionConfig):
    patch_size: int = 4
    hidden_size: int = 64
    mix_patch_size: int = 512
    mix_hidden_size: int = 512
    num_blocks: int = 4


@dataclass
class TimeSeriesConfig(DiffusionConfig):
    # Example time-series config
    hidden_dim: int = 64
    time_emb_dim: int = 64
    num_layers: int = 4
    seq_length: int = 128  # e.g. time series length
