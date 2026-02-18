# quantbayes/ball_dp/reconstruction/scripts/demo_cifar10_reconstruction.py
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from quantbayes.ball_dp.reconstruction import PoolBallPrior


class SmallCIFAREncoder(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_cifar10(n_train: int, seed: int = 0):
    tfm = T.Compose([T.ToTensor()])
    train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=tfm
    )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train), size=n_train, replace=False)

    X, y = [], []
    for i in idx:
        xi, yi = train[int(i)]
        X.append(xi.numpy())
        y.append(int(yi))
    return np.stack(X, axis=0), np.array(y, dtype=np.int64)


def flatten_pixels(X: np.ndarray) -> np.ndarray:
    return X.reshape((X.shape[0], -1)).astype(np.float64)


def compute_embeddings(encoder: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    encoder.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X).to(device=device, dtype=torch.float32)
        e = encoder(xb).cpu().numpy().astype(np.float64)
    return e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pixel", "embedding"], default="embedding")
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-index", type=int, default=0)
    p.add_argument("--radius", type=float, default=10.0)
    p.add_argument("--m-candidates", type=int, default=50)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    Xtr, ytr = load_cifar10(args.n_train, seed=args.seed)

    if args.mode == "pixel":
        Feat = flatten_pixels(Xtr)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = SmallCIFAREncoder(embed_dim=128).to(device)
        Feat = compute_embeddings(encoder, Xtr, device=device)

    j = int(args.target_index) % Feat.shape[0]
    x_target = Feat[j]
    y_target = int(ytr[j])

    prior = PoolBallPrior(
        pool_X=Feat, pool_y=ytr, radius=float(args.radius), label_fixed=y_target
    )
    candidates = prior.sample(center=x_target, m=int(args.m_candidates), rng=rng)

    print(
        f"[CIFAR10] mode={args.mode}  n={Feat.shape[0]}  target_y={y_target}  |C|={len(candidates)}"
    )
    print(
        "This script focuses on candidate-set construction; plug in the convex/nonconvex attacks exactly like MNIST demo."
    )


if __name__ == "__main__":
    main()
