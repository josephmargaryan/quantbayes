# quantbayes/ball_dp/reconstruction/vision/mnist_autoencoder_torch.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud


class MNISTConvAutoencoder(nn.Module):
    """
    Simple conv AE:
      x (1x28x28) -> embed_dim -> xhat (1x28x28)
    Public encoder+decoder for embedding-space experiments.
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = int(embed_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, self.embed_dim),
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(self.embed_dim, 64 * 7 * 7),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 28x28
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, e: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(e)
        h = h.view(h.shape[0], 64, 7, 7)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


@torch.no_grad()
def encode_numpy(
    model: MNISTConvAutoencoder, X: np.ndarray, *, device: str, batch_size: int = 512
) -> np.ndarray:
    model.eval()
    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    dl = tud.DataLoader(tud.TensorDataset(X_t), batch_size=batch_size, shuffle=False)
    outs = []
    for (xb,) in dl:
        xb = xb.to(device=device)
        e = model.encode(xb).detach().cpu().numpy().astype(np.float64)
        outs.append(e)
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def decode_numpy(
    model: MNISTConvAutoencoder, E: np.ndarray, *, device: str, batch_size: int = 512
) -> np.ndarray:
    model.eval()
    E_t = torch.from_numpy(np.asarray(E, dtype=np.float32))
    dl = tud.DataLoader(tud.TensorDataset(E_t), batch_size=batch_size, shuffle=False)
    outs = []
    for (eb,) in dl:
        eb = eb.to(device=device)
        xh = model.decode(eb).detach().cpu().numpy().astype(np.float64)
        outs.append(xh)
    return np.concatenate(outs, axis=0)  # (N,1,28,28)


def train_autoencoder(
    *,
    model: MNISTConvAutoencoder,
    X_train: np.ndarray,
    X_val: np.ndarray,
    device: str,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
) -> Dict[str, list]:
    _set_torch_seed(seed)
    model.to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
    Xv = torch.from_numpy(np.asarray(X_val, dtype=np.float32))
    dl = tud.DataLoader(
        tud.TensorDataset(Xt), batch_size=batch_size, shuffle=True, drop_last=False
    )
    dlv = tud.DataLoader(
        tud.TensorDataset(Xv), batch_size=batch_size, shuffle=False, drop_last=False
    )

    hist = {"train": [], "val": []}

    for ep in range(1, int(epochs) + 1):
        # train
        s = 0.0
        nb = 0
        model.train()
        for (xb,) in dl:
            xb = xb.to(device=device)
            opt.zero_grad(set_to_none=True)
            xh = model(xb)
            loss = loss_fn(xh, xb)
            loss.backward()
            opt.step()
            s += float(loss.item())
            nb += 1
        tr = s / max(1, nb)

        # val
        s = 0.0
        nb = 0
        model.eval()
        with torch.no_grad():
            for (xb,) in dlv:
                xb = xb.to(device=device)
                xh = model(xb)
                loss = loss_fn(xh, xb)
                s += float(loss.item())
                nb += 1
        va = s / max(1, nb)

        hist["train"].append(tr)
        hist["val"].append(va)
        print(f"[AE] epoch {ep:03d}/{epochs} | train_mse={tr:.6f} | val_mse={va:.6f}")

    return hist


def save_autoencoder(path: str | Path, model: MNISTConvAutoencoder) -> None:
    path = Path(path)
    payload = {
        "embed_dim": int(model.embed_dim),
        "state_dict": model.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_autoencoder(path: str | Path, *, device: str) -> MNISTConvAutoencoder:
    path = Path(path)
    payload = torch.load(str(path), map_location=device)
    model = MNISTConvAutoencoder(embed_dim=int(payload["embed_dim"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def load_or_train_mnist_autoencoder(
    *,
    ckpt_path: str | Path,
    X_train: np.ndarray,
    X_val: np.ndarray,
    device: str,
    embed_dim: int = 64,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
) -> Tuple[MNISTConvAutoencoder, Dict[str, list]]:
    ckpt_path = Path(ckpt_path)
    if ckpt_path.exists():
        print(f"[AE] loading from {ckpt_path}")
        model = load_autoencoder(ckpt_path, device=device)
        return model, {"train": [], "val": []}

    print(f"[AE] training new model -> {ckpt_path}")
    model = MNISTConvAutoencoder(embed_dim=int(embed_dim))
    hist = train_autoencoder(
        model=model,
        X_train=X_train,
        X_val=X_val,
        device=device,
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        seed=int(seed),
    )
    save_autoencoder(ckpt_path, model)
    return model, hist
