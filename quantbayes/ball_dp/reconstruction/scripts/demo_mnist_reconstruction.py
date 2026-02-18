# quantbayes/ball_dp/reconstruction/scripts/demo_mnist_reconstruction.py
from __future__ import annotations

import argparse
import math
import pathlib
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as tud
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

from quantbayes.ball_dp.reconstruction.reporting import (
    ensure_dir,
    save_csv,
    save_json,
    plot_hist,
    plot_image_grid,
)
from quantbayes.ball_dp.reconstruction.convex.equation_solvers import (
    SoftmaxEquationSolver,
)


# -----------------------------
# Public encoder (toy)
# Replace with your ResNet encoder when you re-add vision modules.
# -----------------------------
class SmallMNISTEncoder(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Decoder (post-processing)
# Embedding -> pixels for visualization
# -----------------------------
class MLPDecoder(nn.Module):
    def __init__(self, embed_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, e):
        return self.net(e)


def load_mnist(n_train: int, seed: int = 0):
    tfm = T.Compose([T.ToTensor()])
    train = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=tfm
    )

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train), size=n_train, replace=False)

    X, y = [], []
    for i in idx:
        xi, yi = train[int(i)]
        X.append(xi.numpy())  # (1,28,28)
        y.append(int(yi))
    X = np.stack(X, axis=0)  # (N,1,28,28)
    y = np.array(y, dtype=np.int64)
    return X, y


def flatten_pixels(X: np.ndarray) -> np.ndarray:
    return X.reshape((X.shape[0], -1)).astype(np.float64)


@torch.no_grad()
def compute_embeddings(encoder: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    encoder.eval()
    xb = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    e = encoder(xb).cpu().numpy().astype(np.float64)
    return e


def train_decoder(
    *,
    E: np.ndarray,  # (N,d)
    X: np.ndarray,  # (N,1,28,28)
    device: str,
    hidden: int = 512,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
) -> nn.Module:
    torch.manual_seed(seed)
    dec = MLPDecoder(embed_dim=E.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    E_t = torch.from_numpy(E).to(device=device, dtype=torch.float32)
    X_t = (
        torch.from_numpy(X).to(device=device, dtype=torch.float32).view(X.shape[0], -1)
    )

    ds = tud.TensorDataset(E_t, X_t)
    dl = tud.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    losses = []
    dec.train()
    for ep in range(1, epochs + 1):
        s = 0.0
        nb = 0
        for eb, xb in dl:
            opt.zero_grad(set_to_none=True)
            pred = dec(eb)
            loss = loss_fn(pred, xb)
            loss.backward()
            opt.step()
            s += float(loss.item())
            nb += 1
        losses.append(s / max(1, nb))
        print(f"[Decoder] epoch {ep:02d} | mse={losses[-1]:.6f}")

    # plot decoder loss
    plt.figure()
    plt.plot(losses)
    plt.title("Decoder training loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.show()
    plt.close()

    dec.eval()
    return dec


def train_softmax_lbfgs_torch(
    *,
    X: np.ndarray,  # (N,d)
    y: np.ndarray,  # (N,)
    lam: float,
    max_iter: int = 500,
    tol_grad: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Deterministic-ish LBFGS to get a very accurate optimum.
    Returns (W, b, stationarity_norm) where stationarity_norm = ||sum_grad + lam*n*Wt||_F
    """
    import torch.optim as optim

    X_t = torch.from_numpy(np.asarray(X, dtype=np.float64))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.int64))

    K = int(np.max(y) + 1)
    d = int(X.shape[1])

    W = torch.zeros((K, d), dtype=torch.float64, requires_grad=True)
    b = torch.zeros((K,), dtype=torch.float64, requires_grad=True)

    def closure():
        opt.zero_grad(set_to_none=True)
        logits = X_t @ W.t() + b
        loss = torch.nn.functional.cross_entropy(logits, y_t, reduction="mean")
        reg = 0.5 * float(lam) * (W.pow(2).sum() + b.pow(2).sum())
        obj = loss + reg
        obj.backward()
        return obj

    opt = optim.LBFGS(
        [W, b],
        lr=1.0,
        max_iter=int(max_iter),
        tolerance_grad=float(tol_grad),
        tolerance_change=1e-18,
        line_search_fn="strong_wolfe",
        history_size=50,
    )
    opt.step(closure)

    # stationarity check
    with torch.no_grad():
        logits = X_t @ W.t() + b
        p = torch.softmax(logits, dim=1)
        diff = p.clone()
        diff[torch.arange(X_t.shape[0]), y_t] -= 1.0
        # sum grad wrt W is diff^T @ X; sum grad wrt b is diff^T @ 1
        sum_gW = diff.t() @ X_t
        sum_gb = diff.sum(dim=0)
        n = X_t.shape[0]
        stat = torch.norm(sum_gW + float(lam) * float(n) * W, p="fro") + torch.norm(
            sum_gb + float(lam) * float(n) * b, p=2
        )

    return W.detach().numpy(), b.detach().numpy(), float(stat.item())


def softmax_missing_gradient_fast(
    *,
    W: np.ndarray,
    b: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    target_idx: int,
    lam: float,
) -> np.ndarray:
    """
    Compute G_missing for one target efficiently by using full sum_grad and subtracting grad(target).
    Works even if optimum is approximate.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    n = X.shape[0]
    K = int(b.shape[0])

    Xt = np.concatenate([X, np.ones((n, 1), dtype=np.float64)], axis=1)  # (n,d+1)
    Wt = np.concatenate(
        [W.astype(np.float64), b.astype(np.float64).reshape(-1, 1)], axis=1
    )  # (K,d+1)

    logits = Xt @ Wt.T
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p = p / p.sum(axis=1, keepdims=True)  # (n,K)

    diff = p
    diff[np.arange(n), y] -= 1.0  # (n,K)

    sum_grad = diff.T @ Xt  # (K,d+1)

    j = int(target_idx)
    grad_j = np.outer(diff[j], Xt[j])  # (K,d+1)

    # sum over D^- is sum_grad - grad_j
    G_missing = -float(lam) * float(n) * Wt - (sum_grad - grad_j)
    return G_missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pixel", "embedding"], default="embedding")
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--lbfgs-max-iter", type=int, default=800)
    p.add_argument("--lbfgs-tol-grad", type=float, default=1e-11)

    p.add_argument("--n-targets", type=int, default=32)
    p.add_argument("--n-show", type=int, default=12)

    p.add_argument("--save-dir", type=str, default="./artifacts/mnist_recon_softmax")
    p.add_argument("--decoder-epochs", type=int, default=5)
    p.add_argument("--decoder-hidden", type=int, default=512)

    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    outdir = ensure_dir(args.save_dir)

    Ximg, y = load_mnist(args.n_train, seed=args.seed)

    # feature space
    if args.mode == "pixel":
        Xfeat = flatten_pixels(Ximg)
        decoder = None
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        enc = SmallMNISTEncoder(embed_dim=64).to(device)
        Xfeat = compute_embeddings(enc, Ximg, device=device)
        decoder = train_decoder(
            E=Xfeat,
            X=Ximg,
            device=device,
            epochs=int(args.decoder_epochs),
            hidden=int(args.decoder_hidden),
            seed=args.seed,
        )

    # train convex softmax ERM (noiseless optimum)
    W, b, stationarity = train_softmax_lbfgs_torch(
        X=Xfeat,
        y=y,
        lam=float(args.lam),
        max_iter=int(args.lbfgs_max_iter),
        tol_grad=float(args.lbfgs_tol_grad),
    )
    print(
        f"[Softmax ERM] stationarity_norm ≈ {stationarity:.4e}  (smaller is better; aim ~1e-6 or below)"
    )

    solver = SoftmaxEquationSolver(
        lam=float(args.lam), n_total=int(Xfeat.shape[0]), include_bias=True
    )

    # choose targets
    idx_all = np.arange(Xfeat.shape[0])
    targ_idx = rng.choice(
        idx_all, size=min(int(args.n_targets), idx_all.size), replace=False
    )

    rows = []
    err_list = []
    resid_list = []
    y_acc = []

    # visualization buffers
    viz_images = []
    viz_titles = []

    # precompute NN structure for retrieval-style reconstruction
    # (for embedding mode it’s meaningful; for pixel mode it still works)
    Xpool = Xfeat.astype(np.float64)

    for t, j in enumerate(targ_idx):
        j = int(j)
        Gm = softmax_missing_gradient_fast(
            W=W, b=b, X=Xfeat, y=y, target_idx=j, lam=float(args.lam)
        )
        res = solver.factorize_missing_gradient(Gm)

        y_hat = res.label_hat if res.label_hat is not None else -999
        y_true = int(y[j])
        ok = res.status == "ok"

        if ok:
            e_hat = np.asarray(res.record_hat, dtype=np.float64).reshape(-1)
            e_true = np.asarray(Xfeat[j], dtype=np.float64).reshape(-1)
            err = float(np.linalg.norm(e_hat - e_true))
            resid = float(res.details.get("rank1_resid", np.nan))

            # nearest neighbor reconstruction
            dists = np.linalg.norm(Xpool - e_hat[None, :], axis=1)
            nn_idx = int(np.argmin(dists))
            nn_dist = float(dists[nn_idx])

            err_list.append(err)
            resid_list.append(resid)
            y_acc.append(1.0 if (y_hat == y_true) else 0.0)

            rows.append(
                {
                    "target_idx": j,
                    "y_true": y_true,
                    "y_hat": int(y_hat),
                    "status": res.status,
                    "err_l2": err,
                    "rank1_resid": resid,
                    "nn_idx": nn_idx,
                    "nn_dist": nn_dist,
                    "nn_y": int(y[nn_idx]),
                }
            )

            # visualization
            if t < int(args.n_show):
                x0 = Ximg[j, 0]  # (28,28)
                viz_images.append(x0)
                viz_titles.append(f"orig y={y_true}")

                if args.mode == "pixel":
                    xhat = e_hat.reshape(28, 28)
                    viz_images.append(xhat)
                    viz_titles.append(f"recon (pix) err={err:.3f}")
                    xnn = Ximg[nn_idx, 0]
                    viz_images.append(xnn)
                    viz_titles.append(f"NN y={int(y[nn_idx])} d={nn_dist:.2f}")
                else:
                    # decoder recon
                    with torch.no_grad():
                        et = torch.from_numpy(e_hat[None, :]).to(
                            device=device, dtype=torch.float32
                        )
                        xhat = decoder(et).cpu().numpy().reshape(28, 28)
                    viz_images.append(xhat)
                    viz_titles.append(f"dec(recon) err={err:.3f}")

                    xnn = Ximg[nn_idx, 0]
                    viz_images.append(xnn)
                    viz_titles.append(f"NN y={int(y[nn_idx])} d={nn_dist:.2f}")

        else:
            rows.append(
                {
                    "target_idx": j,
                    "y_true": y_true,
                    "y_hat": int(y_hat),
                    "status": res.status,
                    "err_l2": math.nan,
                    "rank1_resid": math.nan,
                    "nn_idx": -1,
                    "nn_dist": math.nan,
                    "nn_y": -1,
                }
            )

    # save artifacts
    save_csv(outdir / "metrics.csv", rows)

    summary = {
        "mode": args.mode,
        "n_train": int(args.n_train),
        "lam": float(args.lam),
        "lbfgs_max_iter": int(args.lbfgs_max_iter),
        "stationarity_norm": float(stationarity),
        "n_targets": int(len(targ_idx)),
        "n_ok": int(sum(1 for r in rows if r["status"] == "ok")),
        "mean_err_l2": float(np.nanmean(err_list)) if err_list else None,
        "median_err_l2": float(np.nanmedian(err_list)) if err_list else None,
        "mean_rank1_resid": float(np.nanmean(resid_list)) if resid_list else None,
        "label_acc": float(np.mean(y_acc)) if y_acc else None,
    }
    save_json(outdir / "summary.json", summary)

    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # plots
    if err_list:
        plot_hist(
            err_list,
            title=f"Softmax EqSolve reconstruction error (mode={args.mode})",
            xlabel="||e_hat - e||_2",
            save_path=outdir / "error_hist.png",
            show=True,
        )
    if resid_list:
        plot_hist(
            resid_list,
            title="Rank-1 factorization residual",
            xlabel="||G - a e^T||_F / ||G||_F",
            save_path=outdir / "rank1_resid_hist.png",
            show=True,
        )

    # recon grid
    if viz_images:
        ncols = 3
        nrows = int(math.ceil(len(viz_images) / ncols))
        plot_image_grid(
            viz_images,
            viz_titles,
            nrows=nrows,
            ncols=ncols,
            save_path=outdir / "recon_grid.png",
            show=True,
        )

    print(f"\n[Saved] {outdir}/metrics.csv")
    print(f"[Saved] {outdir}/summary.json")
    print(
        f"[Saved] {outdir}/error_hist.png, {outdir}/rank1_resid_hist.png, {outdir}/recon_grid.png"
    )


if __name__ == "__main__":
    main()
