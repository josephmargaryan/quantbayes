# quantbayes/ball_dp/reconstruction/scripts/demo_mnist_reconstruction.py
from __future__ import annotations

import argparse
import numpy as np

# torch is used here only for dataset/optional encoder; reconstruction core is numpy.
import torch
import torch.nn as nn
import torch.utils.data as tud
import torchvision
import torchvision.transforms as T

from quantbayes.ball_dp.reconstruction import (
    PoolBallPrior,
    Candidate,
    SoftmaxEquationSolver,
    BinaryLogisticEquationSolver,
    SquaredHingeEquationSolver,
    RidgePrototypesEquationSolver,
    GaussianOutputIdentifier,
    ShadowModelIdentifier,
    vectorize_softmax,
    vectorize_binary_linear,
    vectorize_prototypes,
    vectorize_eqx_model,
)

# -----------------------------
# Simple public encoder/decoder
# (replace with your ResNet encoder + trained decoder if you want)
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


def load_mnist(n_train: int, n_test: int, seed: int = 0):
    tfm = T.Compose([T.ToTensor()])
    train = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=tfm
    )
    test = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=tfm
    )

    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(len(train), size=n_train, replace=False)
    te_idx = rng.choice(len(test), size=n_test, replace=False)

    def subset(ds, idx):
        X = []
        y = []
        for i in idx:
            xi, yi = ds[int(i)]
            X.append(xi.numpy())
            y.append(int(yi))
        X = np.stack(X, axis=0)  # (N,1,28,28)
        y = np.array(y, dtype=np.int64)
        return X, y

    Xtr, ytr = subset(train, tr_idx)
    Xte, yte = subset(test, te_idx)
    return Xtr, ytr, Xte, yte


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
    p.add_argument(
        "--head",
        choices=["prototypes", "softmax", "binlog", "sqhinge"],
        default="softmax",
    )
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)

    # reconstruction target + candidate set
    p.add_argument("--target-index", type=int, default=0)
    p.add_argument("--radius", type=float, default=5.0)
    p.add_argument("--m-candidates", type=int, default=50)

    # convex ERM params
    p.add_argument("--lam", type=float, default=1.0)

    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    Xtr, ytr, _, _ = load_mnist(args.n_train, args.n_test, seed=args.seed)

    # feature space selection
    if args.mode == "pixel":
        Feat = flatten_pixels(Xtr)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = SmallMNISTEncoder(embed_dim=64).to(device)

        # NOTE: for a real demo, you'd train this encoder on a public task;
        # here we keep it as a fixed public mapping for structural testing.
        Feat = compute_embeddings(encoder, Xtr, device=device)

    # pick a single target record
    j = int(args.target_index) % Feat.shape[0]
    x_target = Feat[j]
    y_target = int(ytr[j])

    # D^- = all but target
    mask = np.ones((Feat.shape[0],), dtype=bool)
    mask[j] = False
    X_minus = Feat[mask]
    y_minus = ytr[mask]

    # Build candidate pool inside the ball around the true target record (in feature space)
    prior = PoolBallPrior(
        pool_X=Feat, pool_y=ytr, radius=float(args.radius), label_fixed=y_target
    )
    candidates = prior.sample(center=x_target, m=int(args.m_candidates), rng=rng)

    print(
        f"[MNIST] mode={args.mode} head={args.head}  n={Feat.shape[0]}  target_y={y_target}  |C|={len(candidates)}"
    )

    # -----------------------------
    # Convex: equation solving demo
    # -----------------------------
    n_total = int(X_minus.shape[0] + 1)
    lam = float(args.lam)

    if args.head == "prototypes":
        # Fake a noiseless prototype release using the closed-form function you already have
        # (in real use: call quantbayes.ball_dp.heads.prototypes.fit_ridge_prototypes)
        from quantbayes.ball_dp.heads.prototypes import fit_ridge_prototypes

        K = int(np.max(ytr) + 1)
        mus, _ = fit_ridge_prototypes(Feat, ytr, num_classes=K, lam=lam)

        solver = RidgePrototypesEquationSolver(lam=lam, n_total=n_total)
        res = solver.reconstruct(
            release_mus=mus, d_minus=(X_minus, y_minus), label_known=y_target
        )

        err = float(np.linalg.norm(res.record_hat - x_target))
        print(f"[Convex][Prototypes] status={res.status}  ||z_hat-z||={err:.4f}")

    elif args.head == "softmax":
        # This demo assumes you already have a way to compute the TRUE optimum (W*,b*) on D.
        # If you already have a convex softmax trainer in reconstruction/convex_softmax.py,
        # replace the stub below with it.
        #
        # For now we do a tiny torch LBFGS just to get a deterministic optimum for the demo.

        import torch.optim as optim

        K = int(np.max(ytr) + 1)
        d = int(Feat.shape[1])

        W = torch.zeros((K, d), dtype=torch.float64, requires_grad=True)
        b = torch.zeros((K,), dtype=torch.float64, requires_grad=True)

        X_t = torch.from_numpy(Feat).to(dtype=torch.float64)
        y_t = torch.from_numpy(ytr).to(dtype=torch.long)

        def closure():
            opt.zero_grad(set_to_none=True)
            logits = X_t @ W.t() + b
            loss = torch.nn.functional.cross_entropy(logits, y_t, reduction="mean")
            reg = 0.5 * lam * (W.pow(2).sum() + b.pow(2).sum())
            obj = loss + reg
            obj.backward()
            return obj

        opt = optim.LBFGS([W, b], lr=1.0, max_iter=50, line_search_fn="strong_wolfe")
        opt.step(closure)

        W_np = W.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()

        solver = SoftmaxEquationSolver(lam=lam, n_total=n_total, include_bias=True)
        res = solver.reconstruct(W=W_np, b=b_np, d_minus=(X_minus, y_minus))

        err = float(np.linalg.norm(res.record_hat - x_target))
        print(
            f"[Convex][Softmax EqSolve] status={res.status}  y_hat={res.label_hat}  ||e_hat-e||={err:.4f}"
        )

    elif args.head == "binlog":
        # Binary logistic needs labels in {-1,+1}
        # We'll restrict to digits {0,1} for demo convenience.
        keep = np.where((ytr == 0) | (ytr == 1))[0]
        Feat2 = Feat[keep]
        y2 = ytr[keep]
        y_pm1 = np.where(y2 == 1, 1, -1).astype(np.int64)

        j = 0
        x_target = Feat2[j]
        y_target = int(y_pm1[j])

        X_minus = Feat2[1:]
        y_minus = y_pm1[1:]
        n_total = int(X_minus.shape[0] + 1)

        # train tiny torch logistic ERM on D (deterministic LBFGS)
        import torch.optim as optim

        d = int(Feat2.shape[1])
        wt = torch.zeros((d,), dtype=torch.float64, requires_grad=True)
        bt = torch.zeros((), dtype=torch.float64, requires_grad=True)

        X_t = torch.from_numpy(Feat2).to(dtype=torch.float64)
        y_t = torch.from_numpy(y_pm1).to(dtype=torch.float64)

        def closure():
            opt.zero_grad(set_to_none=True)
            logits = X_t @ wt + bt
            loss = torch.logaddexp(torch.zeros_like(logits), -y_t * logits).mean()
            reg = 0.5 * lam * (wt.pow(2).sum() + bt.pow(2))
            obj = loss + reg
            obj.backward()
            return obj

        opt = optim.LBFGS([wt, bt], lr=1.0, max_iter=50, line_search_fn="strong_wolfe")
        opt.step(closure)

        w_np = wt.detach().cpu().numpy()
        b_np = float(bt.detach().cpu().numpy())

        solver = BinaryLogisticEquationSolver(lam=lam, n_total=n_total)
        res = solver.reconstruct(w=w_np, b=b_np, d_minus=(X_minus, y_minus))

        err = float(np.linalg.norm(res.record_hat - x_target))
        print(
            f"[Convex][BinLog EqSolve] status={res.status}  y_hat={res.label_hat}  ||e_hat-e||={err:.4f}"
        )

    elif args.head == "sqhinge":
        # Same binary restriction {0,1} => ±1
        keep = np.where((ytr == 0) | (ytr == 1))[0]
        Feat2 = Feat[keep]
        y2 = ytr[keep]
        y_pm1 = np.where(y2 == 1, 1, -1).astype(np.int64)

        j = 0
        x_target = Feat2[j]
        y_target = int(y_pm1[j])

        X_minus = Feat2[1:]
        y_minus = y_pm1[1:]
        n_total = int(X_minus.shape[0] + 1)

        import torch.optim as optim

        d = int(Feat2.shape[1])
        wt = torch.zeros((d,), dtype=torch.float64, requires_grad=True)
        bt = torch.zeros((), dtype=torch.float64, requires_grad=True)

        X_t = torch.from_numpy(Feat2).to(dtype=torch.float64)
        y_t = torch.from_numpy(y_pm1).to(dtype=torch.float64)

        def closure():
            opt.zero_grad(set_to_none=True)
            logits = X_t @ wt + bt
            margin = 1.0 - y_t * logits
            loss = torch.clamp(margin, min=0.0).pow(2).mean()
            reg = 0.5 * lam * (wt.pow(2).sum() + bt.pow(2))
            obj = loss + reg
            obj.backward()
            return obj

        opt = optim.LBFGS([wt, bt], lr=1.0, max_iter=50, line_search_fn="strong_wolfe")
        opt.step(closure)

        w_np = wt.detach().cpu().numpy()
        b_np = float(bt.detach().cpu().numpy())

        solver = SquaredHingeEquationSolver(lam=lam, n_total=n_total)
        res = solver.reconstruct(w=w_np, b=b_np, d_minus=(X_minus, y_minus))

        if res.status == "no_support_vector":
            print(
                f"[Convex][SqHinge EqSolve] status=no_support_vector (as expected for non-support targets)"
            )
        else:
            err = float(np.linalg.norm(res.record_hat - x_target))
            print(
                f"[Convex][SqHinge EqSolve] status={res.status}  y_hat={res.label_hat}  ||e_hat-e||={err:.4f}"
            )

    else:
        raise ValueError("Unsupported head")


if __name__ == "__main__":
    main()
