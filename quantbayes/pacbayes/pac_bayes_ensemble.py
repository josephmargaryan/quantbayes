# state_of_art_pacbayes_heavytail_viz.py
# Modular, open-source-ready implementation of ERM and multiple PAC-Bayes objectives
# Includes Hessian spectrum diagnostics alongside rich visualization suite

import os
import logging
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.autograd.functional import hessian

# ----------- Configuration -----------
CONFIG = {
    "seed": 42,
    "n_samples": 5000,
    "dim": 20,
    "noise_std": 0.1,
    "batch_size": 512,
    "epochs": 200,
    "lr": 1e-2,
    "prior_sigma": 1.0,
    "delta": 0.05,
    "c": 1.1,
    "K_mc": 30,
    "plots_dir": "plots",
    "objectives": [
        "ERM",
        "PB-KL",
        "PB-EB",
    ],  # ERM, PAC-Bayes-kl, PAC-Bayes-Empirical-Bernstein
}

# Create plots directory
os.makedirs(CONFIG["plots_dir"], exist_ok=True)
logging.basicConfig(level=logging.INFO)


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------- Data Loading -----------
def load_synthetic(config):
    seed_everything(config["seed"])
    n, d, noise = config["n_samples"], config["dim"], config["noise_std"]
    X = np.random.randn(n, d)
    w0 = np.zeros(d)
    w0[:5] = np.linspace(0.5, -0.5, 5)
    y = X.dot(w0) + noise * np.random.randn(n)
    X = StandardScaler().fit_transform(X)
    y = (y - y.min()) / (y.max() - y.min())
    return train_test_split(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
        test_size=0.2,
        random_state=config["seed"],
    )


# ----------- Models -----------
class GaussianPosterior(nn.Module):
    """Diagonal Gaussian posterior with reparameterization."""

    def __init__(self, dim, init_rho=-3.0):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(dim))
        self.rho = nn.Parameter(torch.full((dim,), init_rho))

    def sample(self, K):
        eps = torch.randn(K, self.mu.size(0), device=self.mu.device)
        sigma = torch.log1p(self.rho.exp())
        return self.mu.unsqueeze(0) + eps * sigma.unsqueeze(0), sigma

    def kl_div(self, prior_sigma):
        sigma = torch.log1p(self.rho.exp())
        return 0.5 * torch.sum(
            (self.mu**2 + sigma**2 - torch.log(sigma**2) - 1) / prior_sigma**2
        )


# ----------- Objectives -----------
def empirical_loss_and_variance(model, X, y, K):
    W, _ = model.sample(K)
    losses = (W @ X.T - y.unsqueeze(0)) ** 2
    Ln = losses.mean()
    Vn = ((losses - losses.mean(dim=1, keepdim=True)) ** 2).mean()
    return Ln, Vn


def obj_erm(model, X, y, config=None, **kwargs):
    # linear model in mu only
    y_pred = X @ model.mu
    loss = ((y_pred - y) ** 2).mean()
    return loss, {}


def obj_pbkl(model, X, y, config, **kwargs):
    Ln, _ = empirical_loss_and_variance(model, X, y, config["K_mc"])
    kl = model.kl_div(config["prior_sigma"])
    lam = kwargs.get("lam", 1.0 / config["n_samples"])
    return Ln + lam * kl, {"Ln": Ln.item(), "KL": kl.item()}


def obj_pbeb(model, X, y, config, **kwargs):
    n = X.size(0)
    Ln, Vn = empirical_loss_and_variance(model, X, y, config["K_mc"])
    kl = model.kl_div(config["prior_sigma"])
    nu1 = (
        math.ceil(
            math.log(
                math.sqrt((math.e - 2) * n / (4 * math.log(2 / config["delta"])))
                + 1e-12
            )
            / math.log(config["c"])
        )
        + 1
    )
    term = (1 + config["c"]) * torch.sqrt(
        (math.e - 2) * Vn * (kl + math.log(2 * nu1 / config["delta"])) / n
    )
    lower = 2 * (kl + math.log(2 * nu1 / config["delta"])) / n
    return Ln + term + lower, {"Ln": Ln.item(), "Vn": Vn.item(), "KL": kl.item()}


OBJ_FUNCS = {"ERM": obj_erm, "PB-KL": obj_pbkl, "PB-EB": obj_pbeb}

# ----------- Training Pipeline -----------
# Unified training for any nn.Module with objective in OBJ_FUNCS


def train_model(model, X_tr, y_tr, X_val, y_val, X_te, y_te, config, objective):
    stats = {f"{objective}_train": [], f"{objective}_val": [], f"{objective}_test": []}
    if objective.startswith("PB"):
        stats[f"{objective}_kl"] = []
        stats[f"{objective}_var"] = []
        stats[f"{objective}_norm"] = []  # track ||mu||
        stats[f"{objective}_sigma"] = []  # track mean sigma
    if objective == "PB-EB":
        stats[f"{objective}_bound"] = []
    opt = optim.Adam(model.parameters(), lr=config["lr"])
    best_val = float("inf")
    best_state = None

    for ep in range(1, config["epochs"] + 1):
        model.train()
        obj, extra = OBJ_FUNCS[objective](model, X_tr, y_tr, config)
        opt.zero_grad()
        obj.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            tr_obj, _ = OBJ_FUNCS[objective](model, X_tr, y_tr, config)
            val_obj, _ = OBJ_FUNCS[objective](model, X_val, y_val, config)
            test_obj, extra_test = OBJ_FUNCS[objective](model, X_te, y_te, config)

        stats[f"{objective}_train"].append(tr_obj.item())
        stats[f"{objective}_val"].append(val_obj.item())
        stats[f"{objective}_test"].append(test_obj.item())
        if objective.startswith("PB"):
            stats[f"{objective}_kl"].append(extra.get("KL", 0))
            stats[f"{objective}_var"].append(extra.get("Vn", 0))
            if objective == "PB-EB":
                stats[f"{objective}_bound"].append(test_obj.item())

        logging.info(
            f"{objective} Ep {ep}: train={tr_obj:.4f}, val={val_obj:.4f}, test={test_obj:.4f}"
        )
        if val_obj < best_val:
            best_val = val_obj
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return stats


# ----------- Hessian Spectrum -----------
def compute_hessian_spectrum(model, X, y):
    # compute Hessian of ERM loss w.r.t mu (dimension d x d)
    def loss_fn(mu_vec):
        pred = X @ mu_vec
        return ((pred - y) ** 2).mean()

    H = hessian(loss_fn, model.mu)
    eigs = torch.linalg.eigvalsh(H)
    return eigs.detach().cpu().numpy()


# ----------- Visualization -----------
def plot_and_save(x, labels, title, ylabel, fname):
    plt.figure()
    for arr, lbl in zip(x, labels):
        plt.plot(arr, label=lbl)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["plots_dir"], fname))


# ----------- Main Execution -----------
if __name__ == "__main__":
    seed_everything(CONFIG["seed"])
    X_tr, X_te, y_tr, y_te = load_synthetic(CONFIG)
    # Ensure torch.Tensor types
    X_tr = X_tr if isinstance(X_tr, torch.Tensor) else torch.from_numpy(X_tr)
    X_te = X_te if isinstance(X_te, torch.Tensor) else torch.from_numpy(X_te)
    y_tr = y_tr if isinstance(y_tr, torch.Tensor) else torch.from_numpy(y_tr)
    y_te = y_te if isinstance(y_te, torch.Tensor) else torch.from_numpy(y_te)

    all_stats = {}

    # ERM as a proper nn.Module
    class ERMModel(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.mu = nn.Parameter(torch.zeros(dim))

        def forward(self, X):
            return X @ self.mu

    # Instantiate ERM model and warm-start
    erm_model = ERMModel(CONFIG["dim"])
    ridge_w = Ridge(alpha=1.0).fit(X_tr.numpy(), y_tr.numpy()).coef_
    erm_model.mu.data = torch.from_numpy(ridge_w).float()
    erm_stats = train_model(
        erm_model, X_tr, y_tr, X_te, y_te, X_te, y_te, CONFIG, "ERM"
    )
    all_stats["ERM"] = erm_stats

    # PAC-Bayes models
    for obj in ["PB-KL", "PB-EB"]:
        model = GaussianPosterior(CONFIG["dim"])
        # warm-start
        ridge_w = Ridge(alpha=1.0).fit(X_tr.numpy(), y_tr.numpy()).coef_
        model.mu.data = torch.from_numpy(ridge_w).float()
        all_stats[obj] = train_model(
            model, X_tr, y_tr, X_te, y_te, X_te, y_te, CONFIG, obj
        )
        # Hessian at final mu
        eigs = compute_hessian_spectrum(model, X_tr, y_tr)
        np.savetxt(os.path.join(CONFIG["plots_dir"], f"hessian_{obj}.txt"), eigs)

    # Visualization calls
    # 1) ERM vs PB-EB train/val
    plot_and_save(
        [
            erm_stats["ERM_train"],
            erm_stats["ERM_val"],
            all_stats["PB-EB"]["PB-EB_train"],
            all_stats["PB-EB"]["PB-EB_val"],
        ],
        ["ERM Train", "ERM Val", "PB-EB Train", "PB-EB Val"],
        "ERM vs PB-EB Training/Validation",
        "Objective",
        "erm_vs_pbeb.png",
    )
    # 2) Generalization gap
    gap_erm = np.abs(np.array(erm_stats["ERM_train"]) - np.array(erm_stats["ERM_val"]))
    gap_eb = np.abs(
        np.array(all_stats["PB-EB"]["PB-EB_train"])
        - np.array(all_stats["PB-EB"]["PB-EB_val"])
    )
    plot_and_save(
        [gap_erm, gap_eb],
        ["ERM Gap", "PB-EB Gap"],
        "Generalization Gap",
        "Gap",
        "gen_gap.png",
    )
    # 3) Bound vs test MSE
    plot_and_save(
        [all_stats["PB-EB"]["PB-EB_test"], all_stats["PB-EB"]["PB-EB_bound"]],
        ["Test MSE", "PB-EB Bound"],
        "Bound vs Test MSE",
        "Value",
        "bound_vs_testmse.png",
    )
    # 4) Bound tightness
    tight = np.array(all_stats["PB-EB"]["PB-EB_bound"]) - np.array(
        all_stats["PB-EB"]["PB-EB_test"]
    )
    plot_and_save(
        [tight],
        ["Bound-Test"],
        "Bound Tightness",
        "Bound - Test MSE",
        "bound_tightness.png",
    )
    # 5) KL vs var
    plot_and_save(
        [all_stats["PB-EB"]["PB-EB_kl"], all_stats["PB-EB"]["PB-EB_var"]],
        ["KL", "Emp Var"],
        "KL vs Empirical Variance",
        "Term",
        "kl_vs_var.png",
    )
    # 6) Posterior norm & sigma
    plot_and_save(
        [all_stats["PB-EB"]["PB-EB_norm"], all_stats["PB-EB"]["PB-EB_sigma"]],
        ["||mu||", "mean sigma"],
        "Posterior Norm & Spread",
        "Value",
        "norm_sigma.png",
    )

    logging.info(
        "Hessian eigenvalues saved in plots/hessian_PB-KL.txt and hessian_PB-EB.txt"
    )
    logging.info(f"All plots saved in '{CONFIG['plots_dir']}'")
