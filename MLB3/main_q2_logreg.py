"""
Runs all logistic‑regression experiments for Questions 3–6.

Usage:
    python main_q2_logreg.py
Creates:
    images/loss_const_gamma001.png
    images/loss_batchsize.png
    images/loss_diminish.png
and prints loss values to terminal
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_mnist_digits


# --------------------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_loss(w, b, X, y, mu):
    z = X @ w + b
    p = sigmoid(z)
    # avoid log(0) with clipping
    p = np.clip(p, 1e-12, 1 - 1e-12)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    loss += 0.5 * mu * (np.dot(w, w) + b**2)
    return loss


def grad(w, b, X, y, mu):
    z = X @ w + b
    p = sigmoid(z)
    g_w = (X.T @ (p - y)) / len(y) + mu * w
    g_b = np.mean(p - y) + mu * b
    return g_w, g_b


def gd_constant(X, y, X_test, y_test, gamma, T, mu):
    w = np.zeros(X.shape[1])
    b = 0.0
    losses = []
    for t in range(1, T + 1):
        g_w, g_b = grad(w, b, X, y, mu)
        w -= gamma * g_w
        b -= gamma * g_b
        if t % 10 == 0:
            losses.append(compute_loss(w, b, X, y, mu))
    return np.array(losses)


def sgd_mini_batch(X, y, batch_size, gamma_schedule, T, mu):
    rng = np.random.default_rng()
    w = np.zeros(X.shape[1])
    b = 0.0
    losses = []
    for t in range(1, T + 1):
        idx = rng.choice(len(X), batch_size, replace=False)
        g_w, g_b = grad(w, b, X[idx], y[idx], mu)
        gamma = gamma_schedule(t)
        w -= gamma * g_w
        b -= gamma * g_b
        if t % 10 == 0:
            losses.append(compute_loss(w, b, X, y, mu))
    return np.array(losses)


def plot_losses(loss_mat, labels, fname, ylabel="Training loss"):
    plt.figure()
    for losses, label in zip(loss_mat, labels):
        plt.plot(np.arange(10, 10 * len(losses) + 1, 10), losses, label=label)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.legend()
    Path(fname).parent.mkdir(exist_ok=True)
    plt.savefig(fname, dpi=150)
    plt.close()


def main():
    (X_train, y_train), (X_test, y_test) = load_mnist_digits()
    mu = 1.0

    # ------------------------------------------------ Q3
    gd_loss = gd_constant(X_train, y_train, X_test, y_test, gamma=0.001, T=100, mu=mu)
    sgd_loss = sgd_mini_batch(
        X_train, y_train, 10, gamma_schedule=lambda t: 0.001, T=100, mu=mu
    )
    plot_losses(
        [gd_loss, sgd_loss],
        [r"GD, $\gamma=10^{-3}$", r"SGD $b=10$, $\gamma=10^{-3}$"],
        "images/loss_const_gamma001.png",
    )

    # ------------------------------------------------ Q4
    print("\n=== Learning‑rate sweep ===")
    lrs = [1e-4, 1e-3, 1e-2]
    for lr in lrs:
        gd_final = gd_constant(
            X_train, y_train, X_test, y_test, gamma=lr, T=100, mu=mu
        )[-1]
        sgd_final = sgd_mini_batch(
            X_train, y_train, 10, gamma_schedule=lambda t, lr=lr: lr, T=100, mu=mu
        )[-1]
        print(f"γ={lr:<7}  GD loss={gd_final:.4f}  SGD loss={sgd_final:.4f}")

    # ------------------------------------------------ Q5
    print("\n=== Batch‑size sweep (constant γ) ===")
    best_lr = 0.001  # update after inspecting the sweep if needed
    batch_losses = []
    labels = []
    for bsz in (1, 10, 100):
        losses = sgd_mini_batch(
            X_train, y_train, bsz, gamma_schedule=lambda t: best_lr, T=100, mu=mu
        )
        batch_losses.append(losses)
        labels.append(f"b={bsz}")
        print(f"b={bsz:<3}  final loss={losses[-1]:.4f}")
    plot_losses(batch_losses, labels, "images/loss_batchsize.png")

    # ------------------------------------------------ Q6
    diminish_losses = sgd_mini_batch(
        X_train, y_train, 10, gamma_schedule=lambda t: 1.0 / t, T=100, mu=mu
    )
    best_const_losses = sgd_mini_batch(
        X_train, y_train, 10, gamma_schedule=lambda t: best_lr, T=100, mu=mu
    )
    plot_losses(
        [diminish_losses, best_const_losses],
        [r"$\gamma_t=1/t$", rf"Const $\gamma={best_lr}$"],
        "images/loss_diminish.png",
    )


if __name__ == "__main__":
    main()
