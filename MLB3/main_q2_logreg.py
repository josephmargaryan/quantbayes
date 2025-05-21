"""
The following has been run in a kaggle notebook
for easy access to the data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

# ─────────────────────────────────────────────────────────────────────────────
# 1. SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_CSV = "/kaggle/input/digit-recognizer/train.csv"
DIGITS = (3, 8)
MU = 1.0
T = 200
RECORD_EVERY = 10
GAMMAS = [1e-4, 1e-3, 1e-2]
BATCHES = [1, 10, 100]

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(TRAIN_CSV)
df = df[df.label.isin(DIGITS)]
y = (df.label == DIGITS[1]).astype(int).values
X = df.drop("label", axis=1).values.astype(float) / 255.0
n = X.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# 3. UTILITIES: sigmoid & loss
# ─────────────────────────────────────────────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def compute_loss(w, b, X_all, y_all, mu):
    z = X_all.dot(w) + b
    p = np.clip(sigmoid(z), 1e-12, 1 - 1e-12)
    loss = -np.mean(y_all * np.log(p) + (1 - y_all) * np.log(1 - p))
    loss += 0.5 * mu * np.dot(w, w)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# 4. RUN FUNCTION (full‐batch GD, mini‐batch SGD, and invscaling)
# ─────────────────────────────────────────────────────────────────────────────
def run_sgd_sklearn(batch_size, eta0, inv_scaling=False, power_t=1.0):
    params = dict(
        loss="log_loss",
        penalty="l2",
        alpha=MU,
        fit_intercept=True,
        shuffle=True,
        random_state=0,
    )
    if inv_scaling:
        params.update(learning_rate="invscaling", eta0=eta0, power_t=power_t)
    else:
        params.update(learning_rate="constant", eta0=eta0)

    clf = SGDClassifier(**params)
    losses = []
    classes = np.array([0, 1])
    rng = np.random.default_rng(0)

    for t in range(1, T + 1):
        if batch_size >= n:
            Xb, yb = X, y
        else:
            idx = rng.choice(n, batch_size, replace=False)
            Xb, yb = X[idx], y[idx]

        if t == 1:
            clf.partial_fit(Xb, yb, classes=classes)
        else:
            clf.partial_fit(Xb, yb)

        if t % RECORD_EVERY == 0:
            w = clf.coef_.ravel()
            b = float(clf.intercept_[0])
            losses.append(compute_loss(w, b, X, y, MU))

    return np.array(losses)


# ─────────────────────────────────────────────────────────────────────────────
# Q3: full‐batch GD vs. mini‐batch SGD (γ = 1e-3, b = 10)
# ─────────────────────────────────────────────────────────────────────────────
loss_gd = run_sgd_sklearn(batch_size=n, eta0=1e-3)
loss_sgd = run_sgd_sklearn(batch_size=10, eta0=1e-3)

iters = np.arange(RECORD_EVERY, T + 1, RECORD_EVERY)
plt.figure()
plt.plot(iters, loss_gd, label="GD (full-batch) γ=1e-3")
plt.plot(iters, loss_sgd, label="SGD (b=10) γ=1e-3")
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.title("Q3: GD vs. mini-batch SGD")
plt.legend()
plt.show()

print(f"Q3 final loss → GD:  {loss_gd[-1]:.4f}, SGD: {loss_sgd[-1]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Q4: learning‐rate sweep for both algorithms
# ─────────────────────────────────────────────────────────────────────────────
finals = {}
plt.figure(figsize=(8, 6))
for γ in GAMMAS:
    lg = run_sgd_sklearn(batch_size=n, eta0=γ)
    ls = run_sgd_sklearn(batch_size=10, eta0=γ)
    finals[γ] = (lg[-1], ls[-1])
    plt.plot(iters, lg, "--", label=f"GD γ={γ}")
    plt.plot(iters, ls, "-", label=f"SGD γ={γ}")
    print(f"γ={γ}: GD final={lg[-1]:.4f}, SGD final={ls[-1]:.4f}")

plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.title("Q4: Effect of γ on GD & SGD")
plt.legend()
plt.show()

best_gamma = min(GAMMAS, key=lambda g: finals[g][1])
print(f"Best γ for SGD: {best_gamma}")

# ─────────────────────────────────────────────────────────────────────────────
# Q5: batch‐size sweep (γ = best from Q4)
# ─────────────────────────────────────────────────────────────────────────────
plt.figure()
for b in BATCHES:
    lb = run_sgd_sklearn(batch_size=b, eta0=best_gamma)
    plt.plot(iters, lb, label=f"b = {b}")
    print(f"b={b} → final loss={lb[-1]:.4f}")

plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.title(f"Q5: Batch‐size (γ={best_gamma})")
plt.legend()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Q6: diminishing rate γₜ = 1/t vs. best constant γ
# ─────────────────────────────────────────────────────────────────────────────
loss_dim = run_sgd_sklearn(batch_size=10, eta0=1.0, inv_scaling=True, power_t=1.0)
loss_const = run_sgd_sklearn(batch_size=10, eta0=best_gamma, inv_scaling=False)

plt.figure()
plt.plot(iters, loss_const, "--", label=f"const γ={best_gamma}")
plt.plot(iters, loss_dim, "-", label="diminishing γₜ = 1/t")
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.title("Q6: Constant vs. Diminishing Rate")
plt.legend()
plt.show()

print(
    f"Q6 final loss → constant γ={best_gamma}: {loss_const[-1]:.4f}, diminishing 1/t: {loss_dim[-1]:.4f}"
)
