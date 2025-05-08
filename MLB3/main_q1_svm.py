"""
Runs Question 1 + Question 2 SVM experiments.

Usage:
    python main_q1_svm.py --data path/to/nonlinear_svm_data.csv
Creates:
    images/linear_decision.png
    images/rbf_decision.png
and prints a formatted results table.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import hinge_loss

from data_utils import load_nonlinear_svm_csv
from kernels import gaussian_gamma


# --------------------------------------------------------------------------
def decision_plot(clf, X, y, fname):
    """Helper: 2‑D decision boundary plot."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=40, cmap="bwr_r")
    plt.title(f"{fname.stem.replace('_', ' ').title()}")
    fname.parent.mkdir(exist_ok=True)
    plt.savefig(fname, dpi=150)
    plt.close()


def run_linear(X, y):
    print("\n=== Linear SVM ===")
    for C in (1, 100, 1000):
        clf = LinearSVC(C=C, max_iter=5000, dual=True)
        clf.fit(X, y)
        loss = hinge_loss(y, clf.decision_function(X))
        print(
            f"C={C:<5}  loss={loss:.4f}  |w|={np.linalg.norm(clf.coef_):.3f}  "
            f"#SV (support_)={len(clf.coef_)}"
        )


def run_rbf(X, y):
    print("\n=== RBF kernel SVM ===")
    for a in (0.1, 1, 10):
        gamma = gaussian_gamma(a)
        clf = SVC(kernel="rbf", C=1, gamma=gamma)
        clf.fit(X, y)
        loss = hinge_loss(y, clf.decision_function(X))
        print(f"a={a:<4} (γ={gamma:.4g})  loss={loss:.4f}  #SV={len(clf.support_)}")
        if a == 1:  # single illustrative plot
            decision_plot(clf, X, y, Path("images/rbf_decision.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV file for nonlinear SVM data")
    args = ap.parse_args()

    X, y = load_nonlinear_svm_csv(args.data)
    run_linear(X, y)
    run_rbf(X, y)

    # One plot for linear case (C=100) as required
    clf = LinearSVC(C=100, max_iter=5000, dual=True).fit(X, y)
    decision_plot(clf, X, y, Path("images/linear_decision.png"))


if __name__ == "__main__":
    main()
