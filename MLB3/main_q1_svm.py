"""
Runs Question 1 + Question 2 SVM experiments.

Usage:
    python3 main_q1_svm.py --data nonlinear_svm_data.csv

Creates:
    images/linear_decision.png
    images/rbf_decision.png
and prints a formatted results table.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import hinge_loss

from data_utils import load_nonlinear_svm_csv
from kernels import gaussian_gamma


def decision_plot(clf, X, y, fname: Path):
    """Helper: 2-D decision boundary plot."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=40, cmap="bwr_r")
    plt.title(fname.stem.replace("_", " ").title())
    fname.parent.mkdir(exist_ok=True)
    plt.savefig(fname, dpi=150)
    plt.close()


def run_linear(X, y):
    print("\n=== Linear SVM (kernel='linear') ===")
    print(f"{'C':>6}  {'hinge_loss':>10}  {'||w||':>7}  {'#SV':>4}")
    for C in (1, 100, 1000):
        clf = SVC(kernel="linear", C=C)
        clf.fit(X, y)

        loss = hinge_loss(y, clf.decision_function(X))
        norm_w = np.linalg.norm(clf.coef_)
        n_sv = len(clf.support_)

        print(f"{C:6d}  {loss:10.4f}  {norm_w:7.3f}  {n_sv:4d}")

        # Save one illustrative decision boundary
        if C == 100:
            decision_plot(clf, X, y, Path("images/linear_decision.png"))


def run_rbf(X, y):
    print("\n=== RBF-kernel SVM ===")
    print(f"{'a':>4}  {'gamma':>8}  {'hinge_loss':>10}  {'#SV':>4}")
    for a in (0.1, 1, 10):
        gamma = gaussian_gamma(a)
        clf = SVC(kernel="rbf", C=1, gamma=gamma)
        clf.fit(X, y)

        loss = hinge_loss(y, clf.decision_function(X))
        n_sv = len(clf.support_)

        print(f"{a:4.1f}  {gamma:8.4g}  {loss:10.4f}  {n_sv:4d}")

        # Save one illustrative decision boundary
        if a == 1:
            decision_plot(clf, X, y, Path("images/rbf_decision.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV file for nonlinear SVM data")
    args = ap.parse_args()

    X, y = load_nonlinear_svm_csv(args.data)
    run_linear(X, y)
    run_rbf(X, y)


if __name__ == "__main__":
    main()
