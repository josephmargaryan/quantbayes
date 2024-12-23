import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(X, Y, X_test, mean_prediction, percentiles):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # Plot training data
    ax.plot(X, Y, "kx")
    # Plot 90% confidence level
    ax.fill_between(X_test, percentiles[0, :], percentiles[1, :], color="lightblue")
    # Plot mean prediction
    ax.plot(X_test, mean_prediction, "blue", lw=2.0)

    ax.set(xlabel="X", ylabel="Y", title="Mean predictions with 90% CI")
    plt.savefig("gp_plot.pdf")
