import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_results(year, data, pred_mean, pred_pi):
    """Plots the true and predicted results."""
    plt.figure(figsize=(8, 6), constrained_layout=True)
    plt.plot(year, data[:, 0], "ko", mfc="none", ms=4, label="true hare", alpha=0.67)
    plt.plot(year, data[:, 1], "bx", label="true lynx")
    plt.plot(year, pred_mean[:, 0], "k-.", label="pred hare", lw=1, alpha=0.67)
    plt.plot(year, pred_mean[:, 1], "b--", label="pred lynx")
    plt.fill_between(year, pred_pi[0, :, 0], pred_pi[1, :, 0], color="k", alpha=0.2)
    plt.fill_between(year, pred_pi[0, :, 1], pred_pi[1, :, 1], color="b", alpha=0.3)
    plt.gca().set(ylim=(0, 160), xlabel="year", ylabel="population (in thousands)")
    plt.title("Posterior predictive (80% CI) with predator-prey pattern.")
    plt.legend()
    plt.savefig("ode_plot.pdf")
