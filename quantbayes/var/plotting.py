import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_results(y, samples, T, K):
    mean_prediction = samples["mu"].mean(axis=0)
    lower_bound = jnp.percentile(samples["mu"], 2.5, axis=0)
    upper_bound = jnp.percentile(samples["mu"], 97.5, axis=0)

    fig, axes = plt.subplots(K, 1, figsize=(10, 6), sharex=True)
    time_steps = jnp.arange(T)

    for i in range(K):
        axes[i].plot(time_steps, y[:, i], label=f"True Variable {i + 1}", color="blue")
        axes[i].plot(
            time_steps[2:],
            mean_prediction[:, i],
            label=f"Predicted Mean Variable {i + 1}",
            color="orange",
        )
        axes[i].fill_between(
            time_steps[2:],
            lower_bound[:, i],
            upper_bound[:, i],
            color="orange",
            alpha=0.2,
            label="95% CI",
        )
        axes[i].set_title(f"Variable {i + 1}")
        axes[i].legend()
        axes[i].grid(True)

    plt.xlabel("Time Steps")
    plt.tight_layout()
    plt.savefig("var2.png")
    plt.show()
