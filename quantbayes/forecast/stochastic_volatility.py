import jax.numpy as jnp
import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import matplotlib.pyplot as plt


class StochasticVolatilityModel:
    """
    Stochastic Volatility Model using NumPyro.
    """

    def __init__(self):
        self.mcmc = None
        self.posterior_samples = None

    @staticmethod
    def model(returns):
        """
        NumPyro implementation of the stochastic volatility model.

        Args:
            returns (jnp.ndarray): Observed returns (1D array of shape (N,)).
        """
        # Prior for the step size of Gaussian Random Walk
        step_size = numpyro.sample("sigma", dist.Exponential(50.0))

        # Latent log-volatility as a Gaussian Random Walk
        s = numpyro.sample(
            "s",
            dist.GaussianRandomWalk(scale=step_size, num_steps=jnp.shape(returns)[0]),
        )

        # Degrees of freedom for Student's t-distribution
        nu = numpyro.sample("nu", dist.Exponential(0.1))

        # Observed returns with time-varying volatility
        return numpyro.sample(
            "r", dist.StudentT(df=nu, loc=0.0, scale=jnp.exp(s)), obs=returns
        )

    def fit(self, returns, num_warmup=500, num_samples=1000):
        """
        Fit the stochastic volatility model to observed returns.

        Args:
            returns (jnp.ndarray): Observed returns (1D array of shape (N,)).
            num_warmup (int): Number of warmup steps for MCMC.
            num_samples (int): Number of posterior samples to draw.
        """
        # Define the NUTS sampler
        kernel = NUTS(self.model)

        # Run MCMC
        self.mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        self.mcmc.run(jax.random.PRNGKey(0), returns=returns)

        # Store posterior samples
        self.posterior_samples = self.mcmc.get_samples()

    def plot_latent_volatility(self, returns):
        """
        Visualize the latent log-volatility with uncertainty.

        Args:
            returns (jnp.ndarray): Observed returns (1D array of shape (N,)).
        """
        if self.posterior_samples is None:
            raise ValueError("Model has not been fitted yet. Call `fit` first.")

        # Extract latent log-volatility samples
        s_posterior = self.posterior_samples["s"]  # Shape: (num_samples, N)
        s_mean = s_posterior.mean(axis=0)
        s_lower = jnp.percentile(s_posterior, 5, axis=0)
        s_upper = jnp.percentile(s_posterior, 95, axis=0)

        # Plot observed returns
        plt.figure(figsize=(12, 6))
        plt.plot(returns, label="Observed Returns", color="blue", alpha=0.6)

        # Plot latent log-volatility
        plt.plot(s_mean, label="Latent Log-Volatility (Mean)", color="orange")
        plt.fill_between(
            range(len(s_mean)),
            s_lower,
            s_upper,
            color="orange",
            alpha=0.3,
            label="90% CI",
        )

        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Latent Log-Volatility with Observed Returns")
        plt.legend()
        plt.grid(True)
        plt.show()

    def sample_future_returns(self, returns, num_samples=100):
        """
        Generate future return samples using the posterior distribution.

        Args:
            returns (jnp.ndarray): Observed returns (1D array of shape (N,)).
            num_samples (int): Number of return samples to generate.

        Returns:
            jnp.ndarray: Future return samples of shape (num_samples, N).
        """
        if self.posterior_samples is None:
            raise ValueError("Model has not been fitted yet. Call `fit` first.")

        # Define a predictive distribution
        predictive = Predictive(self.model, self.posterior_samples)

        # Generate samples
        predictive_samples = predictive(jax.random.PRNGKey(1), returns=returns)
        return predictive_samples["r"]

    def summary(self):
        """
        Print the summary of the fitted model's posterior samples.
        """
        if self.mcmc is None:
            raise ValueError("Model has not been fitted yet. Call `fit` first.")
        print(self.mcmc.print_summary())


if __name__ == "__main__":
    import jax.numpy as jnp
    import numpy as np

    # from stochastic_volatility_model import StochasticVolatilityModel

    # Generate synthetic returns
    np.random.seed(42)
    N = 200
    true_volatility = 0.2 + 0.1 * np.sin(
        np.linspace(0, 10, N)
    )  # Time-varying volatility
    returns = np.random.normal(0, true_volatility, size=N)
    returns = jnp.array(returns)

    # Initialize and fit the model
    sv_model = StochasticVolatilityModel()
    sv_model.fit(returns, num_warmup=1000, num_samples=2000)
    # Print a summary of the posterior
    sv_model.summary()
    # Plot the latent volatility with uncertainty
    sv_model.plot_latent_volatility(returns)
    # Sample future returns
    future_samples = sv_model.sample_future_returns(returns, num_samples=100)

    # Plot some trajectories of sampled returns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    for i in range(10):  # Plot 10 sampled trajectories
        plt.plot(future_samples[i], alpha=0.7)
    plt.title("Sampled Future Returns")
    plt.xlabel("Time")
    plt.ylabel("Returns")
    plt.grid(True)
    plt.show()
