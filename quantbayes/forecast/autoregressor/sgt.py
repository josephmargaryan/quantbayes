import jax.numpy as jnp
from jax import random
import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import matplotlib.pyplot as plt

def sgt_model(y, seasonality, future=0):
    """
    The SGT (Student's T Generalized) model definition for univariate time series.

    Args:
        y (jnp.array): Time series data of shape (N,).
        seasonality (int): Seasonal period (e.g., 12 for monthly data with yearly seasonality).
        future (int): Number of future steps to forecast.

    Returns:
        None. (NumPyro uses this as a probabilistic program.)
    """
    cauchy_sd = jnp.max(y) / 150.0

    # ========================
    # 1. Define Priors
    # ========================
    nu = numpyro.sample("nu", dist.Uniform(2, 20))
    powx = numpyro.sample("powx", dist.Uniform(0, 1))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(cauchy_sd))
    offset_sigma = numpyro.sample(
        "offset_sigma", dist.TruncatedCauchy(low=1e-10, loc=1e-10, scale=cauchy_sd)
    )
    coef_trend = numpyro.sample("coef_trend", dist.Cauchy(0, cauchy_sd))
    pow_trend_beta = numpyro.sample("pow_trend_beta", dist.Beta(1, 1))
    pow_trend = 1.5 * pow_trend_beta - 0.5
    pow_season = numpyro.sample("pow_season", dist.Beta(1, 1))
    level_sm = numpyro.sample("level_sm", dist.Beta(1, 2))
    s_sm = numpyro.sample("s_sm", dist.Uniform(0, 1))
    init_s = numpyro.sample(
        "init_s",
        dist.Cauchy(0, jnp.clip(y[:seasonality] * 0.3, a_min=1e-6))
    )

    # ========================
    # 2. Transition function
    # ========================
    def transition_fn(carry, t):
        level, s, moving_sum, rng_key = carry
        rng_key, subkey = random.split(rng_key)

        season = s[0] * level**pow_season
        exp_val = level + coef_trend * level**pow_trend + season
        exp_val = jnp.clip(exp_val, 0)

        # If t >= N, we sample "future" (post-training), else we follow observed y[t]
        y_t = jnp.where(t >= N, exp_val, y[t])

        # Rolling sum for seasonal adjustment
        moving_sum = (
            moving_sum
            + y_t
            - jnp.where(t >= seasonality, y[t - seasonality], 0.0)
        )

        # Update level
        level_p = jnp.where(t >= seasonality, moving_sum / seasonality, y_t - season)
        level = level_sm * level_p + (1 - level_sm) * level
        level = jnp.clip(level, 0)

        # Update seasonal state
        new_s = (s_sm * (y_t - level) / season + (1 - s_sm)) * s[0]
        new_s = jnp.where(t >= N, s[0], new_s)
        s = jnp.concatenate([s[1:], new_s[None]], axis=0)

        # Likelihood
        omega = sigma * exp_val**powx + offset_sigma
        y_ = numpyro.sample("y", dist.StudentT(nu, exp_val, omega), rng_key=subkey)

        return (level, s, moving_sum, rng_key), y_

    # ========================
    # 3. Initialization
    # ========================
    N = y.shape[0]
    level_init = y[0]
    s_init = jnp.concatenate([init_s[1:], init_s[:1]], axis=0)
    moving_sum = level_init
    rng_key = random.PRNGKey(0)

    # We place a condition on the observed portion (y[1:]) so the model sees actual data.
    with numpyro.handlers.condition(data={"y": y[1:]}):
        _, ys = scan(
            transition_fn,
            (level_init, s_init, moving_sum, rng_key),
            jnp.arange(1, N + future),
        )

    # Save the future forecast if future > 0
    if future > 0:
        numpyro.deterministic("y_forecast", ys[-future:])

class SGTModel:
    """
    SGTModel: A univariate time-series forecasting model using a 
    Student's T Generalized (SGT) approach in NumPyro, with NUTS MCMC.

    This class provides a scikit-learn-like interface:

        model = SGTModel(seasonality=12)
        model.fit(y_train)                       # y_train: jnp or np array
        forecast_samples = model.predict(steps=24)
        model.plot_forecast(y_train, forecast_samples)

    Args:
        seasonality (int): Seasonal period (e.g., 12 for monthly data).
        num_warmup (int): Number of warmup steps in MCMC.
        num_samples (int): Number of post-warmup MCMC samples.
        rng_key (int): Random seed or PRNGKey for MCMC runs.
    """
    def __init__(self, seasonality=12, num_warmup=500, num_samples=1000, rng_key=0):
        self.seasonality = seasonality
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.rng_key = rng_key
        self.mcmc = None
        self.fitted_ = False
        self.y_train = None

    def fit(self, y):
        """
        Fit the SGT model to the given univariate time series y.

        Args:
            y (array-like): 1D array of shape (N,) representing the time series.

        Returns:
            self: The fitted instance (for chaining).
        """
        self.y_train = jnp.array(y)  # Save training data
        nuts_kernel = NUTS(sgt_model)
        self.mcmc = MCMC(nuts_kernel, num_warmup=self.num_warmup, num_samples=self.num_samples)
        self.mcmc.run(random.PRNGKey(self.rng_key), y=self.y_train, seasonality=self.seasonality, future=0)
        self.fitted_ = True
        return self

    def predict(self, steps):
        """
        Predict future values for 'steps' time steps ahead.

        Args:
            steps (int): Number of future (out-of-sample) time points to forecast.

        Returns:
            jnp.array: Forecast samples of shape (num_samples, steps).
                       Each row is a sample from the predictive distribution 
                       for the entire future horizon.
        """
        if not self.fitted_:
            raise RuntimeError("SGTModel is not fitted yet. Call .fit(...) first.")
        samples = self.mcmc.get_samples()
        predictive = Predictive(sgt_model, samples, return_sites=["y_forecast"])
        # We'll pass in the original data again for the 'y' argument
        # but now with 'future=steps' to get the forecast
        chain = predictive(random.PRNGKey(self.rng_key + 1), 
                           y=self.y_train, 
                           seasonality=self.seasonality, 
                           future=steps)
        # shape => {"y_forecast": (num_samples, steps)}
        forecast_samples = chain["y_forecast"]
        return forecast_samples

    def plot_forecast(self, y_train, forecast_samples, ax=None, credible_interval=(5, 95)):
        """
        Plot the training data, forecast mean, and credible intervals.

        Args:
            y_train (array-like): 1D array of training data used in fit().
            forecast_samples (jnp.array): Forecast samples of shape (num_samples, future_steps)
                (output of self.predict(...)).
            ax (matplotlib.axes.Axes, optional): Axis to plot on. 
            credible_interval (tuple of int): Lower/upper percentile for uncertainty band.
        """
        import numpy as np

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        # Convert y_train to numpy if needed
        y_train = np.array(y_train)
        future_steps = forecast_samples.shape[1]

        # Plot training data
        ax.plot(y_train, label="Training Data", color="blue")

        # Plot forecast mean
        forecast_mean = jnp.mean(forecast_samples, axis=0)
        x_axis_forecast = range(len(y_train), len(y_train) + future_steps)
        ax.plot(
            x_axis_forecast,
            forecast_mean,
            label="Forecast Mean",
            color="orange",
            linestyle="--",
        )

        # Plot credible intervals
        lower_percentile, upper_percentile = credible_interval
        forecast_lower = jnp.percentile(forecast_samples, lower_percentile, axis=0)
        forecast_upper = jnp.percentile(forecast_samples, upper_percentile, axis=0)
        ax.fill_between(
            x_axis_forecast,
            forecast_lower,
            forecast_upper,
            alpha=0.3,
            color="orange",
            label=f"{upper_percentile - lower_percentile}% Credible Interval",
        )

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title("SGT Model Forecast")
        ax.legend()
        ax.grid(True)

        if ax is None:
            plt.show()

    def get_samples(self):
        """
        Retrieve the posterior samples from the fitted MCMC run.

        Returns:
            dict: A dictionary of posterior samples. 
        """
        if not self.fitted_:
            raise RuntimeError("Model not fitted yet; no samples to retrieve.")
        return self.mcmc.get_samples()


if __name__ == "__main__":
    from quantbayes.fake_data import create_synthetic_time_series 
    from sklearn.preprocessing import MinMaxScaler
    _, _, y_train, y_test = create_synthetic_time_series()
    scaler = MinMaxScaler()
    y_train = scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    # Suppose y is a 1D array or list of length N
    model = SGTModel(seasonality=12, num_warmup=500, num_samples=10000, rng_key=42)
    model.fit(y_train)

    forecast_samples = model.predict(steps=50)  # returns shape (num_samples, 24)
    fig, ax = plt.subplots(figsize=(10,5))
    model.plot_forecast(y_train, forecast_samples, ax=ax, credible_interval=(5,95))
    plt.show()
