import pickle
import jax
from numpyro.contrib.einstein import MixtureGuidePredictive, RBFKernel, SteinVI
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adagrad, Adam


class Module:
    """
    Base class for probabilistic models with support for modularized inference,
    prediction, and visualization.
    """

    def __init__(self, method="nuts"):
        """
        Initialize the Module.

        :param method: str
            Inference method ('nuts', 'svi', or 'steinvi').
        """
        self.method = method
        self.inference = None
        self.samples = None
        self.params = None
        self.losses = None
        self.stein_result = None

    def compile(self, guide=None, **kwargs):
        """
        Compiles the model for the specified inference method.

        :param guide: numpyro.autoguide.AutoGuide or any custom guide instance
            If None, defaults to AutoNormal(self.__call__) for SVI/SteinVI.
        :param kwargs: additional keyword arguments for inference setup.
        """
        if self.method == "nuts":
            kernel = NUTS(self.__call__)
            self.inference = MCMC(
                kernel,
                num_warmup=kwargs.get("num_warmup", 500),
                num_samples=kwargs.get("num_samples", 1000),
                num_chains=kwargs.get("num_chains", 1),
            )

        elif self.method == "svi":
            # Use provided guide instance or default to AutoNormal
            if guide is None:
                guide = AutoNormal(self.__call__)
            optimizer = Adam(kwargs.get("learning_rate", 0.01))
            self.inference = SVI(self.__call__, guide, optimizer, loss=Trace_ELBO())

        elif self.method == "steinvi":
            # Use provided guide instance or default to AutoNormal
            if guide is None:
                guide = AutoNormal(self.__call__)
            self.inference = SteinVI(
                model=self.__call__,
                guide=guide,
                optim=Adagrad(kwargs.get("learning_rate", 0.01)),
                kernel_fn=RBFKernel(),
                repulsion_temperature=kwargs.get("repulsion_temperature", 1.0),
                num_stein_particles=kwargs.get("num_stein_particles", 10),
                num_elbo_particles=kwargs.get("num_elbo_particles", 1),
            )

        else:
            raise ValueError(f"Unknown inference method: {self.method}")

    def fit(self, X_train, y_train, rng_key, **kwargs):
        """
        Fits the model using the selected inference method.

        :param X_train: jnp.ndarray
            Training features.
        :param y_train: jnp.ndarray
            Training targets.
        :param rng_key: jax.random.PRNGKey
            Random key for reproducibility.
        :param kwargs: additional keyword arguments, e.g., num_steps, progress_bar.
        """
        if isinstance(self.inference, MCMC):
            self.inference.run(rng_key, X_train, y_train)
            self.samples = self.inference.get_samples()

        elif isinstance(self.inference, SVI):
            svi_state = self.inference.init(rng_key, X_train, y_train)
            self.losses = []
            num_steps = kwargs.get("num_steps", 1000)
            for step in range(num_steps):
                svi_state, loss = self.inference.update(svi_state, X_train, y_train)
                self.losses.append(loss)
                # Print after completing each 100th step (i.e., steps: 100, 200, …)
                if (step + 1) % 100 == 0:
                    print(f"Step {step + 1}, Loss: {loss:.4f}")
            self.params = self.inference.get_params(svi_state)

        elif isinstance(self.inference, SteinVI):
            num_steps = kwargs.get("num_steps", 1000)
            self.stein_result = self.inference.run(
                rng_key,
                num_steps,
                X_train,
                y_train,
                progress_bar=kwargs.get("progress_bar", True),
            )

        else:
            raise ValueError("Inference method not initialized. Call `compile` first.")

    def predict(self, X_test, rng_key, posterior="logits", num_samples=None):
        """
        Generates predictions using the specified posterior.

        :param X_test: jnp.ndarray
            Test data for prediction.
        :param rng_key: jax.random.PRNGKey
            Random key for sampling predictions.
        :param posterior: str
            Name of the posterior to sample from (default: 'logits').
        :param num_samples: int
            Number of posterior samples to use (only for SVI/SteinVI).
        :return: dict or array
            Predictions for the requested posterior site.
        """
        if isinstance(self.inference, MCMC):
            predictive = Predictive(self.__call__, posterior_samples=self.samples)

        elif isinstance(self.inference, SVI):
            if self.params is None:
                raise ValueError("SVI parameters are not available. Ensure `fit` was called.")
            predictive = Predictive(
                self.__call__,
                guide=self.inference.guide,
                params=self.params,
                num_samples=num_samples or 100,
            )

        elif isinstance(self.inference, SteinVI):
            if self.stein_result is None:
                raise ValueError("SteinVI results are not available. Ensure `fit` was called.")
            params = self.inference.get_params(self.stein_result.state)
            predictive = MixtureGuidePredictive(
                model=self.__call__,
                guide=self.inference.guide,
                params=params,
                num_samples=num_samples or 100,
                guide_sites=self.inference.guide_sites,
            )

        else:
            raise ValueError("Inference method not initialized. Call `compile` first.")

        preds = predictive(rng_key, X_test)
        if posterior not in preds:
            raise ValueError(f"The posterior '{posterior}' is not available.")
        return preds[posterior]

    @property
    def get_samples(self):
        if isinstance(self.inference, MCMC):
            if self.samples is None:
                raise ValueError("MCMC samples are not available. Ensure `fit` was called.")
            return self.samples
        raise ValueError("MCMC is not the selected inference method.")

    @property
    def get_params(self):
        if isinstance(self.inference, SVI):
            if self.params is None:
                raise ValueError("SVI parameters are not available. Ensure `fit` was called.")
            return self.params
        raise ValueError("SVI is not the selected inference method.")

    @property
    def get_losses(self):
        if isinstance(self.inference, SVI):
            if self.losses is None:
                raise ValueError("SVI losses are not available. Ensure `fit` was called.")
            return self.losses
        raise ValueError("SVI is not the selected inference method.")

    @property
    def get_stein_result(self):
        if isinstance(self.inference, SteinVI):
            if self.stein_result is None:
                raise ValueError("SteinVI results are not available. Ensure `fit` was called.")
            return self.stein_result
        raise ValueError("SteinVI is not the selected inference method.")

    def save_params(self, file_path):
        """
        Saves trained model parameters to a file.

        :param file_path: str
            Path to save the parameters.
        """
        if isinstance(self.inference, SVI):
            if self.params is None:
                raise ValueError("SVI parameters are not available. Ensure `fit` was called.")
            params_to_save = self.params

        elif isinstance(self.inference, MCMC):
            if self.samples is None:
                raise ValueError("MCMC samples are not available. Ensure `fit` was called.")
            params_to_save = self.samples

        elif isinstance(self.inference, SteinVI):
            if self.stein_result is None:
                raise ValueError("SteinVI results are not available. Ensure `fit` was called.")
            params_to_save = self.stein_result.state

        else:
            raise ValueError("Inference method not initialized. Call `compile` first.")

        with open(file_path, "wb") as f:
            pickle.dump(params_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"✅ Model parameters successfully saved to {file_path}")

    def load_params(self, file_path):
        """
        Loads trained model parameters from a file.

        :param file_path: str
            Path to load the parameters from.
        """
        with open(file_path, "rb") as f:
            loaded_params = pickle.load(f)

        if isinstance(self.inference, SVI):
            self.params = loaded_params

        elif isinstance(self.inference, MCMC):
            self.samples = loaded_params

        elif isinstance(self.inference, SteinVI):
            if self.stein_result is None:
                # Initialize SteinVI with a dummy run if it hasn't been set up yet
                self.stein_result = self.inference.run(
                    jax.random.PRNGKey(0), num_steps=1, progress_bar=False
                )
            self.stein_result.state = loaded_params

        else:
            raise ValueError("Inference method not initialized. Call `compile` first.")

        print(f"✅ Model parameters successfully loaded from {file_path}")
