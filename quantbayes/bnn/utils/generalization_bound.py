import jax
import jax.numpy as jnp


def extract_posterior_samples(
    inference_type, posterior_samples, num_samples=100, rng_key=jax.random.PRNGKey(0)
):
    """
    Extracts posterior samples based on the inference type.

    Args:
        inference_type (str): Type of inference ("mcmc", "svi", "steinvi").
        posterior_samples (dict): Posterior samples or parameters from the model.
        num_samples (int): Number of samples to draw for SVI.
        rng_key (jax.random.PRNGKey): RNG key for sampling.

    Returns:
        dict: Transformed posterior samples.
    """
    if inference_type == "mcmc":
        return posterior_samples

    elif inference_type == "svi":
        # Transform SVI params into posterior samples
        transformed_samples = {}
        for key in posterior_samples.keys():
            if key.endswith("_auto_loc"):
                base_name = key.replace("_auto_loc", "")
                mean = posterior_samples[key]
                stddev = posterior_samples[f"{base_name}_auto_scale"]
                samples = (
                    jax.random.normal(rng_key, shape=(num_samples,) + mean.shape)
                    * stddev
                    + mean
                )
                transformed_samples[base_name] = samples
        return transformed_samples

    elif inference_type == "steinvi":
        # SteinVI posterior samples are already grouped by layer
        return {
            layer_name: param_value
            for layer_name, param_value in posterior_samples.params.items()
            if param_value.ndim >= 2  # Exclude scalars or single values
        }

    else:
        raise ValueError(f"Unsupported inference type: {inference_type}")


class BayesianAnalysis:
    def __init__(
        self,
        num_samples,
        delta=0.05,
        task_type="regression",
        inference_type=None,
        posterior_samples=None,
    ):
        """
        Initialize the BayesianAnalysis class.

        Parameters:
        - num_samples: The number of samples (data points).
        - delta: Confidence level (default=0.05 for 95% confidence).
        - task_type: Type of task (regression, binary, or multiclass).
        - inference_type: Type of inference used ("mcmc", "svi", or "steinvi").
        - posterior_samples: Posterior samples or parameters returned by the inference method.
        """
        self.delta = delta
        self.task_type = task_type.lower()
        self.num_samples = num_samples
        self.inference_type = inference_type

        # Standardize posterior samples
        self.posterior_samples = extract_posterior_samples(
            inference_type=self.inference_type,
            posterior_samples=posterior_samples,
            num_samples=100,
            rng_key=jax.random.PRNGKey(0),
        )

        # Automatically extract layer names
        self.layer_names = self.extract_layer_names()

    def get_loss_function(self):
        """
        Retrieve the appropriate loss function based on the task type.
        """
        if self.task_type == "regression":
            return self.regression_mse
        elif self.task_type in ["binary", "multiclass"]:
            return self.classification_loss
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    @staticmethod
    def regression_mse(pred_values, true_values):
        return jnp.mean((pred_values - true_values) ** 2)

    @staticmethod
    def classification_loss(pred_probs, true_labels):
        """
        Compute the multiclass log-loss for posterior samples.
        """
        from sklearn.metrics import log_loss

        losses_per_sample = jnp.array(
            [
                log_loss(jnp.array(true_labels), jnp.array(pred_probs[i]))
                for i in range(pred_probs.shape[0])
            ]
        )
        return losses_per_sample.mean()

    @staticmethod
    def compute_empirical_risk(predictions, y_true, loss_fn):
        return loss_fn(predictions, y_true)

    @staticmethod
    def compute_kl_divergence(
        mean_posterior, std_posterior, mean_prior=0, std_prior=1, eps=1e-6
    ):
        """
        Compute the KL divergence between two multivariate Gaussian distributions with diagonal covariance.
        """
        std_posterior = jnp.clip(std_posterior, eps, None)
        std_prior = jnp.clip(std_prior, eps, None)

        kl_divergence = (
            0.5
            * (
                (std_posterior / std_prior) ** 2
                + ((mean_posterior - mean_prior) / std_prior) ** 2
                - 1
                + 2 * jnp.log(std_prior / std_posterior)
            ).sum()
        )
        return kl_divergence

    def extract_layer_names(self):
        """
        Automatically extract layer names based on the inference type.
        """
        if self.inference_type in ["mcmc", "svi"]:
            return [
                key
                for key in self.posterior_samples.keys()
                if not key.startswith(("logits",))
            ]
        elif self.inference_type == "steinvi":
            return list(self.posterior_samples.keys())
        else:
            raise ValueError(f"Unsupported inference type: {self.inference_type}")

    def extract_posteriors(self):
        """
        Extract posterior means and standard deviations for each layer.
        """
        means = []
        stds = []

        for layer in self.layer_names:
            if layer in self.posterior_samples:
                param_values = self.posterior_samples[layer]
                means.append(param_values.mean(axis=0).flatten())
                stds.append(param_values.std(axis=0).flatten())
            else:
                raise ValueError(f"Layer '{layer}' not found in posterior samples.")

        mean_posterior = jnp.concatenate(means) if means else jnp.array([])
        std_posterior = jnp.concatenate(stds) if stds else jnp.array([])
        return mean_posterior, std_posterior

    def compute_confidence_term(self, kl_divergence):
        return jnp.sqrt(
            (kl_divergence + jnp.log(1 / self.delta)) / (2 * self.num_samples)
        )

    def compute_pac_bayesian_bound(
        self,
        predictions: jax.Array,
        y_true: jax.Array,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
    ) -> float:
        """
        Computes the PAC-Bayesian Bound.

        :param predictions, 2D/3D array of full posteriors (probs)
        :param y_true, jax.Array of ground truth
        """
        loss_fn = self.get_loss_function()
        empirical_risk = self.compute_empirical_risk(predictions, y_true, loss_fn)

        mean_posterior, std_posterior = self.extract_posteriors()
        kl_divergence = self.compute_kl_divergence(
            mean_posterior, std_posterior, prior_mean, prior_std
        )
        confidence_term = self.compute_confidence_term(kl_divergence)

        print(f"Empirical risk: {empirical_risk:.3f}")
        print(f"Confidence term: {confidence_term:.3f}")
        print(f"KL-Divergence: {kl_divergence:.3f}")
        print(f"Pac-Bayes Bound: {empirical_risk + confidence_term:.3f}")

        return empirical_risk + confidence_term

    def compute_mutual_information_bound(self, prior_mean=0, prior_std=1):
        """
        Compute the mutual information bound.
        """
        mean_posterior, std_posterior = self.extract_posteriors()
        kl_divergence = self.compute_kl_divergence(
            mean_posterior, std_posterior, prior_mean, prior_std
        )
        return kl_divergence / self.num_samples

    def compute_bernstein_pac_bayes_bound(
        self,
        predictions: jax.Array,
        y_true: jax.Array,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        C: float = 1.0,
        D: float = 1.0,
    ) -> float:
        """
        Computes a Bernstein-PAC Bayesian Bound.

        :param predictions: Array of predictions (could be posterior samples).
        :param y_true: Ground truth labels.
        :param prior_mean: Mean of the Gaussian prior.
        :param prior_std: Standard deviation of the Gaussian prior.
        :param C: Constant in the confidence term.
        :param D: Constant multiplier for the linear term.
        :return: Bernstein-PAC Bayesian bound value.
        """
        loss_fn = self.get_loss_function()
        # Compute the empirical risk (mean loss over the dataset)
        empirical_risk = self.compute_empirical_risk(predictions, y_true, loss_fn)

        # Compute losses per sample to estimate variance.
        # This requires that `loss_fn` returns a scalar loss for a single sample.
        losses = self.compute_losses_per_sample(predictions, y_true, loss_fn)
        empirical_variance = jnp.var(losses, ddof=1)

        # Compute KL divergence as before:
        mean_posterior, std_posterior = self.extract_posteriors()
        kl_divergence = self.compute_kl_divergence(
            mean_posterior, std_posterior, prior_mean, prior_std
        )

        # Compute the complexity term
        complexity = kl_divergence + jnp.log(C / self.delta)
        # Bernstein type terms: variance term & linear term in complexity
        variance_term = jnp.sqrt(
            (2 * empirical_variance * complexity) / self.num_samples
        )
        linear_term = (D * complexity) / self.num_samples

        # Bernstein-PAC Bayes bound
        bernstein_bound = empirical_risk + variance_term + linear_term

        print(f"Empirical risk: {empirical_risk:.3f}")
        print(f"Empirical variance: {empirical_variance:.3f}")
        print(f"KL-Divergence: {kl_divergence:.3f}")
        print(f"Variance term: {variance_term:.3f}")
        print(f"Linear term: {linear_term:.3f}")
        print(f"Bernstein-PAC Bayes Bound: {bernstein_bound:.3f}")

        return bernstein_bound
