import jax.numpy as jnp
import jax


class BayesianGeneralizationBounds:
    def __init__(self, num_samples, delta=0.05):
        """
        Initialize the Bayesian Generalization Bounds class.

        Parameters:
        - num_samples: Number of training samples.
        - delta: Confidence level (default=0.05 for 95% confidence).
        """
        self.num_samples = num_samples
        self.delta = delta

    @staticmethod
    def multiclass_log_loss(pred_probs, true_labels):
        pred_probs = jnp.clip(pred_probs, 1e-7, 1 - 1e-7)
        log_probs = jnp.log(pred_probs)
        return -jnp.mean(log_probs[jnp.arange(len(true_labels)), true_labels])

    @staticmethod
    def binary_log_loss(pred_probs, true_labels):
        pred_probs = jnp.clip(pred_probs, 1e-7, 1 - 1e-7)
        return -jnp.mean(
            true_labels * jnp.log(pred_probs)
            + (1 - true_labels) * jnp.log(1 - pred_probs)
        )

    @staticmethod
    def regression_mse(pred_values, true_values):
        return jnp.mean((pred_values - true_values) ** 2)

    @staticmethod
    def compute_empirical_risk(predictions, y_true, loss_fn):
        risks = jnp.array([loss_fn(pred, y_true) for pred in predictions])
        return risks.mean()

    @staticmethod
    def compute_kl_divergence(mean_posterior, std_posterior, mean_prior=0, std_prior=1):
        std_posterior = jnp.clip(std_posterior, 1e-6, None)
        std_prior = jnp.clip(std_prior, 1e-6, None)
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

    def extract_posteriors(self, posterior_samples, layer_names):
        """
        Extract posterior mean and standard deviation for specified layers.

        Parameters:
        - posterior_samples: Dictionary containing posterior samples.
        - layer_names: List of layer names to extract.

        Returns:
        - mean_posterior: Concatenated means of specified layers.
        - std_posterior: Concatenated standard deviations of specified layers.
        """
        means, stds = [], []
        for layer in layer_names:
            if layer in posterior_samples:
                param_value = posterior_samples[layer]
                if param_value.ndim >= 2:  # For weights or 2D+ structures
                    means.append(param_value.mean(axis=0).flatten())
                    stds.append(param_value.std(axis=0).flatten())
                elif param_value.ndim == 1:  # For biases or 1D structures
                    means.append(param_value.mean().reshape(-1))
                    stds.append(param_value.std().reshape(-1))
                else:
                    raise ValueError(
                        f"Unsupported parameter shape: {param_value.shape}"
                    )
            else:
                raise ValueError(f"Layer '{layer}' not found in posterior samples.")

        mean_posterior = jnp.concatenate(means) if means else jnp.array([])
        std_posterior = jnp.concatenate(stds) if stds else jnp.array([])
        return mean_posterior, std_posterior

    def compute_confidence_term(self, kl_divergence):
        return jnp.sqrt(
            (kl_divergence + jnp.log(1 / self.delta)) / (2 * self.num_samples)
        )

    def pac_bayesian_bound(
        self,
        predictions,
        y_true,
        posterior_samples,
        prior_mean,
        prior_std,
        loss_fn,
        layer_names,
    ):
        """
        Compute the PAC-Bayesian bound.

        Parameters:
        - predictions: Predictions as probabilities from the model.
        - y_true: True labels.
        - posterior_samples: Dictionary containing posterior samples.
        - prior_mean: Prior mean.
        - prior_std: Prior standard deviation.
        - loss_fn: Loss function to use.
        - layer_names: List of layer names to extract.

        Returns:
        - PAC-Bayesian bound.
        """
        empirical_risk = self.compute_empirical_risk(predictions, y_true, loss_fn)

        mean_posterior, std_posterior = self.extract_posteriors(
            posterior_samples, layer_names
        )

        kl_divergence = self.compute_kl_divergence(
            mean_posterior, std_posterior, prior_mean, prior_std
        )

        confidence_term = self.compute_confidence_term(kl_divergence)

        return empirical_risk + confidence_term

    def mutual_information_bound(
        self, posterior_samples, layer_names, prior_mean=0, prior_std=1
    ):
        """
        Compute the mutual information bound.

        Parameters:
        - posterior_samples: Dictionary containing posterior samples.
        - layer_names: List of layer names to extract.
        - prior_mean: Prior mean.
        - prior_std: Prior standard deviation.

        Returns:
        - Mutual information bound.
        """
        mean_posterior, std_posterior = self.extract_posteriors(
            posterior_samples, layer_names
        )

        kl_divergence = self.compute_kl_divergence(
            mean_posterior, std_posterior, prior_mean, prior_std
        )

        mutual_info = kl_divergence / self.num_samples

        return mutual_info


# Example usage:
# layer_names = ["w_hidden", "b_hidden", "w_out", "b_out"]
############## Under construction #################


# For SVI
def transform_params(params, num_samples=100, rng_key=jax.random.PRNGKey(0)):
    """
    Transform SVI params into posterior samples grouped by layer.

    Parameters:
    - params: Dictionary of variational parameters (keys ending with '_auto_loc' and '_auto_scale').
    - num_samples: Number of samples to generate.
    - rng_key: JAX random key.

    Returns:
    - posterior_samples: Dictionary containing posterior samples for each layer.
    """
    posterior_samples = {}
    for key in params.keys():
        if key.endswith("_auto_loc"):
            # Extract the base layer name (without '_auto_loc' or '_auto_scale')
            base_name = key.replace("_auto_loc", "")

            # Extract mean and stddev
            mean = params[key]
            stddev = params[f"{base_name}_auto_scale"]

            # Generate samples from the posterior
            samples = (
                jax.random.normal(rng_key, shape=(num_samples,) + mean.shape) * stddev
                + mean
            )
            posterior_samples[base_name] = samples

    return posterior_samples


def transform_params_stein(stein_result):
    """
    Transform SteinVI results into posterior samples grouped by layer.

    Parameters:
    - stein_result: A SteinVI result object containing particles.

    Returns:
    - posterior_samples: Dictionary containing particles for each layer.
    """
    posterior_samples = {}
    for layer_name, param_value in stein_result.params.items():
        # Check if param_value is a valid particle representation
        if param_value.ndim >= 2:  # Ensure at least (num_particles, ...)
            posterior_samples[layer_name] = param_value
        else:
            print(f"Skipping parameter '{layer_name}' with shape {param_value.shape}")
    return posterior_samples
