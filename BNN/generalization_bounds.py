import jax.numpy as jnp

class BayesianGeneralizationBounds:
    def __init__(self, num_samples, delta=0.05, model_type="gaussian"):
        """
        Initialize the Bayesian Generalization Bounds class.

        Parameters:
        - num_samples: Number of training samples.
        - delta: Confidence level (default=0.05 for 95% confidence).
        - model_type: Type of model ("gaussian" or "hierarchical").
        """
        self.num_samples = num_samples
        self.delta = delta
        self.model_type = model_type

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
                + 2 * jnp.log(std_prior / (std_posterior + 1e-8))
            ).sum()
        )
        return kl_divergence

    @staticmethod
    def compute_kl_divergence_hierarchical(
        mean_posterior,
        std_posterior,
        mean_prior=0,
        std_prior=1,
        alpha_prior=1.0,
        beta_prior=0.1,
    ):
        std_posterior = jnp.clip(std_posterior, 1e-6, None)
        std_prior = jnp.clip(std_prior, 1e-6, None)
        kl_weights = (
            0.5
            * (
                (std_posterior / std_prior) ** 2
                + ((mean_posterior - mean_prior) / std_prior) ** 2
                - 1
                + 2 * jnp.log(std_prior / std_posterior)
            ).sum()
        )
        return kl_weights

    def extract_posteriors(self, posterior_samples):
        """
        Extract posterior mean and standard deviation for weights, biases, and precision
        (if hierarchical priors are used).
        """
        w1_mean = posterior_samples["w_hidden"].mean(axis=0)
        w1_std = posterior_samples["w_hidden"].std(axis=0)

        b1_mean = posterior_samples["b_hidden"].mean(axis=0)
        b1_std = posterior_samples["b_hidden"].std(axis=0)

        w2_mean = posterior_samples["w_out"].mean(axis=0)
        w2_std = posterior_samples["w_out"].std(axis=0)

        b2_mean = posterior_samples["b_out"].mean(axis=0)
        b2_std = posterior_samples["b_out"].std(axis=0)

        mean_posterior = jnp.concatenate(
            [w1_mean.flatten(), b1_mean, w2_mean.flatten(), b2_mean]
        )
        std_posterior = jnp.concatenate(
            [w1_std.flatten(), b1_std, w2_std.flatten(), b2_std]
        )

        if self.model_type == "hierarchical" and "precision" in posterior_samples:
            # Hierarchical model includes precision
            prec_mean = posterior_samples["precision"].mean()
            prec_std = posterior_samples["precision"].std()
        else:
            # Gaussian model does not include precision
            prec_mean = None
            prec_std = None

        return mean_posterior, std_posterior, prec_mean, prec_std

    def compute_confidence_term(self, kl_divergence):
        return jnp.sqrt((kl_divergence + jnp.log(1 / self.delta)) / (2 * self.num_samples))

    def pac_bayesian_bound(self, predictions, y_true, posterior_samples, prior_mean, prior_std, loss_fn):
        """
        Compute the PAC-Bayesian bound for Gaussian or hierarchical models.
        """
        empirical_risk = self.compute_empirical_risk(predictions, y_true, loss_fn)

        mean_posterior, std_posterior, prec_mean, prec_std = self.extract_posteriors(
            posterior_samples
        )

        if self.model_type == "hierarchical":
            kl_divergence = self.compute_kl_divergence_hierarchical(
                mean_posterior, std_posterior, prior_mean, prior_std, 
                prec_mean, prec_std
            )
        else:  # Gaussian prior
            kl_divergence = self.compute_kl_divergence(
                mean_posterior, std_posterior, prior_mean, prior_std
            )

        confidence_term = self.compute_confidence_term(kl_divergence)

        return empirical_risk + confidence_term


    def mutual_information_bound(
        self, predictions, y_true, posterior_samples, prior_mean, prior_std, loss_fn
    ):
        """
        Compute the Mutual Information bound for Gaussian or hierarchical models.
        """
        empirical_risk = self.compute_empirical_risk(predictions, y_true, loss_fn)

        mean_posterior, std_posterior, prec_mean, prec_std = self.extract_posteriors(
            posterior_samples
        )

        if self.model_type == "hierarchical":
            mutual_information = self.compute_kl_divergence_hierarchical(
                mean_posterior, std_posterior, prior_mean, prior_std, 
                prec_mean, prec_std
            )
        else:  # Gaussian prior
            mutual_information = self.compute_kl_divergence(
                mean_posterior, std_posterior, prior_mean, prior_std
            )

        confidence_term = self.compute_confidence_term(mutual_information)

        return empirical_risk + confidence_term

class SVIBayesianGeneralizationBounds(BayesianGeneralizationBounds):
    def extract_posteriors(self, svi, params):
        guide = svi.guide
        posterior_means = {
            name: params[f"{name}_{guide.prefix}_loc"] for name in guide._init_locs
        }
        posterior_stds = {
            name: params[f"{name}_{guide.prefix}_scale"] for name in guide._init_locs
        }

        mean_posterior = jnp.concatenate([v.flatten() for v in posterior_means.values()])
        std_posterior = jnp.concatenate([v.flatten() for v in posterior_stds.values()])

        if "precision" in guide._init_locs:
            prec_mean = posterior_means["precision"]
            prec_std = posterior_stds["precision"]
        else:
            prec_mean = None
            prec_std = None

        return mean_posterior, std_posterior, prec_mean, prec_std

class SteinVIBayesianGeneralizationBounds(BayesianGeneralizationBounds):
    def extract_posteriors(self, particles):
        posterior_means = {}
        posterior_stds = {}
        prec_mean, prec_std = None, None

        for key in particles:
            if key.endswith("_auto_loc"):
                param_name = key.replace("_auto_loc", "")
                mean = particles[key]
                std_key = f"{param_name}_auto_scale"
                std = particles[std_key] if std_key in particles else None

                posterior_means[param_name] = mean
                if std is not None:
                    posterior_stds[param_name] = std

        mean_posterior = jnp.concatenate([v.flatten() for v in posterior_means.values()])
        std_posterior = jnp.concatenate([v.flatten() for v in posterior_stds.values()])

        if "precision_auto_loc" in particles and "precision_auto_scale" in particles:
            prec_mean = particles["precision_auto_loc"]
            prec_std = particles["precision_auto_scale"]

        return mean_posterior, std_posterior, prec_mean, prec_std
