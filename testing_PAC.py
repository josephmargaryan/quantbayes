from BNN.generalization_bounds import (
    BayesianGeneralizationBounds,
    transform_params,
    transform_params_stein,
)
from BNN.FFT.MCMC_METHOD.models import binary_model
from BNN.FFT.MCMC_METHOD.utils import run_inference, predict_binary
from fake_data import generate_binary_classification_data
import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import numpy as np


def test_mcmc():
    rng_key = jax.random.key(0)
    df = generate_binary_classification_data()
    X, y = df.drop(columns=["target"], axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=24, test_size=0.2
    )

    mcmc = run_inference(binary_model, rng_key, X_train, y_train)
    predictions = predict_binary(mcmc, X_test, binary_model, sample_from="logits")
    probs = jax.nn.sigmoid(predictions)

    layers = [key for key in mcmc.get_samples().keys() if key not in ["logits"]]
    bound = BayesianGeneralizationBounds(len(X_train))
    pac_bound = bound.pac_bayesian_bound(
        predictions=probs,
        y_true=y_test,
        posterior_samples=mcmc.get_samples(),
        prior_mean=0,
        prior_std=1,
        loss_fn=BayesianGeneralizationBounds.binary_log_loss,
        layer_names=layers,
    )
    empirical_risk = bound.compute_empirical_risk(
        probs, y_test, BayesianGeneralizationBounds.binary_log_loss
    )
    mean_posterior, std_posterior = bound.extract_posteriors(mcmc.get_samples(), layers)
    kl_divergence = bound.compute_kl_divergence(mean_posterior, std_posterior, 0, 1)
    confidence_term = bound.compute_confidence_term(kl_divergence)

    print(f"Empirical Risk: {empirical_risk}")
    print(f"KL Divergence: {kl_divergence}")
    print(f"Confidence Term: {confidence_term}")
    print(f"Pac-Bayesian-Bound: {pac_bound}")


def test_SVI():
    from BNN.FFT.SVI_METHOD.models import binary_model
    from BNN.FFT.SVI_METHOD.utils import train_binary, predict_binary

    rng_key = jax.random.key(0)
    df = generate_binary_classification_data()
    X, y = df.drop(columns=["target"], axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=24, test_size=0.2
    )

    svi, params = train_binary(X_train, y_train, binary_model, num_steps=100)
    predictions = predict_binary(svi, params, X_test, sample_from="logits")
    probs = jax.nn.sigmoid(predictions)
    loss = log_loss(np.array(y_test), np.array(probs.mean(axis=0)))
    print(f"Log Loss for mean preds: {loss}")
    rng_key = jax.random.PRNGKey(42)
    posterior_samples = transform_params(params, num_samples=100, rng_key=rng_key)

    layer_names = list(posterior_samples.keys())
    bound = BayesianGeneralizationBounds(len(X_train))
    pac_bound = bound.pac_bayesian_bound(
        predictions=probs,
        y_true=y_test,
        posterior_samples=posterior_samples,
        prior_mean=0,
        prior_std=1,
        loss_fn=BayesianGeneralizationBounds.binary_log_loss,
        layer_names=layer_names,
    )
    mean_posterior, std_posterior = bound.extract_posteriors(
        posterior_samples, layer_names
    )
    empirical_risk = bound.compute_empirical_risk(
        probs, y_test, BayesianGeneralizationBounds.binary_log_loss
    )
    kl_divergence = bound.compute_kl_divergence(
        mean_posterior=mean_posterior,
        std_posterior=std_posterior,
        mean_prior=0,
        std_prior=1,
    )
    confidence_term = bound.compute_confidence_term(kl_divergence)
    print(f"Empirical Risk: {empirical_risk}")
    print(f"KL Divergence: {kl_divergence}")
    print(f"Confidence Term: {confidence_term}")
    print(f"PAC-Bayesian Bound: {pac_bound}")


def test_SteinVI():
    from BNN.FFT.STEIN_VI.models import binary_model
    from BNN.FFT.STEIN_VI.utils import train_binary, predict_binary

    df = generate_binary_classification_data()
    X, y = df.drop(columns=["target"], axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=24, test_size=0.2
    )

    stein, stein_result = train_binary(binary_model, X_train, y_train, 100)
    predictions = predict_binary(
        stein, binary_model, stein_result, X_test, sample_from="logits"
    )
    probs = jax.nn.sigmoid(predictions)
    loss = log_loss(np.array(y_test), np.array(probs.mean(axis=0)))
    print(f"The log loss of the mean posteriors: {loss}")

    posterior_samples = transform_params_stein(stein_result)
    layer_names = list(posterior_samples.keys())

    bound = BayesianGeneralizationBounds(len(X_train))
    pac_bound = bound.pac_bayesian_bound(
        predictions=predictions,
        y_true=y_test,
        posterior_samples=posterior_samples,
        prior_mean=0,
        prior_std=1,
        loss_fn=BayesianGeneralizationBounds.binary_log_loss,
        layer_names=layer_names,
    )

    empirical_risk = bound.compute_empirical_risk(
        probs, y_test, BayesianGeneralizationBounds.binary_log_loss
    )
    mean_posterior, std_posterior = bound.extract_posteriors(
        posterior_samples, layer_names
    )
    kl_divergence = bound.compute_kl_divergence(
        mean_posterior, std_posterior, mean_prior=0, std_prior=1
    )
    confidence_term = bound.compute_confidence_term(kl_divergence)
    print(f"Empirical Risk: {empirical_risk}")
    print(f"KL Divergence: {kl_divergence}")
    print(f"Confidence Term: {confidence_term}")
    print(f"PAC-Bayesian Bound: {pac_bound}")


if __name__ == "__main__":
    test_mcmc()
    test_SVI()
    test_SteinVI()
