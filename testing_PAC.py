########### Test #########
from BNN.DENSE.MCMC_METHOD.models import multiclass_model
from BNN.DENSE.MCMC_METHOD.utils import run_inference, predict_multiclass
from BNN.FFT.MCMC_METHOD.utils import visualize_multiclass
from BNN.DENSE.MCMC_METHOD.fake_data import generate_multiclass_classification_data
from calibration_uncertainty.pac_bayes import (
    multiclass_log_loss,
    compute_confidence_term,
    compute_empirical_risk,
    compute_kl_divergence,
    extract_posteriors,
)
from calibration_uncertainty.uncertainty import (
    plot_calibration_curve,
    expected_calibration_error,
    compute_entropy_multiclass,
    compute_mutual_information_multiclass,
    visualize_entropy_and_mi_with_average,
)
import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split

rng_key = jax.random.key(2)
df = generate_multiclass_classification_data()
X, y = df.drop(columns=["target"], axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=45, test_size=0.2
)

mcmc = run_inference(multiclass_model, rng_key, X_train, y_train, 100, 50)
predictions = predict_multiclass(mcmc, X_test, multiclass_model, sample_from="logits")
probabilities = jax.nn.softmax(predictions, axis=-1)
final = probabilities.mean(axis=0)

empirical_risk = compute_empirical_risk(probabilities, y_test, multiclass_log_loss)

# Compute posterior mean and standard deviation for KL divergence
mean_preds = predictions.mean(axis=0)
std_preds = predictions.std(axis=0)

mean_posterior, std_posterior, _, _ = extract_posteriors(mcmc.get_samples())
kl_divergence = compute_kl_divergence(mean_posterior, std_posterior)

# Compute confidence term and PAC-Bayesian bound
confidence_term = compute_confidence_term(kl_divergence, len(X_train))
pac_bound = empirical_risk + confidence_term

print(f"Empirical Risk: {empirical_risk}")
print(f"KL Divergence: {kl_divergence}")
print(f"PAC-Bayesian Confidence Term: {confidence_term}")
print(f"PAC-Bayesian Bound: {pac_bound}")

ECE = expected_calibration_error(y_test, final)
print(f"ECE: {ECE}")
entropy = compute_entropy_multiclass(probabilities).mean(axis=0)
MI, predictive_entropy = compute_mutual_information_multiclass(probabilities)
visualize_entropy_and_mi_with_average(MI, predictive_entropy, entropy)

plot_calibration_curve(y_test, final, plot_type="multiclass", num_bins=25)
visualize_multiclass(X_test, y_test, mcmc, predict_multiclass, multiclass_model)
