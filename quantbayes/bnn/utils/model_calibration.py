import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import auc, roc_curve

__all__ = [
    "CalibratedBNN",
    "plot_roc_curve",
    "plot_calibration_curve",
    "expected_calibration_error",
    "maximum_calibration_error",
    "multiclass_brier_score",
    "binary_nll",
    "multiclass_nll",
]


class TemperatureScaler:
    """Simple temperature scaling calibration."""

    def __init__(self, init_temp=1.0):
        self.temperature = init_temp

    def fit(self, logits, y_true):
        """
        Fit temperature parameter using validation logits and true labels.
        This example searches over a grid to minimize negative log-likelihood.
        """
        temps = np.linspace(0.5, 2.0, num=50)
        best_temp, best_nll = self.temperature, np.inf
        for t in temps:
            scaled_logits = logits / t
            probs = expit(scaled_logits)
            # Negative log-likelihood for binary classification
            nll = -np.mean(
                y_true * np.log(probs + 1e-8) + (1 - y_true) * np.log(1 - probs + 1e-8)
            )
            if nll < best_nll:
                best_nll = nll
                best_temp = t
        self.temperature = best_temp

    def calibrate(self, logits):
        return logits / self.temperature


class IsotonicCalibrator:
    """A wrapper around IsotonicRegression for calibration."""

    def __init__(self):
        # In binary settings, we calibrate probabilities directly.
        self.ir = IsotonicRegression(out_of_bounds="clip")
        self.fitted = False

    def fit(self, logits, y_true):
        """
        Fit the isotonic regression model.
        We first convert logits to probabilities.
        """
        probs = expit(logits)
        self.ir.fit(probs, y_true)
        self.fitted = True

    def calibrate(self, logits):
        """
        Calibrate new logits using the fitted isotonic model.
        """
        probs = expit(logits)
        if not self.fitted:
            raise RuntimeError("IsotonicCalibrator is not fitted yet.")
        # Return calibrated logits by applying the inverse sigmoid to the calibrated probabilities.
        calibrated_probs = self.ir.predict(probs)
        # Avoid issues with 0 or 1 by clipping.
        calibrated_probs = np.clip(calibrated_probs, 1e-8, 1 - 1e-8)
        # Inverse sigmoid transformation (logit)
        return np.log(calibrated_probs / (1 - calibrated_probs))


class PlattScaler:
    """A simple Platt scaling calibrator (which, for binary tasks, ends up similar to temperature scaling)."""

    def __init__(self):
        self.a = 1.0  # slope
        self.b = 0.0  # intercept

    def fit(self, logits, y_true):
        """
        Fit a logistic regression model to calibrate the logits.
        Here we use a simple grid search to adjust slope and intercept.
        """
        a_range = np.linspace(0.5, 2.0, 20)
        b_range = np.linspace(-1, 1, 20)
        best_a, best_b, best_nll = self.a, self.b, np.inf
        for a in a_range:
            for b in b_range:
                scaled_logits = a * logits + b
                probs = expit(scaled_logits)
                nll = -np.mean(
                    y_true * np.log(probs + 1e-8)
                    + (1 - y_true) * np.log(1 - probs + 1e-8)
                )
                if nll < best_nll:
                    best_nll = nll
                    best_a, best_b = a, b
        self.a, self.b = best_a, best_b

    def calibrate(self, logits):
        return self.a * logits + self.b


def get_calibrator(method: str):
    """
    Factory function to select a calibrator based on the method string.
    """
    if method == "temp":
        return TemperatureScaler()
    elif method == "isotonic":
        return IsotonicCalibrator()
    elif method == "platt":
        return PlattScaler()
    else:
        raise ValueError(f"Unknown calibration method: {method}")


class CalibratedBNN:
    """
    A wrapper that combines a Bayesian Neural Network (BNN) with post-hoc probability calibration.

    This class first trains the underlying BNN to obtain posterior logits, and then applies a
    calibration method to adjust these logits so that the predicted probabilities better reflect
    true outcome frequencies. The calibration method is selected via a string (e.g., "temp",
    "isotonic", or "platt") and is instantiated using the factory function `get_calibrator`.

    Parameters
    ----------
    bayesnet : bnn.Module
        An instance of your Bayesian Neural Network model.
    calibration_method : str, default="temp"
        The calibration method to be used. Options include:
          - "temp": Temperature scaling.
          - "isotonic": Isotonic regression.
          - "platt": Platt scaling.

    Attributes
    ----------
    bayesnet : bnn.Module
        The underlying Bayesian Neural Network model.
    calibrator : object
        The calibration model instance selected based on the `calibration_method` argument.
    """

    def __init__(self, bayesnet, calibration_method: str = "temp"):
        """
        Initialize the CalibratedBayesNet wrapper.

        Parameters
        ----------
        bayesnet : bnn.Module
            The Bayesian neural network model to wrap.
        calibration_method : str, default="temp"
            The calibration method to use. Valid options are "temp", "isotonic", or "platt".
            The corresponding calibrator is created via the `get_calibrator` factory function.
        """
        self.bayesnet = bayesnet
        self.calibrator = get_calibrator(calibration_method)

    def fit(self, X_train, y_train, X_calib, y_calib, rng_key, **kwargs):
        """
        Train the underlying BNN and calibrate its output logits.

        First, the Bayesian network is compiled and fitted on the training data (X_train, y_train)
        using the provided inference parameters. Then, using a separate calibration dataset
        (X_calib, y_calib), the model's predictive logits are generated and converted to a NumPy
        array. These logits are used to fit the calibration model (e.g., temperature scaling,
        isotonic regression, or Platt scaling).

        Parameters
        ----------
        X_train : array-like
            Training input data for the BNN.
        y_train : array-like
            Training targets for the BNN.
        X_calib : array-like
            Calibration input data used to fit the calibration model.
        y_calib : array-like
            Calibration targets used to fit the calibration model.
        rng_key : jax.random.PRNGKey
            Random key used for JAX-based operations.
        **kwargs : dict
            Additional keyword arguments to be passed to the BNN's compile and fit methods
            (e.g., number of warmup steps, number of samples, etc.).

        Returns
        -------
        self : CalibratedBayesNet
            The fitted CalibratedBayesNet instance.
        """
        # Compile and train the Bayesian network
        self.bayesnet.compile(**kwargs)
        self.bayesnet.fit(X_train, y_train, rng_key, **kwargs)

        # Generate logits on the calibration set using the BNN
        calib_preds = self.bayesnet.predict(X_calib, rng_key, posterior="logits")
        # Convert logits from JAX array to NumPy array for calibration processing
        logits = np.array(calib_preds)

        # Fit the calibrator using the calibration dataset
        self.calibrator.fit(logits, np.array(y_calib))
        return self

    def predict_proba(self, X_test, rng_key, num_samples=None):
        """
        Generate calibrated prediction probabilities for the test data.

        This method uses the trained BNN to predict posterior logits for X_test, applies the
        calibration model to adjust these logits, and then converts the calibrated logits into
        probabilities using the sigmoid function. For binary classification, it returns a
        two-column array where each row contains the probabilities for the negative and positive classes.

        Parameters
        ----------
        X_test : array-like
            Test input data for which predictions are to be generated.
        rng_key : jax.random.PRNGKey
            Random key used for prediction sampling.
        num_samples : int, optional
            The number of posterior samples to use during prediction (if applicable).

        Returns
        -------
        probs : ndarray
            A NumPy array of calibrated prediction probabilities with shape
            (2, n_test_samples) or (n_test_samples, 2), depending on how stacking is handled.
        """
        # Get predictive logits from the underlying BNN
        preds = self.bayesnet.predict(
            X_test, rng_key, posterior="logits", num_samples=num_samples
        )
        logits = np.array(preds)
        # Calibrate the logits using the chosen calibration method
        calibrated_logits = self.calibrator.calibrate(logits)
        # Convert calibrated logits to probabilities with the sigmoid function
        probs = expit(calibrated_logits)
        # For binary classification, return a two-column probability array:
        return np.stack([1 - probs, probs], axis=-1)

    def predict(self, X_test, rng_key, num_samples=None):
        """
        Predict class labels for the test data using calibrated probabilities.

        This method calls `predict_proba` to obtain calibrated probabilities and then selects
        the class with the highest probability as the predicted label.

        Parameters
        ----------
        X_test : array-like
            Test input data for which class predictions are to be made.
        rng_key : jax.random.PRNGKey
            Random key used for prediction sampling.
        num_samples : int, optional
            The number of posterior samples to use during prediction (if applicable).

        Returns
        -------
        predictions : ndarray
            A NumPy array of predicted class labels.
        """
        proba = self.predict_proba(X_test, rng_key, num_samples)
        return np.argmax(proba, axis=-1)


def plot_roc_curve(y_true, y_scores, title="ROC Curve"):
    """
    Plot an ROC curve for binary classification.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True binary labels (0 or 1).
    - y_scores: array-like of shape (n_samples,)
        Target scores, which can be probabilities or confidence values.
    - title: str
        Title for the plot (default: "ROC Curve").

    Returns:
    - None. Displays the plot.
    """
    # Calculate false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Calculate AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)  # Diagonal line
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()


def plot_calibration_curve(y_true, y_prob, num_bins=10, plot_type="binary"):
    """
    Plots calibration curves for binary or multiclass classification.

    Parameters:
    - y_true (array-like): True labels.
    - y_prob (array-like): Predicted probabilities.
    - num_bins (int): Number of bins for the calibration curve.
    - plot_type (str): "binary" or "multiclass".

    Returns:
    None (plots are shown).
    """
    if plot_type == "binary":
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=num_bins, strategy="uniform"
        )
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker="o", label="Calibration Curve")
        plt.plot([0, 1], [0, 1], "--", label="Perfect Calibration")
        plt.title("Calibration Curve for Binary Classification")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.legend()
        plt.grid()
        plt.show()

    elif plot_type == "multiclass":
        num_classes = y_prob.shape[1]
        plt.figure(figsize=(12, 6))

        # Plot calibration curve for each class
        for class_idx in range(num_classes):
            prob_true, prob_pred = calibration_curve(
                y_true == class_idx,
                y_prob[:, class_idx],
                n_bins=num_bins,
                strategy="uniform",
            )
            plt.plot(prob_pred, prob_true, marker="o", label=f"Class {class_idx}")

        plt.plot([0, 1], [0, 1], "--", label="Perfect Calibration")
        plt.title("Calibration Curve for Multiclass Classification (Per Class)")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.legend()
        plt.grid()
        plt.show()

        # Plot average calibration curve
        avg_prob_true = np.zeros(num_bins)
        avg_prob_pred = np.zeros(num_bins)
        for class_idx in range(num_classes):
            prob_true, prob_pred = calibration_curve(
                y_true == class_idx,
                y_prob[:, class_idx],
                n_bins=num_bins,
                strategy="uniform",
            )
            avg_prob_true += np.interp(
                np.linspace(0, 1, num_bins), prob_pred, prob_true, left=0, right=0
            )
            avg_prob_pred += np.linspace(0, 1, num_bins)

        avg_prob_true /= num_classes
        avg_prob_pred /= num_classes

        plt.figure(figsize=(8, 6))
        plt.plot(
            avg_prob_pred, avg_prob_true, marker="o", label="Average Calibration Curve"
        )
        plt.plot([0, 1], [0, 1], "--", label="Perfect Calibration")
        plt.title("Average Calibration Curve for Multiclass Classification")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.legend()
        plt.grid()
        plt.show()

    else:
        raise ValueError("Invalid plot_type. Choose either 'binary' or 'multiclass'.")


def multiclass_brier_score(y_true, y_prob):
    """
    Compute the multiclass Brier score.

    Parameters:
      y_true (array-like): True labels (integer encoded).
      y_prob (array-like): Predicted probabilities for each class.

    Returns:
      brier_score (float): The multiclass Brier score.
    """
    # One-hot encode y_true
    n_samples = y_prob.shape[0]
    n_classes = y_prob.shape[1]
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1

    # Compute squared differences
    squared_diff = (y_prob - y_true_onehot) ** 2

    # Average over classes and samples
    brier_score = np.mean(np.sum(squared_diff, axis=1))
    return brier_score


def maximum_calibration_error(y_true, y_prob, num_bins=10):
    """
    Computes the Maximum Calibration Error (MCE) for binary or multiclass classification.

    For each bin, the absolute difference between the empirical accuracy and the mean predicted probability is computed,
    and then the maximum error is returned. For multiclass, this function averages the MCE across classes (one-vs-all).

    Parameters:
      y_true (array-like): True labels.
      y_prob (array-like): Predicted probabilities.
          For multiclass, shape should be (n_samples, n_classes).
      num_bins (int): Number of bins to use for calibration.

    Returns:
      mce (float): The maximum calibration error.
    """
    if y_prob.ndim == 1:
        # Binary classification
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=num_bins, strategy="uniform"
        )
        mce = np.max(np.abs(prob_true - prob_pred))
    else:
        # Multiclass: Compute MCE for each class and average.
        num_classes = y_prob.shape[1]
        mce_total = 0.0
        for class_idx in range(num_classes):
            binary_true = y_true == class_idx
            prob_class = y_prob[:, class_idx]
            prob_true, prob_pred = calibration_curve(
                binary_true, prob_class, n_bins=num_bins, strategy="uniform"
            )
            mce_total += np.max(np.abs(prob_true - prob_pred))
        mce = mce_total / num_classes
    return mce


def expected_calibration_error(y_true, y_prob, num_bins=10):
    """
    Computes the Expected Calibration Error (ECE) for binary or multiclass classification.

    Parameters:
      y_true (array-like): True labels.
      y_prob (array-like): Predicted probabilities.
          For multiclass, shape should be (n_samples, n_classes).
      num_bins (int): Number of bins to use for calibration.

    Returns:
      ece (float): The expected calibration error.
    """

    # Binary classification: y_prob is 1D.
    if y_prob.ndim == 1:
        # Compute calibration curve; both arrays have length equal to number of non-empty bins.
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=num_bins, strategy="uniform"
        )
        # To weight the error by bin frequency, we compute bin counts using the same bin edges.
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges, right=True) - 1
        # Clip negative indices to 0.
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        bin_counts = np.bincount(bin_indices, minlength=num_bins)
        # Filter out bins with no samples.
        valid_bins = bin_counts > 0
        # ECE is the sum over bins of |accuracy - confidence| weighted by the fraction of samples.
        ece = np.sum(
            np.abs(prob_true - prob_pred) * (bin_counts[valid_bins] / len(y_true))
        )
    else:
        # Multiclass classification: average the ECE over each class (one-vs-all).
        num_classes = y_prob.shape[1]
        ece_total = 0.0
        for class_idx in range(num_classes):
            # Create binary labels for the current class.
            binary_true = y_true == class_idx
            # Get predicted probability for the current class.
            prob_class = y_prob[:, class_idx]
            prob_true, prob_pred = calibration_curve(
                binary_true, prob_class, n_bins=num_bins, strategy="uniform"
            )
            bin_edges = np.linspace(0, 1, num_bins + 1)
            bin_indices = np.digitize(prob_class, bin_edges, right=True) - 1
            # Clip negative indices to 0.
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)
            bin_counts = np.bincount(bin_indices, minlength=num_bins)
            valid_bins = bin_counts > 0
            ece_class = np.sum(
                np.abs(prob_true - prob_pred) * (bin_counts[valid_bins] / len(y_true))
            )
            ece_total += ece_class
        ece = ece_total / num_classes
    return ece


def binary_nll(y_true, y_prob, eps=1e-15, normalize=True):
    """
    Compute the Negative Log‑Likelihood (log‑loss) for binary classification.

    Parameters:
      y_true (array-like, shape (n_samples,)): True labels (0 or 1).
      y_prob (array-like, shape (n_samples,)): Predicted probability of class 1.
      eps (float): Small constant to avoid log(0).
      normalize (bool): If True, return mean NLL; else return summed NLL.

    Returns:
      nll (float): Binary log‑loss (mean or sum).
    """
    p = np.clip(y_prob, eps, 1 - eps)
    # elementwise loss
    losses = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    return losses.mean() if normalize else losses.sum()


def multiclass_nll(y_true, y_prob, eps=1e-15, normalize=True):
    """
    Compute the Negative Log‑Likelihood (log‑loss) for multiclass classification.

    Parameters:
      y_true (array-like, shape (n_samples,)): True labels as integers 0..K-1.
      y_prob (array-like, shape (n_samples, n_classes)): Predicted class probabilities.
      eps (float): Small constant to avoid log(0).
      normalize (bool): If True, return mean NLL; else return summed NLL.

    Returns:
      nll (float): Multiclass log‑loss (mean or sum).
    """
    p = np.clip(y_prob, eps, 1 - eps)
    # pick out the probability assigned to the true class for each sample
    idx = np.arange(len(y_true))
    true_class_probs = p[idx, y_true]
    losses = -np.log(true_class_probs)
    return losses.mean() if normalize else losses.sum()
