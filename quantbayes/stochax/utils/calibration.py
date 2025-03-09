import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # Sigmoid function
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from quantbayes.bnn.utils import expected_calibration_error, plot_calibration_curve
from quantbayes.fake_data import generate_binary_classification_data
from quantbayes.stochax.tabular import BinaryModel


class CalibratedClassifier:
    """
    A calibrated classifier for Equinox-based binary models.

    This wrapper takes a trained Equinox model and calibrates its logits
    using one of the following methods:
      - "temp": Temperature scaling
      - "platt": Platt scaling (logistic regression calibration)
      - "isotonic": Isotonic regression calibration

    The calibration is performed on a given calibration dataset (X_calib, y_calib)
    and can be used to achieve lower expected calibration error (ECE).
    """

    def __init__(self, base_model, state, method: str = "temp"):
        """
        Parameters
        ----------
        base_model : eqx.Module
            A trained Equinox model that returns raw logits.
        state : Any
            The corresponding state (e.g., for layers like dropout/BN).
        method : str, default "temp"
            Calibration method. Should be one of "temp", "platt", or "isotonic".
        """
        self.base_model = base_model
        self.state = state
        if method not in ("temp", "platt", "isotonic"):
            raise ValueError(f"Unsupported calibration method: {method}")
        self.method = method
        self.calibrator_ = None
        self.fitted = False

    def _get_logits(self, X, key):
        """
        Utility to get raw logits from the underlying model.
        Uses BinaryModel.predict() as defined in your original wrapper.
        """
        # Instantiate a temporary BinaryModel for prediction.
        temp_trainer = BinaryModel()
        logits = temp_trainer.predict(self.base_model, self.state, X, key=key)
        return jnp.array(logits)

    def fit(self, X_calib, y_calib, key=jr.PRNGKey(0)):
        """
        Calibrate the underlying model using the calibration data.

        Parameters
        ----------
        X_calib : array-like
            Calibration features.
        y_calib : array-like
            Calibration labels (0 or 1).
        key : jax.random.PRNGKey, default jr.PRNGKey(0)
            A JAX random key.

        Returns
        -------
        self
        """
        # Get the raw logits from the model on the calibration set.
        logits = self._get_logits(X_calib, key)
        logits_np = np.array(logits)
        y_calib_np = np.array(y_calib)

        eps = 1e-7  # small constant to avoid log(0)

        if self.method == "temp":
            # Temperature scaling: find T such that
            # calibrated probabilities = expit(logits / T)
            def loss(T):
                T = T[0]
                probs = expit(logits_np / T)
                return -np.mean(
                    y_calib_np * np.log(probs + eps)
                    + (1 - y_calib_np) * np.log(1 - probs + eps)
                )

            # Initialize temperature T=1 and constrain T > 0.
            res = minimize(loss, [1.0], bounds=[(1e-6, None)])
            self.calibrator_ = ("temp", res.x[0])
            print(f"[Calibration] Learned temperature: {res.x[0]:.4f}")

        elif self.method == "platt":
            # Platt scaling: fit parameters a and b such that
            # calibrated probabilities = expit(a * logits + b)
            def loss(params):
                a, b = params
                probs = expit(a * logits_np + b)
                return -np.mean(
                    y_calib_np * np.log(probs + eps)
                    + (1 - y_calib_np) * np.log(1 - probs + eps)
                )

            res = minimize(loss, [1.0, 0.0])
            self.calibrator_ = ("platt", res.x[0], res.x[1])
            print(
                f"[Calibration] Learned Platt parameters: a={res.x[0]:.4f}, b={res.x[1]:.4f}"
            )

        elif self.method == "isotonic":
            # Fit isotonic regression to map logits to probabilities.
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(logits_np, y_calib_np)
            self.calibrator_ = ("isotonic", iso)
            print("[Calibration] Isotonic regression fitted.")

        self.fitted = True
        return self

    def predict_proba(self, X, key=jr.PRNGKey(0)):
        """
        Predict calibrated probabilities for the positive and negative class.

        Parameters
        ----------
        X : array-like
            Input features.
        key : jax.random.PRNGKey, default jr.PRNGKey(0)
            A JAX random key.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Calibrated probabilities for the negative and positive class.
        """
        if not self.fitted:
            raise RuntimeError("Calibrator has not been fitted yet. Call fit() first.")

        logits = self._get_logits(X, key)
        logits_np = np.array(logits)

        if self.calibrator_[0] == "temp":
            T = self.calibrator_[1]
            probs = expit(logits_np / T)
        elif self.calibrator_[0] == "platt":
            a, b = self.calibrator_[1], self.calibrator_[2]
            probs = expit(a * logits_np + b)
        elif self.calibrator_[0] == "isotonic":
            iso = self.calibrator_[1]
            probs = iso.predict(logits_np)
        else:
            raise ValueError("Invalid calibrator stored.")

        # Clip the probabilities for numerical stability.
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        # Return a two-column array: probability for class 0 and class 1.
        return np.vstack([1 - probs, probs]).T

    def predict(self, X, key=jr.PRNGKey(0)):
        """
        Predict binary labels based on calibrated probabilities.

        Parameters
        ----------
        X : array-like
            Input features.
        key : jax.random.PRNGKey, default jr.PRNGKey(0)
            A JAX random key.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted binary labels (0 or 1).
        """
        proba = self.predict_proba(X, key)
        # For binary classification, use threshold 0.5.
        return (proba[:, 1] > 0.5).astype(int)


class SimpleBinaryModel(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear

    def __init__(self, input_dim, hidden_dim, key):
        k1, k2, k3 = jr.split(key, 3)
        self.fc1 = eqx.nn.Linear(input_dim, hidden_dim, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=k2)
        self.fc3 = eqx.nn.Linear(hidden_dim, 1, key=k3)

    def __call__(self, x, state=None, *, key=None):
        x = jax.nn.relu(self.fc1(x))
        x = jax.nn.relu(self.fc2(x))
        logits = self.fc3(x).squeeze()  # raw logits
        return logits, state


if __name__ == "__main__":
    # Generate fake binary classification data.
    df = generate_binary_classification_data(n_categorical=16, n_continuous=12)
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)

    # Split data: training (60%), calibration (20%), testing (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        np.array(X), np.array(y), test_size=0.4, random_state=42
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    X_train, X_calib, X_test = jnp.array(X_train), jnp.array(X_calib), jnp.array(X_test)
    y_train, y_calib, y_test = jnp.array(y_train), jnp.array(y_calib), jnp.array(y_test)

    # Instantiate a simple model and train it using BinaryModel.
    key_model = jr.PRNGKey(1)
    model = SimpleBinaryModel(input_dim=X.shape[1], hidden_dim=32, key=key_model)
    state = None  # no state since we are not using BN/Dropout in this simple model.
    trainer = BinaryModel(lr=1e-2)

    print("Training the uncalibrated model...")
    model, state = trainer.fit(
        model,
        state,
        X_train,
        y_train,
        X_test,
        y_test,
        num_epochs=100,
        patience=10,
        key=jr.PRNGKey(123),
    )

    # Get predictions from the uncalibrated model.
    logits = trainer.predict(model, state, X_test, key=jr.PRNGKey(35))
    # Convert raw logits to probabilities using sigmoid.
    probs_uncalibrated = jax.nn.sigmoid(logits)
    probs_uncalibrated = np.array(probs_uncalibrated)

    # Plot calibration curve and compute ECE for the uncalibrated model.
    plot_calibration_curve(np.array(y_test), probs_uncalibrated)
    ece_uncalibrated = expected_calibration_error(np.array(y_test), probs_uncalibrated)
    print(f"ECE of uncalibrated model: {ece_uncalibrated:.4f}")

    # Now, calibrate the model using EquinoxCalibratedClassifier.
    # Here we use the calibration split.
    calibrator = CalibratedClassifier(base_model=model, state=state, method="isotonic")
    calibrator.fit(X_calib, y_calib, key=jr.PRNGKey(42))

    # Get calibrated probabilities on the test set.
    calibrated_probs = calibrator.predict_proba(X_test, key=jr.PRNGKey(123))
    # For binary classification, our calibrated wrapper returns two columns:
    # column 1 is the probability for the positive class.
    calibrated_probs_pos = calibrated_probs[:, 1]

    # Plot calibration curve and compute ECE for the calibrated model.
    plot_calibration_curve(np.array(y_test), calibrated_probs_pos)
    ece_calibrated = expected_calibration_error(np.array(y_test), calibrated_probs_pos)
    print(f"ECE of calibrated model: {ece_calibrated:.4f}")
