import abc
from typing import Optional, Union, Tuple
import jax.numpy as jnp


class BaseInference(abc.ABC):
    """
    Abstract base class for probabilistic models using different inference methods.
    """

    def __init__(self):
        self.fitted = False

    @abc.abstractmethod
    def fit(self, X_train: jnp.ndarray, y_train: jnp.ndarray) -> Union[object, Tuple]:
        """
        Fit the model to training data.
        """
        pass

    @abc.abstractmethod
    def predict(self, X_test: jnp.ndarray) -> jnp.ndarray:
        """
        Predict on new test data.
        """
        pass

    @abc.abstractmethod
    def visualize(self, *args, **kwargs):
        """
        Visualize predictions and uncertainties.
        """
        pass


class BaseMCMC(BaseInference):
    """
    Abstract base class for MCMC-based inference.
    """

    @abc.abstractmethod
    def fit(self, X_train: jnp.ndarray, y_train: jnp.ndarray) -> object:
        """
        Fit using MCMC and return the MCMC object.
        """
        pass


class BaseSVI(BaseInference):
    """
    Abstract base class for SVI-based inference.
    """

    @abc.abstractmethod
    def fit(self, X_train: jnp.ndarray, y_train: jnp.ndarray) -> Tuple[object, dict]:
        """
        Fit using SVI and return the guide and parameters.
        """
        pass


class BaseSteinVI(BaseInference):
    """
    Abstract base class for SteinVI-based inference.
    """

    @abc.abstractmethod
    def fit(self, X_train: jnp.ndarray, y_train: jnp.ndarray) -> Tuple[object, object]:
        """
        Fit using SteinVI and return the Stein object and result.
        """
        pass
