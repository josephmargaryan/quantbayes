import abc
from typing import Any, Union, Tuple, Callable
import jax.numpy as jnp


class BaseTask(abc.ABC):
    """
    Abstract base class for tasks (regression, binary, multiclass).
    """

    def __init__(self):
        self.fitted = False

    @abc.abstractmethod
    def get_default_model(self) -> Callable:
        """
        Define the default probabilistic model for the task.
        """
        pass

    @abc.abstractmethod
    def fit(self, X_train, y_train, rng_key, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, X_test, rng_key):
        pass

    @abc.abstractmethod
    def visualize(self, X_test, y_test, posteriors, feature_index=None):
        pass
