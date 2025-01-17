import abc
from typing import Any
import jax


class BaseInference(abc.ABC):
    """
    Abstract base class for inference methods (MCMC, SVI, SteinVI).
    """

    @abc.abstractmethod
    def bayesian_inference(
        self, X_train: jax.Array, y_train: jax.Array, **kwargs
    ) -> Any:
        """
        Perform the inference method-specific training.
        """
        pass

    @abc.abstractmethod
    def retrieve_results(self) -> Any:
        """
        Retrieve inference results.
        """
        pass
