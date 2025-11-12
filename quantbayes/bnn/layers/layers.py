import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

__all__ = [
    "Linear",
    "GaussianProcessLayer",
]


class Linear:
    """
    A fully connected layer with weights and biases sampled from specified distributions.

    Transforms inputs via a linear operation: `output = X @ weights + biases`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        name: str = "layer",
        weight_prior_fn=lambda shape: dist.Normal(0, 1)
        .expand(shape)
        .to_event(len(shape)),
        bias_prior_fn=lambda shape: dist.Normal(0, 1).expand(shape).to_event(1),
    ):
        """
        Initializes the Linear layer.

        :param in_features: int
            Number of input features.
        :param out_features: int
            Number of output features.
        :param name: str
            Name of the layer for parameter tracking (default: "layer").
        :param weight_prior_fn: function
            A function that takes a shape and returns a NumPyro distribution for the weights.
        :param bias_prior_fn: function
            A function that takes a shape and returns a NumPyro distribution for the biases.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.weight_prior_fn = weight_prior_fn
        self.bias_prior_fn = bias_prior_fn

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the linear transformation on the input.

        :param X: jnp.ndarray
            Input array of shape `(batch_size, in_features)`.
        :returns: jnp.ndarray
            Output array of shape `(batch_size, out_features)`.
        """
        w = numpyro.sample(
            f"{self.name}_w",
            self.weight_prior_fn([self.in_features, self.out_features]),
        )
        b = numpyro.sample(f"{self.name}_b", self.bias_prior_fn([self.out_features]))
        return jnp.dot(X, w) + b


# --- Unified Gaussian Process Layer ---
class GaussianProcessLayer:
    def __init__(
        self,
        input_dim: int,
        kernel_type: str = "rbf",
        name: str = "gp_layer",
        **kernel_kwargs,
    ):
        """
        A unified GP layer that supports multiple kernels.

        Parameters:
          input_dim: int - dimensionality of input features.
          kernel_type: str - one of: "rbf", "spectralmixture", "matern32", "matern52",
                            "periodic", "rq" (rational quadratic), "linear", "poly"
          name: str - parameter name prefix.
          kernel_kwargs: extra parameters passed to the kernel (e.g., Q for spectral mixture, degree for poly, etc.)
        """
        self.input_dim = input_dim
        self.name = name
        self.kernel_type = kernel_type.lower()
        # Choose kernel function
        if self.kernel_type == "rbf":
            self.kernel_fn = self.rbf_kernel
        elif self.kernel_type == "spectralmixture":
            self.kernel_fn = SpectralMixtureKernel(input_dim, **kernel_kwargs)
        elif self.kernel_type == "matern32":
            self.kernel_fn = Matern32Kernel(input_dim, **kernel_kwargs)
        elif self.kernel_type == "matern52":
            self.kernel_fn = Matern52Kernel(input_dim, **kernel_kwargs)
        elif self.kernel_type == "periodic":
            self.kernel_fn = PeriodicKernel(input_dim, **kernel_kwargs)
        elif self.kernel_type == "rq":
            self.kernel_fn = RationalQuadraticKernel(input_dim, **kernel_kwargs)
        elif self.kernel_type == "linear":
            self.kernel_fn = LinearKernel(input_dim, **kernel_kwargs)
        elif self.kernel_type == "poly":
            self.kernel_fn = PolynomialKernel(input_dim, **kernel_kwargs)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def __call__(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        if X2 is None:
            X2 = X
        # Compute the kernel matrix using the chosen kernel function.
        K = self.kernel_fn(X, X2)
        # For full covariance (i.e. when X2 is X), add noise.
        if X is X2:
            # Store the noise parameter in the instance.
            self.noise = numpyro.param(
                f"{self.name}_noise",
                jnp.array(1.0),
                constraint=dist.constraints.positive,
            )
            K = K + self.noise * jnp.eye(X.shape[0]) + 1e-6 * jnp.eye(X.shape[0])
        return K

    # --- RBF Kernel Implementation ---
    def rbf_kernel(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        if X2 is None:
            X2 = X
        # Retrieve kernel parameters
        length_scale = numpyro.param(
            f"{self.name}_length_scale",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        variance = numpyro.param(
            f"{self.name}_variance",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        # Compute squared Euclidean distances.
        X_sq = jnp.sum(X**2, axis=-1, keepdims=True)
        X2_sq = jnp.sum(X2**2, axis=-1, keepdims=True)
        pairwise_sq_dists = X_sq - 2 * jnp.dot(X, X2.T) + X2_sq.T
        return variance * jnp.exp(-0.5 * pairwise_sq_dists / (length_scale**2))


# --- Kernel Variants from Before ---
# Spectral Mixture Kernel:
class SpectralMixtureKernel:
    def __init__(self, input_dim: int, Q: int = 1, name: str = "sm_kernel"):
        self.input_dim = input_dim
        self.Q = Q
        self.name = name

    def __call__(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        if X2 is None:
            X2 = X
        diff = jnp.linalg.norm(X[:, None, :] - X2[None, :, :], axis=-1)
        weights = numpyro.param(
            f"{self.name}_weights",
            jnp.ones(self.Q) / self.Q,
            constraint=dist.constraints.simplex,
        )
        means = numpyro.param(
            f"{self.name}_means", jnp.ones(self.Q), constraint=dist.constraints.positive
        )
        variances = numpyro.param(
            f"{self.name}_variances",
            jnp.ones(self.Q),
            constraint=dist.constraints.positive,
        )
        kernel = 0.0
        for q in range(self.Q):
            wq = weights[q]
            vq = variances[q]
            muq = means[q]
            kernel += (
                wq
                * jnp.exp(-2 * (jnp.pi**2) * (diff**2) * vq)
                * jnp.cos(2 * jnp.pi * diff * muq)
            )
        return kernel


# Matern 3/2 Kernel:
class Matern32Kernel:
    def __init__(self, input_dim: int, name: str = "matern32"):
        self.input_dim = input_dim
        self.name = name

    def __call__(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        if X2 is None:
            X2 = X
        diff = jnp.linalg.norm(X[:, None, :] - X2[None, :, :], axis=-1)
        length_scale = numpyro.param(
            f"{self.name}_length_scale",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        variance = numpyro.param(
            f"{self.name}_variance",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        sqrt3 = jnp.sqrt(3.0)
        return (
            variance
            * (1 + sqrt3 * diff / length_scale)
            * jnp.exp(-sqrt3 * diff / length_scale)
        )


# Matern 5/2 Kernel:
class Matern52Kernel:
    def __init__(self, input_dim: int, name: str = "matern52"):
        self.input_dim = input_dim
        self.name = name

    def __call__(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        if X2 is None:
            X2 = X
        diff = jnp.linalg.norm(X[:, None, :] - X2[None, :, :], axis=-1)
        length_scale = numpyro.param(
            f"{self.name}_length_scale",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        variance = numpyro.param(
            f"{self.name}_variance",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        sqrt5 = jnp.sqrt(5.0)
        return (
            variance
            * (1 + sqrt5 * diff / length_scale + (5 * diff**2) / (3 * length_scale**2))
            * jnp.exp(-sqrt5 * diff / length_scale)
        )


# Periodic Kernel:
class PeriodicKernel:
    def __init__(self, input_dim: int, name: str = "periodic"):
        self.input_dim = input_dim
        self.name = name

    def __call__(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        if X2 is None:
            X2 = X
        diff = jnp.linalg.norm(X[:, None, :] - X2[None, :, :], axis=-1)
        length_scale = numpyro.param(
            f"{self.name}_length_scale",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        variance = numpyro.param(
            f"{self.name}_variance",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        period = numpyro.param(
            f"{self.name}_period", jnp.array(1.0), constraint=dist.constraints.positive
        )
        return variance * jnp.exp(
            -2 * (jnp.sin(jnp.pi * diff / period) ** 2) / (length_scale**2)
        )


# Rational Quadratic Kernel:
class RationalQuadraticKernel:
    def __init__(self, input_dim: int, name: str = "rq"):
        self.input_dim = input_dim
        self.name = name

    def __call__(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        if X2 is None:
            X2 = X
        diff = jnp.linalg.norm(X[:, None, :] - X2[None, :, :], axis=-1)
        length_scale = numpyro.param(
            f"{self.name}_length_scale",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        variance = numpyro.param(
            f"{self.name}_variance",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        alpha = numpyro.param(
            f"{self.name}_alpha", jnp.array(1.0), constraint=dist.constraints.positive
        )
        return variance * (1 + (diff**2) / (2 * alpha * length_scale**2)) ** (-alpha)


# Linear Kernel:
class LinearKernel:
    def __init__(self, input_dim: int, name: str = "linear"):
        self.input_dim = input_dim
        self.name = name

    def __call__(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        if X2 is None:
            X2 = X
        bias = numpyro.param(f"{self.name}_bias", jnp.array(0.0))
        return (X @ X2.T) + bias


# Polynomial Kernel:
class PolynomialKernel:
    def __init__(self, input_dim: int, degree: int = 2, name: str = "poly"):
        self.input_dim = input_dim
        self.degree = degree
        self.name = name

    def __call__(self, X: jnp.ndarray, X2: jnp.ndarray = None) -> jnp.ndarray:
        if X2 is None:
            X2 = X
        gamma = numpyro.param(
            f"{self.name}_gamma", jnp.array(1.0), constraint=dist.constraints.positive
        )
        coef0 = numpyro.param(f"{self.name}_coef0", jnp.array(1.0))
        return (gamma * (X @ X2.T) + coef0) ** self.degree
