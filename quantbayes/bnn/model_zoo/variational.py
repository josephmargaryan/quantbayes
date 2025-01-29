from quantbayes.bnn.layers import VariationalLayer
import jax


class VariationalMLP:
    """
    A simple 2-layer MLP that uses `VariationalLayer` for Bayesian weights.
    We do X -> hidden -> output for regression.
    """

    def __init__(self, input_dim, hidden_dim=32, output_dim=1, name="variational_mlp"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.name = name

        self.layer1 = VariationalLayer(input_dim, hidden_dim, name=f"{name}_layer1")
        self.layer2 = VariationalLayer(hidden_dim, output_dim, name=f"{name}_layer2")

    def __call__(self, X):
        """
        X: shape (batch_size, input_dim)
        returns: shape (batch_size, output_dim)
        """
        h = jax.nn.relu(self.layer1(X))
        out = self.layer2(h)
        return out
