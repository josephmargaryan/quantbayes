import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from quantbayes.bnn.layers import TransformerEncoder, Linear

class TransformerSequenceClassifier:
    """
    Minimal example of a transformer-based sequence classifier. 
    Uses a single TransformerEncoder block for demonstration.
    """

    def __init__(self, vocab_size, embed_dim=32, num_heads=2, hidden_dim=64, num_classes=2, name="transformer_seq_classifier"):
        """
        :param vocab_size: int, size of the vocabulary (if it's a text problem).
                           If your input is already embeddings, you can skip embedding inside this class.
        :param embed_dim: int, dimension of token embeddings + the transformer's expected input
        :param num_heads: int, number of attention heads
        :param hidden_dim: int, dimension of feedforward
        :param num_classes: int, number of output classes
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.name = name

        self.transformer_block = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            name=f"{name}_encoder_block"
        )
        # We'll do a final classification linear
        self.classifier = Linear(embed_dim, num_classes, name=f"{name}_classifier")

    def __call__(self, input_tokens, mask=None):
        """
        input_tokens: jnp array of shape (batch_size, seq_len) with integer tokens
        mask: optional jnp array of shape (batch_size, seq_len, seq_len) for attention masking

        returns: logits shape (batch_size, num_classes)
        """
        batch_size, seq_len = input_tokens.shape

        # 1) Embedding each token. 
        #    We'll sample an embedding matrix from a Normal prior for demonstration:
        #    shape = (vocab_size, embed_dim).
        embed_matrix = numpyro.sample(
            f"{self.name}_embedding",
            dist.Normal(0, 1).expand([self.vocab_size, self.embed_dim])
        )

        # Gather embeddings
        embedded = jnp.take(embed_matrix, input_tokens, axis=0)  
        # shape = (batch_size, seq_len, embed_dim)

        # 2) Pass through a single TransformerEncoder block
        transformed = self.transformer_block(embedded, mask)  
        # shape = (batch_size, seq_len, embed_dim)

        # 3) Let's take the last position's hidden as a representation
        last_hidden = transformed[:, -1, :]  # shape = (batch_size, embed_dim)

        # 4) Classifier
        logits = self.classifier(last_hidden)  # shape = (batch_size, num_classes)
        return logits
