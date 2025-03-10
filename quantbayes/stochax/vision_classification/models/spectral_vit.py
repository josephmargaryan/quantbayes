import jax.random as jr
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from quantbayes.stochax.layers import JVPCirculantProcess
from quantbayes.stochax.vision_classification.models.vit import PatchEmbedding, AttentionBlock

class SpectralVisionTransformer(eqx.Module):
    # Same patch_embedding, positional_embedding, cls_token, etc.
    patch_embedding: PatchEmbedding
    positional_embedding: jnp.ndarray
    cls_token: jnp.ndarray
    attention_blocks: list[AttentionBlock]
    dropout: eqx.nn.Dropout
    spectral_mlp: list[JVPCirculantProcess]  # List of JVPCirculantProcess layers
    final_linear: eqx.nn.Linear  # Final projection to num_classes
    num_layers: int

    def __init__(self, embedding_dim: int, hidden_dim: int, num_heads: int,
                 num_layers: int, dropout_rate: float, patch_size: int,
                 num_patches: int, num_classes: int, key: PRNGKeyArray,
                 channels: int, spectral_padded_dim: int = None, alpha: float = 1.0, K: int = None):
        # Create keys.
        key1, key2, key3, key4, key5, key6 = jr.split(key, 6)
        self.patch_embedding = PatchEmbedding(channels, embedding_dim, patch_size, key1)
        self.positional_embedding = jr.normal(key2, (num_patches + 1, embedding_dim))
        self.cls_token = jr.normal(key3, (1, embedding_dim))
        self.num_layers = num_layers
        # Use same attention blocks.
        self.attention_blocks = [
            AttentionBlock(embedding_dim, hidden_dim, num_heads, dropout_rate, key4)
            for _ in range(num_layers)
        ]
        self.dropout = eqx.nn.Dropout(dropout_rate)
        
        # Define spectral MLP layers replacing traditional dense layers.
        # For example, two spectral layers.
        spectral_padded_dim = spectral_padded_dim if spectral_padded_dim is not None else hidden_dim
        sp_key1, sp_key2 = jr.split(key5)
        self.spectral_mlp = [
            JVPCirculantProcess(in_features=embedding_dim, padded_dim=spectral_padded_dim, alpha=alpha, K=K, key=sp_key1),
            JVPCirculantProcess(in_features=spectral_padded_dim, padded_dim=spectral_padded_dim, alpha=alpha, K=K, key=sp_key2)
        ]
        # Final projection to num_classes (a standard dense layer).
        self.final_linear = eqx.nn.Linear(spectral_padded_dim, num_classes, key=key6)

    def __call__(self, x: Float[Array, "channels height width"], key: PRNGKeyArray, state):
        x = self.patch_embedding(x)
        x = jnp.concatenate((self.cls_token, x), axis=0)
        x += self.positional_embedding[:x.shape[0]]
        keys = jr.split(key, self.num_layers + 1)
        x = self.dropout(x, key=keys[0])
        for block, block_key in zip(self.attention_blocks, keys[1:]):
            x = block(x, key=block_key)
        x = x[0]  # CLS token
        # Pass through spectral MLP layers.
        for layer in self.spectral_mlp:
            x = layer(x)
        logits = self.final_linear(x)
        return logits, state
    
# Quick test
def test_spectral_vit():
    key = jr.PRNGKey(0)
    # Dummy hyperparameters.
    channels = 3
    height, width = 32, 32
    patch_size = 4
    num_patches = (height // patch_size) * (width // patch_size)
    embedding_dim = 64
    hidden_dim = 128
    num_heads = 4
    num_layers = 2
    dropout_rate = 0.1
    num_classes = 10
    spectral_padded_dim = 128  # Could be different from hidden_dim.
    
    # Instantiate the model.
    model = SpectralVisionTransformer(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        patch_size=patch_size,
        num_patches=num_patches,
        num_classes=num_classes,
        key=key,
        channels=channels,
        spectral_padded_dim=spectral_padded_dim
    )
    
    # Create a dummy input image.
    dummy_input = jr.normal(key, (channels, height, width))
    
    # Dummy state (if your model uses any state; otherwise, pass None)
    state = None
    
    # Run a forward pass.
    logits, state_out = model(dummy_input, key, state)
    
    print("Logits shape:", logits.shape)
    # Optionally, test visualization functions:
    # For example, visualize the Fourier coefficients from the first spectral layer.
    fft_coeffs = model.spectral_mlp[0].get_fourier_coeffs()
    print("First spectral layer Fourier coefficients shape:", fft_coeffs.shape)

if __name__ == "__main__":
    test_spectral_vit()