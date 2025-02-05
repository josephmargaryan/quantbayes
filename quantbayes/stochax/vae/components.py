import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt 
import numpy as np 
import optax

from quantbayes.stochax.vae.base import BaseDecoder, BaseEncoder, BaseVAE

#############################
# 1. Standard MLP Versions  #
#############################

class MLPEncoder(eqx.Module, BaseEncoder):
    """
    A standard MLP encoder that outputs 2*latent_dim values (interpreted as μ and log(σ²))
    for each input.
    """
    net: eqx.nn.MLP
    latent_dim: int

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, *, key):
        self.latent_dim = latent_dim
        self.net = eqx.nn.MLP(
            in_size=input_dim,
            out_size=2 * latent_dim,
            width_size=hidden_dim,
            depth=2,
            key=key
        )

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x assumed to have shape (batch, input_dim)
        out = jax.vmap(self.net)(x)
        mu, logvar = jnp.split(out, 2, axis=-1)
        return mu, logvar


class MLPDecoder(eqx.Module, BaseDecoder):
    """
    A standard MLP decoder that maps a latent variable z to a reconstruction.
    """
    net: eqx.nn.MLP

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, *, key):
        self.net = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=output_dim,
            width_size=hidden_dim,
            depth=2,
            key=key
        )

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        # z assumed to have shape (batch, latent_dim)
        return jax.vmap(self.net)(z)

#################################
# 2. CNN-Based Versions         #
#################################
# These are designed for image data.
# Note: The Conv2d and ConvTranspose2d modules are assumed to be available from equinox.nn.
# You may need to adjust input shapes (e.g. NHWC vs. NCHW) as desired.

class CNNEncoder(eqx.Module, BaseEncoder):
    """
    A convolutional encoder for image data.
    It applies two Conv2d layers, flattens the result, and then passes through an MLP.
    Assumes inputs of shape (batch, H, W, C).
    """
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    mlp: eqx.nn.MLP
    latent_dim: int
    image_size: int  # the height (and width) of the input image

    def __init__(self, input_channels: int, image_size: int, hidden_channels: int, 
                 latent_dim: int, *, key):
        keys = jax.random.split(key, 3)
        # Create two convolutional layers.
        self.conv1 = eqx.nn.Conv2d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,
            padding="SAME",
            key=keys[0]
        )
        self.conv2 = eqx.nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,
            padding="SAME",
            key=keys[1]
        )
        # After two strided convs, image dimensions reduce by ~factor 4.
        flat_dim = (image_size // 4) * (image_size // 4) * hidden_channels
        self.mlp = eqx.nn.MLP(
            in_size=flat_dim,
            out_size=2 * latent_dim,
            width_size=hidden_channels,
            depth=1,
            key=keys[2]
        )
        self.latent_dim = latent_dim
        self.image_size = image_size

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x assumed to have shape (batch, H, W, C)
        # For conv layers, we convert to (batch, C, H, W)
        x = jnp.transpose(x, (0, 3, 1, 2))
        x = jax.vmap(self.conv1)(x)
        x = jax.vmap(self.conv2)(x)
        # Flatten each feature map.
        x = x.reshape(x.shape[0], -1)
        out = jax.vmap(self.mlp)(x)
        mu, logvar = jnp.split(out, 2, axis=-1)
        return mu, logvar


class CNNDecoder(eqx.Module, BaseDecoder):
    """
    A convolutional decoder for image data.
    It first maps the latent variable to a flat feature map using an MLP,
    then uses two ConvTranspose2d layers to upsample back to the image dimensions.
    The output is reshaped to (batch, H, W, C).
    """
    mlp: eqx.nn.MLP
    deconv1: eqx.nn.ConvTranspose2d
    deconv2: eqx.nn.ConvTranspose2d
    image_size: int

    def __init__(self, latent_dim: int, hidden_channels: int, output_channels: int, 
                 image_size: int, *, key):
        keys = jax.random.split(key, 3)
        flat_dim = (image_size // 4) * (image_size // 4) * hidden_channels
        self.mlp = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=flat_dim,
            width_size=hidden_channels,
            depth=1,
            key=keys[0]
        )
        self.deconv1 = eqx.nn.ConvTranspose2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,
            padding="SAME",
            key=keys[1]
        )
        self.deconv2 = eqx.nn.ConvTranspose2d(
            in_channels=hidden_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=2,
            padding="SAME",
            key=keys[2]
        )
        self.image_size = image_size

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        # z assumed to have shape (batch, latent_dim)
        x = jax.vmap(self.mlp)(z)
        # Reshape to (batch, hidden_channels, H//4, W//4)
        batch = x.shape[0]
        H = self.image_size // 4
        W = self.image_size // 4
        # Infer number of channels from the flat dimension.
        hidden_channels = x.shape[-1] // (H * W)
        x = x.reshape(batch, hidden_channels, H, W)
        x = jax.vmap(self.deconv1)(x)
        x = jax.vmap(self.deconv2)(x)
        # Convert back to NHWC format.
        x = jnp.transpose(x, (0, 2, 3, 1))
        return x

###########################################
# 3. Residual MLP Versions (with skip connections)
###########################################
# These versions add a simple residual (skip) connection in each layer.
# For simplicity, we assume that each hidden layer in the MLP has the same size
# and that the input/output dimensions are compatible for the skip connection.

class ResidualMLPEncoder(eqx.Module, BaseEncoder):
    """
    An MLP encoder that uses residual connections.
    This implementation builds a stack of residual blocks and then produces 2*latent_dim outputs.
    """
    layers: list[eqx.nn.MLP]
    latent_dim: int

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int, *, key):
        self.latent_dim = latent_dim
        keys = jax.random.split(key, num_layers + 1)
        self.layers = []
        # First layer maps input to hidden_dim
        self.layers.append(eqx.nn.MLP(
            in_size=input_dim,
            out_size=hidden_dim,
            width_size=hidden_dim,
            depth=1,
            key=keys[0]
        ))
        # Subsequent layers are residual blocks (hidden_dim to hidden_dim)
        for i in range(1, num_layers):
            self.layers.append(eqx.nn.MLP(
                in_size=hidden_dim,
                out_size=hidden_dim,
                width_size=hidden_dim,
                depth=1,
                key=keys[i]
            ))
        # Final linear layer to output 2*latent_dim
        self.layers.append(eqx.nn.MLP(
            in_size=hidden_dim,
            out_size=2 * latent_dim,
            width_size=hidden_dim,
            depth=1,
            key=keys[-1]
        ))

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x shape: (batch, input_dim)
        h = x
        # Apply first layer (no skip)
        h = jax.vmap(self.layers[0])(h)
        # Apply residual blocks
        for layer in self.layers[1:-1]:
            h_res = jax.vmap(layer)(h)
            h = h + h_res  # skip connection
        # Final layer produces output
        out = jax.vmap(self.layers[-1])(h)
        mu, logvar = jnp.split(out, 2, axis=-1)
        return mu, logvar


class ResidualMLPDecoder(eqx.Module, BaseDecoder):
    """
    An MLP decoder with residual connections.
    It maps z to a reconstruction through several residual layers.
    """
    layers: list[eqx.nn.MLP]

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, num_layers: int, *, key):
        keys = jax.random.split(key, num_layers + 1)
        self.layers = []
        # First layer: latent_dim -> hidden_dim
        self.layers.append(eqx.nn.MLP(
            in_size=latent_dim,
            out_size=hidden_dim,
            width_size=hidden_dim,
            depth=1,
            key=keys[0]
        ))
        # Residual blocks
        for i in range(1, num_layers):
            self.layers.append(eqx.nn.MLP(
                in_size=hidden_dim,
                out_size=hidden_dim,
                width_size=hidden_dim,
                depth=1,
                key=keys[i]
            ))
        # Final layer: hidden_dim -> output_dim
        self.layers.append(eqx.nn.MLP(
            in_size=hidden_dim,
            out_size=output_dim,
            width_size=hidden_dim,
            depth=1,
            key=keys[-1]
        ))

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        # z shape: (batch, latent_dim)
        h = jax.vmap(self.layers[0])(z)
        for layer in self.layers[1:-1]:
            h_res = jax.vmap(layer)(h)
            h = h + h_res
        x_recon = jax.vmap(self.layers[-1])(h)
        return x_recon
    
#####################################
# 2. New GRU-based Encoder/Decoder  #
#####################################

class GRUEncoder(eqx.Module, BaseEncoder):
    """
    Encodes a sequence using a GRUCell.
    Expects input x of shape (batch, seq_length, input_dim) and produces (mu, logvar)
    based on the final hidden state.
    """
    cell: eqx.nn.GRUCell
    linear: eqx.nn.Linear
    latent_dim: int
    hidden_size: int

    def __init__(self, input_dim: int, hidden_size: int, latent_dim: int, *, key):
        key_cell, key_lin = jax.random.split(key)
        self.cell = eqx.nn.GRUCell(input_size=input_dim, hidden_size=hidden_size, key=key_cell)
        self.linear = eqx.nn.Linear(in_features=hidden_size, out_features=2 * latent_dim, key=key_lin)
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x shape: (batch, seq_length, input_dim)
        def scan_fn(h, x_t):
            h_new = self.cell(x_t, h)
            return h_new, h_new

        def process_sequence(seq):
            init_h = jnp.zeros(self.hidden_size)
            final_h, _ = jax.lax.scan(scan_fn, init_h, seq)
            return final_h

        h_final = jax.vmap(process_sequence)(x)  # (batch, hidden_size)
        out = jax.vmap(self.linear)(h_final)        # (batch, 2 * latent_dim)
        mu, logvar = jnp.split(out, 2, axis=-1)
        return mu, logvar


class GRUDecoder(eqx.Module, BaseDecoder):
    """
    Decodes a latent variable into a sequence using a GRUCell.
    Given z (batch, latent_dim) it first computes an initial hidden state,
    then decodes a sequence of fixed length.
    """
    cell: eqx.nn.GRUCell
    init_linear: eqx.nn.Linear  # maps latent -> initial hidden state
    output_linear: eqx.nn.Linear
    seq_length: int
    latent_dim: int
    hidden_size: int
    output_dim: int

    def __init__(self, latent_dim: int, hidden_size: int, output_dim: int, seq_length: int, *, key):
        keys = jax.random.split(key, 3)
        self.init_linear = eqx.nn.Linear(in_features=latent_dim, out_features=hidden_size, key=keys[0])
        self.cell = eqx.nn.GRUCell(input_size=output_dim, hidden_size=hidden_size, key=keys[1])
        self.output_linear = eqx.nn.Linear(in_features=hidden_size, out_features=output_dim, key=keys[2])
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        # z shape: (batch, latent_dim)
        init_h = jax.vmap(self.init_linear)(z)  # (batch, hidden_size)
        # We use a fixed input (e.g. zeros) at each time step.
        init_input = jnp.zeros(self.output_dim)

        def decode_step(h, _):
            h_new = self.cell(init_input, h)
            output = self.output_linear(h_new)
            return h_new, output

        def decode_sequence(h0):
            _, outputs = jax.lax.scan(decode_step, h0, None, length=self.seq_length)
            return outputs  # (seq_length, output_dim)

        outputs = jax.vmap(decode_sequence)(init_h)  # (batch, seq_length, output_dim)
        return outputs

class LSTMEncoder(eqx.Module, BaseEncoder):
    """
    Encodes a sequence with an LSTMCell.
    Expects input x of shape (batch, seq_length, input_dim); uses the final hidden state.
    """
    cell: eqx.nn.LSTMCell
    linear: eqx.nn.Linear
    latent_dim: int
    hidden_size: int

    def __init__(self, input_dim: int, hidden_size: int, latent_dim: int, *, key):
        key_cell, key_lin = jax.random.split(key)
        self.cell = eqx.nn.LSTMCell(input_size=input_dim, hidden_size=hidden_size, key=key_cell)
        self.linear = eqx.nn.Linear(in_features=hidden_size, out_features=2 * latent_dim, key=key_lin)
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x shape: (batch, seq_length, input_dim)
        def scan_fn(state, x_t):
            new_state = self.cell(x_t, state)
            # new_state is (h, c)
            return new_state, new_state[0]  # we keep only h for each step

        def process_sequence(seq):
            init_state = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
            final_state, _ = jax.lax.scan(scan_fn, init_state, seq)
            return final_state[0]  # use final hidden state

        h_final = jax.vmap(process_sequence)(x)
        out = jax.vmap(self.linear)(h_final)
        mu, logvar = jnp.split(out, 2, axis=-1)
        return mu, logvar


class LSTMDecoder(eqx.Module, BaseDecoder):
    """
    Decodes a latent variable into a sequence using an LSTMCell.
    The latent vector is first projected to an initial state (both h and c) and then a sequence is generated.
    """
    cell: eqx.nn.LSTMCell
    init_linear: eqx.nn.Linear  # maps latent -> initial hidden state (for both h and c)
    output_linear: eqx.nn.Linear
    seq_length: int
    latent_dim: int
    hidden_size: int
    output_dim: int

    def __init__(self, latent_dim: int, hidden_size: int, output_dim: int, seq_length: int, *, key):
        keys = jax.random.split(key, 3)
        self.init_linear = eqx.nn.Linear(in_features=latent_dim, out_features=hidden_size, key=keys[0])
        self.cell = eqx.nn.LSTMCell(input_size=output_dim, hidden_size=hidden_size, key=keys[1])
        self.output_linear = eqx.nn.Linear(in_features=hidden_size, out_features=output_dim, key=keys[2])
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:

        init_h = jax.vmap(self.init_linear)(z)
        init_state = (init_h, init_h)  # (h, c) are both set from the latent projection
        init_input = jnp.zeros(self.output_dim)

        def decode_step(state, _):
            new_state = self.cell(init_input, state)
            output = self.output_linear(new_state[0])
            return new_state, output

        def decode_sequence(state0):
            _, outputs = jax.lax.scan(decode_step, state0, None, length=self.seq_length)
            return outputs

        outputs = jax.vmap(decode_sequence)(init_state)
        return outputs

class AttentionEncoder(eqx.Module, BaseEncoder):
    """
    A simple attention–based encoder.
    It applies multi–head self–attention to the input sequence and then averages over time.
    """
    attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.Linear  # projects the pooled output to 2*latent_dim
    latent_dim: int
    input_dim: int
    num_heads: int

    def __init__(self, input_dim: int, latent_dim: int, num_heads: int, *, key):
        keys = jax.random.split(key, 2)
        self.attn = eqx.nn.MultiheadAttention(num_heads=num_heads, query_size=input_dim, key=keys[0])
        self.mlp = eqx.nn.Linear(in_features=input_dim, out_features=2 * latent_dim, key=keys[1])
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.num_heads = num_heads

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x shape: (batch, seq_length, input_dim)
        def attn_fn(seq):
            # seq shape: (seq_length, input_dim)
            out = self.attn(seq, seq, seq)  # self–attention; output shape: (seq_length, input_dim)
            return jnp.mean(out, axis=0)     # average pooling over time
        pooled = jax.vmap(attn_fn)(x)         # (batch, input_dim)
        out = jax.vmap(self.mlp)(pooled)        # (batch, 2 * latent_dim)
        mu, logvar = jnp.split(out, 2, axis=-1)
        return mu, logvar


class AttentionDecoder(eqx.Module, BaseDecoder):
    """
    A simple attention–based decoder.
    It starts from the latent z, projects it into a hidden token,
    repeats it to form a sequence, refines it with self–attention,
    and finally maps each token to the output dimension.
    """
    attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.Linear  # projects latent to a hidden dimension
    output_linear: eqx.nn.Linear
    seq_length: int
    latent_dim: int
    hidden_dim: int
    output_dim: int
    num_heads: int

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int,
                 seq_length: int, num_heads: int, *, key):
        keys = jax.random.split(key, 3)
        self.mlp = eqx.nn.Linear(in_features=latent_dim, out_features=hidden_dim, key=keys[0])
        self.attn = eqx.nn.MultiheadAttention(num_heads=num_heads, query_size=hidden_dim, key=keys[1])
        self.output_linear = eqx.nn.Linear(in_features=hidden_dim, out_features=output_dim, key=keys[2])
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        # z shape: (batch, latent_dim)
        hidden = jax.vmap(self.mlp)(z)  # (batch, hidden_dim)
        # Create a sequence by repeating the hidden vector.
        repeated = jnp.repeat(hidden[:, None, :], self.seq_length, axis=1)  # (batch, seq_length, hidden_dim)
        def attn_fn(seq):
            return self.attn(seq, seq, seq)  # (seq_length, hidden_dim)
        attn_out = jax.vmap(attn_fn)(repeated)
        # Map each time–step to output_dim.
        output = jax.vmap(lambda seq: jax.vmap(self.output_linear)(seq))(attn_out)
        return output  # (batch, seq_length, output_dim)


##########################################################
# Full VAEs



class MLP_VAE(eqx.Module, BaseVAE):
    encoder: MLPEncoder
    decoder: MLPDecoder
    latent_dim: int

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, output_dim: int, *, key):
        enc_key, dec_key = jax.random.split(key)
        self.encoder = MLPEncoder(input_dim, hidden_dim, latent_dim, key=enc_key)
        self.decoder = MLPDecoder(latent_dim, hidden_dim, output_dim, key=dec_key)
        self.latent_dim = latent_dim

    def sample_z(self, rng, mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
        eps = jax.random.normal(rng, shape=mu.shape)
        sigma = jnp.exp(0.5 * logvar)
        return mu + sigma * eps

    def __call__(self, x: jnp.ndarray, rng) -> tuple:
        mu, logvar = self.encoder(x)
        z = self.sample_z(rng, mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    # ------------------------------
    # Training logic encapsulated here
    # ------------------------------
    def fit(
        self,
        data: jnp.ndarray,
        batch_size: int = 64,
        n_epochs: int = 50,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> "MLP_VAE":
        """
        Trains the VAE on the provided data.
        Because Equinox modules are immutable, this method returns the new (trained) model.
        """
        # Create optimizer and initialize its state.
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(eqx.filter(self, eqx.is_array))
        rng = jax.random.PRNGKey(seed)
        n_data = data.shape[0]
        n_batches = int(np.ceil(n_data / batch_size))

        # Copy the model to update it iteratively.
        model = self

        # Define the loss and training step functions (using your existing code)
        @eqx.filter_jit
        def loss_fn(model: MLP_VAE, x_batch, rng):
            """
            Negative ELBO loss
            """
            recon, mu, logvar = model(x_batch, rng)
            # Reconstruction loss: mean squared error
            recon_loss = jnp.mean(jnp.sum((x_batch - recon) ** 2, axis=-1))
            # KL divergence
            kl_div = 0.5 * jnp.mean(
                jnp.sum(jnp.exp(logvar) + (mu**2) - 1.0 - logvar, axis=-1)
            )
            return recon_loss + kl_div

        @eqx.filter_jit
        def make_step(model: MLP_VAE, x_batch, optimizer, opt_state, rng):
            loss_and_grad_fn = eqx.filter_value_and_grad(loss_fn)
            loss_val, grads = loss_and_grad_fn(model, x_batch, rng)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_val

        # Training loop:
        for epoch in range(n_epochs):
            perm = np.random.permutation(n_data)
            data_shuf = data[perm]
            epoch_loss = 0.0

            for i in range(n_batches):
                batch = data_shuf[i * batch_size : (i + 1) * batch_size]
                rng, step_key = jax.random.split(rng)
                model, opt_state, loss_val = make_step(model, batch, optimizer, opt_state, step_key)
                epoch_loss += loss_val

            epoch_loss /= n_batches
            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")

        return model

    # ------------------------------
    # Reconstruction and visualization
    # ------------------------------
    def reconstruct(
        self, x: jnp.ndarray, rng_key, plot: bool = True
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Given new data x and a random key, returns (reconstructions, mu, logvar).
        If plot=True and the data is 2D, a scatter-plot comparing original vs. reconstruction is shown.
        """
        recon, mu, logvar = self(x, rng_key)

        if plot:
            plt.figure(figsize=(6, 6))
            if x.shape[1] == 2:
                plt.scatter(x[:, 0], x[:, 1], label="Original", alpha=0.5)
                plt.scatter(recon[:, 0], recon[:, 1], label="Reconstructed", marker="x")
                plt.title("VAE Reconstruction")
                plt.legend()
                plt.show()
            else:
                print("Data is not 2D—skipping visualization. Returning reconstruction outputs.")

        return recon, mu, logvar

class ConvVAE(eqx.Module, BaseVAE):
    encoder: CNNEncoder
    decoder: CNNDecoder
    latent_dim: int
    image_size: int
    channels: int

    def __init__(
        self,
        image_size: int,
        channels: int,
        hidden_channels: int,
        latent_dim: int,
        *,
        key
    ):
        # Split key for encoder and decoder.
        enc_key, dec_key = jax.random.split(key)
        self.encoder = CNNEncoder(
            input_channels=channels,
            image_size=image_size,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            key=enc_key,
        )
        self.decoder = CNNDecoder(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            output_channels=channels,
            image_size=image_size,
            key=dec_key,
        )
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.channels = channels

    def sample_z(self, rng, mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
        eps = jax.random.normal(rng, shape=mu.shape)
        sigma = jnp.exp(0.5 * logvar)
        return mu + sigma * eps

    def __call__(self, x: jnp.ndarray, rng) -> tuple:
        mu, logvar = self.encoder(x)
        z = self.sample_z(rng, mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def fit(
        self,
        data: jnp.ndarray,
        batch_size: int = 64,
        n_epochs: int = 50,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> "ConvVAE":
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(eqx.filter(self, eqx.is_array))
        rng = jax.random.PRNGKey(seed)
        n_data = data.shape[0]
        n_batches = int(jnp.ceil(n_data / batch_size))
        model = self

        @eqx.filter_jit
        def loss_fn(model: ConvVAE, x_batch, rng):
            recon, mu, logvar = model(x_batch, rng)
            # For images, we sum over H, W, and channels.
            recon_loss = jnp.mean(jnp.sum((x_batch - recon) ** 2, axis=(1,2,3)))
            kl_div = 0.5 * jnp.mean(jnp.sum(jnp.exp(logvar) + mu**2 - 1.0 - logvar, axis=-1))
            return recon_loss + kl_div

        @eqx.filter_jit
        def make_step(model: ConvVAE, x_batch, optimizer, opt_state, rng):
            loss_and_grad_fn = eqx.filter_value_and_grad(loss_fn)
            loss_val, grads = loss_and_grad_fn(model, x_batch, rng)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_val

        for epoch in range(n_epochs):
            perm = jax.random.permutation(rng, n_data)
            data_shuf = data[perm]
            epoch_loss = 0.0
            for i in range(n_batches):
                batch = data_shuf[i * batch_size : (i + 1) * batch_size]
                rng, step_key = jax.random.split(rng)
                model, opt_state, loss_val = make_step(model, batch, optimizer, opt_state, step_key)
                epoch_loss += loss_val
            epoch_loss /= n_batches
            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"[ConvVAE] Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")
        return model

    def reconstruct(self, x: jnp.ndarray, rng_key, plot: bool = True) -> tuple:
        recon, mu, logvar = self(x, rng_key)
        if plot:
            # Plot a grid of images.
            n = min(8, x.shape[0])
            plt.figure(figsize=(12, 3))
            for i in range(n):
                plt.subplot(2, n, i + 1)
                plt.imshow(x[i, :, :, 0], cmap="gray")
                plt.axis("off")
                if i == 0:
                    plt.title("Original")
                plt.subplot(2, n, i + 1 + n)
                plt.imshow(recon[i, :, :, 0], cmap="gray")
                plt.axis("off")
                if i == 0:
                    plt.title("Reconstructed")
            plt.show()
        return recon, mu, logvar
    

class ResidualVAE(eqx.Module, BaseVAE):
    encoder: ResidualMLPEncoder
    decoder: ResidualMLPDecoder
    latent_dim: int

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        output_dim: int,
        num_layers: int,
        *,
        key
    ):
        enc_key, dec_key = jax.random.split(key)
        self.encoder = ResidualMLPEncoder(input_dim, hidden_dim, latent_dim, num_layers, key=enc_key)
        self.decoder = ResidualMLPDecoder(latent_dim, hidden_dim, output_dim, num_layers, key=dec_key)
        self.latent_dim = latent_dim

    def sample_z(self, rng, mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
        eps = jax.random.normal(rng, shape=mu.shape)
        sigma = jnp.exp(0.5 * logvar)
        return mu + sigma * eps

    def __call__(self, x: jnp.ndarray, rng) -> tuple:
        mu, logvar = self.encoder(x)
        z = self.sample_z(rng, mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def fit(
        self,
        data: jnp.ndarray,
        batch_size: int = 64,
        n_epochs: int = 50,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> "ResidualVAE":
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(eqx.filter(self, eqx.is_array))
        rng = jax.random.PRNGKey(seed)
        n_data = data.shape[0]
        n_batches = int(jnp.ceil(n_data / batch_size))
        model = self

        @eqx.filter_jit
        def loss_fn(model: ResidualVAE, x_batch, rng):
            recon, mu, logvar = model(x_batch, rng)
            recon_loss = jnp.mean(jnp.sum((x_batch - recon) ** 2, axis=-1))
            kl_div = 0.5 * jnp.mean(jnp.sum(jnp.exp(logvar) + mu**2 - 1.0 - logvar, axis=-1))
            return recon_loss + kl_div

        @eqx.filter_jit
        def make_step(model: ResidualVAE, x_batch, optimizer, opt_state, rng):
            loss_and_grad_fn = eqx.filter_value_and_grad(loss_fn)
            loss_val, grads = loss_and_grad_fn(model, x_batch, rng)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_val

        for epoch in range(n_epochs):
            perm = jax.random.permutation(rng, n_data)
            data_shuf = data[perm]
            epoch_loss = 0.0
            for i in range(n_batches):
                batch = data_shuf[i * batch_size : (i + 1) * batch_size]
                rng, step_key = jax.random.split(rng)
                model, opt_state, loss_val = make_step(model, batch, optimizer, opt_state, step_key)
                epoch_loss += loss_val
            epoch_loss /= n_batches
            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"[ResidualVAE] Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")
        return model

    def reconstruct(self, x: jnp.ndarray, rng_key, plot: bool = True) -> tuple:
        recon, mu, logvar = self(x, rng_key)
        if plot:
            plt.figure(figsize=(6, 6))
            if x.shape[1] == 2:
                plt.scatter(x[:, 0], x[:, 1], label="Original", alpha=0.5)
                plt.scatter(recon[:, 0], recon[:, 1], label="Reconstructed", marker="x")
                plt.title("Residual VAE Reconstruction")
                plt.legend()
                plt.show()
            else:
                print("Data is not 2D—skipping plot.")
        return recon, mu, logvar
    

class GRU_VAE(eqx.Module, BaseVAE):
    """
    A VAE that encodes sequences with a GRUEncoder and decodes with a GRUDecoder.
    """
    encoder: GRUEncoder
    decoder: GRUDecoder
    latent_dim: int
    seq_length: int

    def __init__(self, input_dim: int, hidden_size: int, latent_dim: int,
                 output_dim: int, seq_length: int, *, key):
        enc_key, dec_key = jax.random.split(key)
        self.encoder = GRUEncoder(input_dim, hidden_size, latent_dim, key=enc_key)
        self.decoder = GRUDecoder(latent_dim, hidden_size, output_dim, seq_length, key=dec_key)
        self.latent_dim = latent_dim
        self.seq_length = seq_length

    def sample_z(self, rng, mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
        eps = jax.random.normal(rng, shape=mu.shape)
        sigma = jnp.exp(0.5 * logvar)
        return mu + sigma * eps

    def __call__(self, x: jnp.ndarray, rng) -> tuple:
        mu, logvar = self.encoder(x)
        z = self.sample_z(rng, mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def fit(
        self,
        data: jnp.ndarray,
        batch_size: int = 64,
        n_epochs: int = 50,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> "GRU_VAE":
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(eqx.filter(self, eqx.is_array))
        rng_local = jax.random.PRNGKey(seed)
        n_data = data.shape[0]
        n_batches = int(np.ceil(n_data / batch_size))
        model = self

        @eqx.filter_jit
        def loss_fn(model: GRU_VAE, x_batch, rng):
            recon, mu, logvar = model(x_batch, rng)
            # Sum over sequence length and output dim.
            recon_loss = jnp.mean(jnp.sum((x_batch - recon) ** 2, axis=(1, 2)))
            kl_div = 0.5 * jnp.mean(jnp.sum(jnp.exp(logvar) + (mu ** 2) - 1 - logvar, axis=-1))
            return recon_loss + kl_div

        @eqx.filter_jit
        def make_step(model: GRU_VAE, x_batch, optimizer, opt_state, rng):
            loss_and_grad_fn = eqx.filter_value_and_grad(loss_fn)
            loss_val, grads = loss_and_grad_fn(model, x_batch, rng)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_val

        for epoch in range(n_epochs):
            perm = np.random.permutation(n_data)
            data_shuf = data[perm]
            epoch_loss = 0.0
            for i in range(n_batches):
                batch = data_shuf[i * batch_size: (i + 1) * batch_size]
                rng_local, step_key = jax.random.split(rng_local)
                model, opt_state, loss_val = make_step(model, batch, optimizer, opt_state, step_key)
                epoch_loss += loss_val
            epoch_loss /= n_batches
            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"[GRU_VAE] Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")
        return model

    def reconstruct(self, x: jnp.ndarray, rng_key, plot: bool = True) -> tuple:
        recon, mu, logvar = self(x, rng_key)
        if plot:
            n = min(8, x.shape[0])
            plt.figure(figsize=(12, 6))
            for i in range(n):
                plt.subplot(2, n, i + 1)
                plt.plot(x[i, :, 0])
                plt.title("Original")
                plt.subplot(2, n, i + 1 + n)
                plt.plot(recon[i, :, 0])
                plt.title("Reconstructed")
            plt.tight_layout()
            plt.show()
        return recon, mu, logvar

class LSTM_VAE(eqx.Module, BaseVAE):
    """
    A VAE built from an LSTMEncoder and LSTMDecoder.
    """
    encoder: LSTMEncoder
    decoder: LSTMDecoder
    latent_dim: int
    seq_length: int

    def __init__(self, input_dim: int, hidden_size: int, latent_dim: int,
                 output_dim: int, seq_length: int, *, key):
        enc_key, dec_key = jax.random.split(key)
        self.encoder = LSTMEncoder(input_dim, hidden_size, latent_dim, key=enc_key)
        self.decoder = LSTMDecoder(latent_dim, hidden_size, output_dim, seq_length, key=dec_key)
        self.latent_dim = latent_dim
        self.seq_length = seq_length

    def sample_z(self, rng, mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
        eps = jax.random.normal(rng, shape=mu.shape)
        sigma = jnp.exp(0.5 * logvar)
        return mu + sigma * eps

    def __call__(self, x: jnp.ndarray, rng) -> tuple:
        mu, logvar = self.encoder(x)
        z = self.sample_z(rng, mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def fit(
        self,
        data: jnp.ndarray,
        batch_size: int = 64,
        n_epochs: int = 50,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> "LSTM_VAE":
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(eqx.filter(self, eqx.is_array))
        rng_local = jax.random.PRNGKey(seed)
        n_data = data.shape[0]
        n_batches = int(np.ceil(n_data / batch_size))
        model = self

        @eqx.filter_jit
        def loss_fn(model: LSTM_VAE, x_batch, rng):
            recon, mu, logvar = model(x_batch, rng)
            recon_loss = jnp.mean(jnp.sum((x_batch - recon) ** 2, axis=(1, 2)))
            kl_div = 0.5 * jnp.mean(jnp.sum(jnp.exp(logvar) + (mu ** 2) - 1 - logvar, axis=-1))
            return recon_loss + kl_div

        @eqx.filter_jit
        def make_step(model: LSTM_VAE, x_batch, optimizer, opt_state, rng):
            loss_and_grad_fn = eqx.filter_value_and_grad(loss_fn)
            loss_val, grads = loss_and_grad_fn(model, x_batch, rng)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_val

        for epoch in range(n_epochs):
            perm = np.random.permutation(n_data)
            data_shuf = data[perm]
            epoch_loss = 0.0
            for i in range(n_batches):
                batch = data_shuf[i * batch_size: (i + 1) * batch_size]
                rng_local, step_key = jax.random.split(rng_local)
                model, opt_state, loss_val = make_step(model, batch, optimizer, opt_state, step_key)
                epoch_loss += loss_val
            epoch_loss /= n_batches
            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"[LSTM_VAE] Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")
        return model

    def reconstruct(self, x: jnp.ndarray, rng_key, plot: bool = True) -> tuple:
        recon, mu, logvar = self(x, rng_key)
        if plot:
            n = min(8, x.shape[0])
            plt.figure(figsize=(12, 6))
            for i in range(n):
                plt.subplot(2, n, i + 1)
                plt.plot(x[i, :, 0])
                plt.title("Original")
                plt.subplot(2, n, i + 1 + n)
                plt.plot(recon[i, :, 0])
                plt.title("Reconstructed")
            plt.tight_layout()
            plt.show()
        return recon, mu, logvar

class MultiHeadAttentionVAE(eqx.Module, BaseVAE):
    """
    A VAE that uses self–attention both to encode and decode sequences.
    """
    encoder: AttentionEncoder
    decoder: AttentionDecoder
    latent_dim: int
    seq_length: int

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int,
                 output_dim: int, seq_length: int, num_heads: int, *, key):
        enc_key, dec_key = jax.random.split(key)
        self.encoder = AttentionEncoder(input_dim, latent_dim, num_heads, key=enc_key)
        self.decoder = AttentionDecoder(latent_dim, hidden_dim, output_dim, seq_length, num_heads, key=dec_key)
        self.latent_dim = latent_dim
        self.seq_length = seq_length

    def sample_z(self, rng, mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
        eps = jax.random.normal(rng, shape=mu.shape)
        sigma = jnp.exp(0.5 * logvar)
        return mu + sigma * eps

    def __call__(self, x: jnp.ndarray, rng) -> tuple:
        mu, logvar = self.encoder(x)
        z = self.sample_z(rng, mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def fit(
        self,
        data: jnp.ndarray,
        batch_size: int = 64,
        n_epochs: int = 50,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> "MultiHeadAttentionVAE":
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(eqx.filter(self, eqx.is_array))
        rng_local = jax.random.PRNGKey(seed)
        n_data = data.shape[0]
        n_batches = int(np.ceil(n_data / batch_size))
        model = self

        @eqx.filter_jit
        def loss_fn(model: MultiHeadAttentionVAE, x_batch, rng):
            recon, mu, logvar = model(x_batch, rng)
            recon_loss = jnp.mean(jnp.sum((x_batch - recon) ** 2, axis=(1, 2)))
            kl_div = 0.5 * jnp.mean(jnp.sum(jnp.exp(logvar) + (mu ** 2) - 1 - logvar, axis=-1))
            return recon_loss + kl_div

        @eqx.filter_jit
        def make_step(model: MultiHeadAttentionVAE, x_batch, optimizer, opt_state, rng):
            loss_and_grad_fn = eqx.filter_value_and_grad(loss_fn)
            loss_val, grads = loss_and_grad_fn(model, x_batch, rng)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_val

        for epoch in range(n_epochs):
            perm = jax.random.permutation(rng_local, n_data)
            data_shuf = data[perm]
            epoch_loss = 0.0
            for i in range(n_batches):
                batch = data_shuf[i * batch_size: (i + 1) * batch_size]
                rng_local, step_key = jax.random.split(rng_local)
                model, opt_state, loss_val = make_step(model, batch, optimizer, opt_state, step_key)
                epoch_loss += loss_val
            epoch_loss /= n_batches
            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"[MultiHeadAttentionVAE] Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")
        return model

    def reconstruct(self, x: jnp.ndarray, rng_key, plot: bool = True) -> tuple:
        recon, mu, logvar = self(x, rng_key)
        if plot:
            n = min(8, x.shape[0])
            plt.figure(figsize=(12, 6))
            for i in range(n):
                plt.subplot(2, n, i + 1)
                plt.plot(x[i, :, 0])
                plt.title("Original")
                plt.subplot(2, n, i + 1 + n)
                plt.plot(recon[i, :, 0])
                plt.title("Reconstructed")
            plt.tight_layout()
            plt.show()
        return recon, mu, logvar