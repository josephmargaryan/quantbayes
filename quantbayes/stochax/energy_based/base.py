# ebm_base.py
from abc import ABC, abstractmethod
from typing import Callable, Optional
import jax
import jax.random as jrandom
import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx


class BaseEBM(eqx.Module, ABC):
    """
    Abstract base class for Energy-Based Models.

    Required Methods:
      - energy(x): Returns the scalar energy (or batch of energies) for input x.
    """

    @abstractmethod
    def energy(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the energy of input x.
        This should return a shape (batch,) or () if single sample.
        """
        pass

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        By default, calling the model is the same as computing the energy.
        Feel free to override if needed.
        """
        return self.energy(x)


class MLPBasedEBM(BaseEBM):
    """
    EBM using a simple MLP for energy computation:
        E(x) = MLP(x)
    """

    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        depth: int,
        key: jrandom.PRNGKey,
        activation: Callable = jnn.relu,
    ):
        super().__init__()
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=1,
            width_size=hidden_size,
            depth=depth,
            activation=activation,
            final_activation=lambda x: x,  # Identity on the final layer.
            key=key,
        )

    def energy(self, x: jnp.ndarray) -> jnp.ndarray:
        # Apply vmap so that self.mlp processes each sample (of shape (in_size,)) individually.
        energies = jax.vmap(self.mlp)(x)  # Now energies has shape (batch, 1)
        return jnp.squeeze(energies, axis=-1)


class ConvEBM(BaseEBM):
    """
    EBM for image data. Typical architecture for 2D images:
      E(x) = CNN(x)
    We flatten or pool at the end to produce a scalar energy.
    """

    conv_net: eqx.nn.Sequential

    def __init__(self, key, in_channels=3, hidden_channels=32, out_channels=64):
        super().__init__()
        # Define a series of conv -> activation -> conv -> activation -> pooling -> linear.
        k1, k2, k3 = jrandom.split(key, 3)
        self.conv_net = eqx.nn.Sequential(
            [
                eqx.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    key=k1,
                ),
                # Wrap activation to ignore extra keyword arguments.
                lambda x, **kwargs: jnn.relu(x),
                eqx.nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    key=k2,
                ),
                lambda x, **kwargs: jnn.relu(x),
                GlobalAvgPool2d(),
                eqx.nn.Linear(hidden_channels, 1, use_bias=True, key=k3),
            ]
        )

    def energy(self, x: jnp.ndarray) -> jnp.ndarray:
        # x has shape (batch, C, H, W)
        # Use vmap to apply conv_net to each image individually.
        energies = jax.vmap(self.conv_net)(x)  # Now shape is (batch, 1)
        return jnp.squeeze(energies, axis=-1)


class GlobalAvgPool2d(eqx.Module):
    """
    A small helper module to do average pooling across spatial dimensions H, W.
    """

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        # x is shape (channels, H, W) when used with vmap.
        return jnp.mean(x, axis=(-2, -1))  # shape (channels)


class RNNBasedEBM(BaseEBM):
    """
    EBM for sequence data using a GRU cell.
    It scans over the sequence using a GRUCell and then maps the final hidden state to a scalar energy.
    """

    gru_cell: eqx.nn.GRUCell
    linear: eqx.nn.Linear
    hidden_size: int

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        key: jrandom.PRNGKey,
    ):
        super().__init__()
        k1, k2 = jrandom.split(key, 2)
        self.gru_cell = eqx.nn.GRUCell(
            input_size=input_size, hidden_size=hidden_size, key=k1
        )
        self.linear = eqx.nn.Linear(hidden_size, 1, use_bias=True, key=k2)
        self.hidden_size = hidden_size

    def energy(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: shape (batch, seq_len, input_size)
        Processes the sequence using a GRU and returns an energy value per sequence.
        """
        batch_size, seq_len, _ = x.shape
        # Initialize hidden state for each sequence in the batch.
        init_h = jnp.zeros((batch_size, self.hidden_size))

        def scan_fn(h, x_t):
            # h: shape (batch, hidden_size)
            # x_t: shape (batch, input_size)
            new_h = jax.vmap(self.gru_cell)(x_t, h)  # returns (batch, hidden_size)
            return new_h, new_h

        # Transpose x so that time is the leading axis: (seq_len, batch, input_size)
        final_h, _ = jax.lax.scan(scan_fn, init_h, x.transpose(1, 0, 2))
        # final_h has shape (batch, hidden_size)

        # Use vmap to apply the linear layer to each sample.
        energies = jax.vmap(self.linear)(final_h)  # now shape (batch, 1)
        return jnp.squeeze(energies, axis=-1)


class LSTMBasedEBM(BaseEBM):
    """
    EBM for sequence data using an LSTM.
    It scans over the input sequence with an LSTMCell and then maps the final hidden state to a scalar energy.
    """

    lstm_cell: eqx.nn.LSTMCell
    linear: eqx.nn.Linear
    hidden_size: int

    def __init__(self, input_size: int, hidden_size: int, key: jrandom.PRNGKey):
        super().__init__()
        k1, k2 = jrandom.split(key, 2)
        self.lstm_cell = eqx.nn.LSTMCell(
            input_size=input_size, hidden_size=hidden_size, key=k1
        )
        self.linear = eqx.nn.Linear(hidden_size, 1, use_bias=True, key=k2)
        self.hidden_size = hidden_size

    def energy(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: shape (batch, seq_len, input_size)
        Returns energy for each sequence in the batch.
        """
        batch_size, seq_len, _ = x.shape
        # Initialize both hidden and cell states to zeros.
        init_state = (
            jnp.zeros((batch_size, self.hidden_size)),
            jnp.zeros((batch_size, self.hidden_size)),
        )

        def scan_fn(state, x_t):
            # x_t: shape (batch, input_size)
            # Process each time step with the LSTMCell (using vmap over the batch).
            new_state = jax.vmap(self.lstm_cell)(x_t, state)
            # new_state is a tuple (h, c)
            return new_state, new_state[0]  # output the hidden state

        # Transpose x so that time is the leading axis: (seq_len, batch, input_size)
        final_state, _ = jax.lax.scan(scan_fn, init_state, x.transpose(1, 0, 2))
        final_h = final_state[0]  # shape: (batch, hidden_size)

        # Use vmap to apply the linear layer on each sample in the batch.
        energies = jax.vmap(self.linear)(final_h)  # shape: (batch, 1)
        return jnp.squeeze(energies, axis=-1)


class AttentionBasedEBM(BaseEBM):
    """
    EBM that uses multihead attention to process sequence (or set) data.
    It adds learnable positional embeddings to the input before applying attention.
    """

    attn: eqx.nn.MultiheadAttention
    linear: eqx.nn.Linear
    pos_embed: jnp.ndarray  # learnable positional embeddings
    max_seq_len: int
    input_size: int
    num_heads: int

    def __init__(
        self,
        input_size: int,
        num_heads: int,
        max_seq_len: int,
        key: jrandom.PRNGKey,
        qk_size: Optional[int] = None,
        vo_size: Optional[int] = None,
    ):
        super().__init__()
        if qk_size is None:
            qk_size = input_size // num_heads
        if vo_size is None:
            vo_size = input_size // num_heads
        k1, k2, k3 = jrandom.split(key, 3)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=input_size,
            key_size=input_size,
            value_size=input_size,
            output_size=input_size,
            qk_size=qk_size,
            vo_size=vo_size,
            use_query_bias=False,
            use_key_bias=False,
            use_value_bias=False,
            use_output_bias=False,
            dropout_p=0.0,
            inference=True,  # disable dropout during evaluation
            key=k1,
        )
        self.linear = eqx.nn.Linear(input_size, 1, use_bias=True, key=k2)
        # Initialize learnable positional embeddings for up to max_seq_len positions.
        self.max_seq_len = max_seq_len
        self.input_size = input_size
        self.num_heads = num_heads
        # Simply assign a normal array; Equinox treats it as trainable.
        self.pos_embed = jax.random.normal(k3, (max_seq_len, input_size))

    def energy(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: shape (batch, seq_len, input_size)
        Adds positional embeddings and processes through multihead attention.
        """
        batch, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        # Slice positional embeddings to match the current sequence length.
        pos_embeds = self.pos_embed[:seq_len, :]  # shape (seq_len, input_size)
        # Broadcast addition: (batch, seq_len, input_size) + (seq_len, input_size)
        x_pos = x + pos_embeds

        # Process each sample in the batch independently.
        def sample_energy(sample):
            # sample: shape (seq_len, input_size)
            attended = self.attn(sample, sample, sample)
            pooled = jnp.mean(attended, axis=0)  # aggregate over time
            return self.linear(pooled)  # shape (1,)

        energies = jax.vmap(sample_energy)(x_pos)  # shape (batch, 1)
        return jnp.squeeze(energies, axis=-1)
