# score_diffusion/models/timeseries_1d.py

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

class TimeSeriesScoreModel(eqx.Module):
    """
    Score network for 1D time-series data.
    Suppose input shape is [sequence_length].
    We combine the diffusion time t with the sequence dimension.
    """
    embedding: eqx.nn.MLP
    project_in: eqx.nn.Linear
    hidden_layers: list
    project_out: eqx.nn.Linear
    seq_length: int

    def __init__(self, seq_length, hidden_dim=64, time_emb_dim=64, num_layers=4, *, key):
        # key splitting
        keys = jr.split(key, num_layers+3)
        self.seq_length = seq_length
        # Embedding for diffusion time
        self.embedding = eqx.nn.MLP(
            in_size=1, out_size=time_emb_dim, width_size=2*time_emb_dim, depth=2, key=keys[0]
        )
        self.project_in = eqx.nn.Linear(seq_length + time_emb_dim, hidden_dim, key=keys[1])
        self.hidden_layers = []
        for i in range(num_layers-2):
            self.hidden_layers.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[i+2]))
        self.project_out = eqx.nn.Linear(hidden_dim, seq_length, key=keys[num_layers+1])

    def __call__(self, t, x):
        """
        t: scalar or shape [batch] 
        x: shape [seq_length] or [batch, seq_length]
        """
        # Make sure shapes align
        if x.ndim == 1:
            x = x[None, :]  # shape => [1, seq_length]
        batch_size = x.shape[0]
        # embed time
        t_emb = self.embedding(t[:, None])  # shape => [batch, time_emb_dim]
        # concat
        xt = jnp.concatenate([x, t_emb], axis=-1)
        # project in
        h = jax.nn.silu(self.project_in(xt))
        # hidden
        for layer in self.hidden_layers:
            h = jax.nn.silu(layer(h))
        # out
        out = self.project_out(h)
        # shape => [batch, seq_length]
        return out
