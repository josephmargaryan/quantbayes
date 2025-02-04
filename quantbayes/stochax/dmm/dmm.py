#!/usr/bin/env python3
"""
A small Deep Markov Model in Equinox, with:
- p(z_t | z_{t-1}) and p(x_t | z_t) as MLP-based Gaussians
- q(z_t | z_{t-1}, x_{1..t}) using an LSTM-based posterior
- We'll handle mini-batches of sequences by vmap over the batch dimension for MLPs
  and use an explicit python loop + vmap for the LSTM hidden states.
"""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# 1) Synthetic Data
# -----------------------
def generate_synthetic_sequences(rng_key, n_sequences=32, T=10, x_dim=1, z_dim=2):
    """
    We'll create random linear-Gaussian style transitions+emissions:
        z_t = A*z_{t-1} + noise
        x_t = B*z_t       + noise
    """
    rng = np.random.default_rng(int(rng_key[0]))
    A = 0.5 * rng.normal(size=(z_dim, z_dim))
    B = 0.7 * rng.normal(size=(x_dim, z_dim))

    z_list = []
    x_list = []
    for _ in range(n_sequences):
        z_prev = rng.normal(size=(z_dim,))
        seq_z = []
        seq_x = []
        for t in range(T):
            z_t = A @ z_prev + 0.1 * rng.normal(size=(z_dim,))
            x_t = B @ z_t + 0.1 * rng.normal(size=(x_dim,))
            seq_z.append(z_t)
            seq_x.append(x_t)
            z_prev = z_t
        z_list.append(np.stack(seq_z, axis=0))  # (T, z_dim)
        x_list.append(np.stack(seq_x, axis=0))  # (T, x_dim)
    z_data = jnp.stack([jnp.array(z) for z in z_list], axis=0)  # (N, T, z_dim)
    x_data = jnp.stack([jnp.array(x) for x in x_list], axis=0)  # (N, T, x_dim)
    return x_data, z_data


# -----------------------
# 2) Transition, Emitter, Posterior RNN
# -----------------------
class Transition(eqx.Module):
    """
    p(z_t|z_{t-1}) = Normal(loc, scale) 
    We'll have an MLP(in_size=z_dim, out_size=2*z_dim), then split => (loc, scale).
    """
    mlp: eqx.nn.MLP
    z_dim: int

    def __init__(self, z_dim, hidden_dim, *, key):
        super().__init__()
        self.z_dim = z_dim
        self.mlp = eqx.nn.MLP(
            in_size=z_dim,
            out_size=2 * z_dim,
            width_size=hidden_dim,
            depth=2,
            key=key
        )

    def __call__(self, z_prev_batch):
        """
        z_prev_batch shape: (batch, z_dim)
        returns (loc, scale) each (batch, z_dim)
        """
        def forward_single(zp):
            out = self.mlp(zp)  # shape (2*z_dim,)
            loc, log_scale = jnp.split(out, 2, axis=-1)
            scale = jax.nn.softplus(log_scale) + 1e-3
            return loc, scale

        locs, scales = jax.vmap(forward_single)(z_prev_batch)
        return locs, scales


class Emitter(eqx.Module):
    """
    p(x_t|z_t) = Normal(loc, scale)
    MLP => out_size= 2*x_dim => split => (loc, scale).
    """
    mlp: eqx.nn.MLP
    x_dim: int

    def __init__(self, z_dim, x_dim, hidden_dim, *, key):
        super().__init__()
        self.x_dim = x_dim
        self.mlp = eqx.nn.MLP(
            in_size=z_dim,
            out_size=2 * x_dim,
            width_size=hidden_dim,
            depth=2,
            key=key
        )

    def __call__(self, z_batch):
        """
        z_batch shape: (batch, z_dim)
        returns (loc, scale) each (batch, x_dim)
        """
        def forward_single(z):
            out = self.mlp(z)  # shape (2*x_dim,)
            loc, log_scale = jnp.split(out, 2, axis=-1)
            scale = jax.nn.softplus(log_scale) + 1e-3
            return loc, scale

        locs, scales = jax.vmap(forward_single)(z_batch)
        return locs, scales


class PosteriorRNN(eqx.Module):
    """
    We want to process a *batch* of sequences in a single forward pass.
    We'll do a python for-loop across time, but inside it, we apply `vmap` across the batch. 

    We'll store a single eqx.nn.LSTMCell(...) => input_size = x_dim, hidden_size=hidden_dim.

    Then for each time-step, we call `h, c = jax.vmap(lstm)(x_t, (h, c))` 
    to handle the entire batch in parallel. We'll collect the hidden states h_t.
    """
    lstm: eqx.nn.LSTMCell
    hidden_dim: int
    x_dim: int

    def __init__(self, x_dim, hidden_dim, *, key):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.x_dim = x_dim
        self.lstm = eqx.nn.LSTMCell(
            input_size=x_dim,
            hidden_size=hidden_dim,
            key=key
        )

    def __call__(self, x_seq):
        """
        x_seq shape: (batch, T, x_dim)
        We'll produce h_seq shape: (batch, T, hidden_dim)
        """
        B, T, _ = x_seq.shape
        # Initialize hidden states (one for each example in the batch)
        h = jnp.zeros((B, self.hidden_dim))
        c = jnp.zeros((B, self.hidden_dim))

        h_list = []
        for t in range(T):
            x_t = x_seq[:, t, :]         # shape (B, x_dim)
            # We'll batch it up with vmap:
            h, c = jax.vmap(self.lstm)(x_t, (h, c))  # shape (B, hidden_dim)
            h_list.append(h)

        # shape => (T, B, hidden_dim) => transpose => (B, T, hidden_dim)
        h_seq = jnp.stack(h_list, axis=0).transpose((1,0,2))
        return h_seq


class Combiner(eqx.Module):
    """
    q(z_t | z_{t-1}, h_t). We'll do MLP => out_size= 2*z_dim => split => (loc, scale).
    """
    mlp: eqx.nn.MLP
    z_dim: int

    def __init__(self, z_dim, hidden_dim, *, key):
        super().__init__()
        self.z_dim = z_dim
        self.mlp = eqx.nn.MLP(
            in_size=z_dim + hidden_dim,  # we'll concat z_{t-1} and h_t
            out_size=2 * z_dim,
            width_size=hidden_dim,
            depth=2,
            key=key
        )

    def __call__(self, z_prev_batch, h_batch):
        """
        Each shape: (batch, ...)
        We'll produce (q_loc, q_scale).
        """
        def forward_single(zp, h):
            inp = jnp.concatenate([zp, h], axis=-1)  # shape (z_dim+hidden_dim,)
            out = self.mlp(inp)  # shape (2*z_dim,)
            loc, log_scale = jnp.split(out, 2, axis=-1)
            scale = jax.nn.softplus(log_scale) + 1e-3
            return loc, scale

        locs, scales = jax.vmap(forward_single)(z_prev_batch, h_batch)
        return locs, scales


# -----------------------
# 3) The DMM
# -----------------------
class DMM(eqx.Module):
    """
    Full model:
      - Learned p(z_1).
      - Transition: p(z_t|z_{t-1})
      - Emitter: p(x_t|z_t)
      - PosteriorRNN => h_t
      - Combiner => q(z_t|z_{t-1}, h_t)
    We'll sample z_t ~ q(...) and accumulate the ELBO.
    """

    transition: Transition
    emitter: Emitter
    posterior_rnn: PosteriorRNN
    combiner: Combiner
    z_dim: int

    z_init_loc: jnp.ndarray
    z_init_logscale: jnp.ndarray

    def __init__(self, x_dim, z_dim, hidden_dim, *, key):
        super().__init__()
        k_trans, k_emit, k_rnn, k_comb, k_prior = jax.random.split(key, 5)
        self.transition = Transition(z_dim, hidden_dim, key=k_trans)
        self.emitter = Emitter(z_dim, x_dim, hidden_dim, key=k_emit)
        self.posterior_rnn = PosteriorRNN(x_dim, hidden_dim, key=k_rnn)
        self.combiner = Combiner(z_dim, hidden_dim, key=k_comb)
        self.z_dim = z_dim

        # Learned prior for z1
        self.z_init_loc = jnp.zeros((z_dim,))
        self.z_init_logscale = jnp.zeros((z_dim,))

    def reparam_sample(self, rng, loc, scale):
        eps = jax.random.normal(rng, shape=loc.shape)
        return loc + scale * eps

    def log_normal_diag(self, x, loc, scale):
        """
        log N(x; loc, scaleI), sum across last dim => shape (batch,)
        """
        var = scale**2
        logdet = jnp.sum(jnp.log(var + 1e-8), axis=-1)
        sq_term = jnp.sum(((x - loc)**2) / (var + 1e-8), axis=-1)
        dim = x.shape[-1]
        logp = -0.5 * (logdet + sq_term + dim * jnp.log(2*jnp.pi))
        return logp

    def __call__(self, x_seq, rng):
        """
        Negative ELBO for a batch of sequences x_seq. 
        x_seq shape: (batch, T, x_dim).
        We'll do a single sample of z_{1..T} from q, then compute sum(log p - log q).
        """
        B, T, x_dim = x_seq.shape
        # RNN => h_seq (B, T, hidden_dim)
        h_seq = self.posterior_rnn(x_seq)

        # z1 prior
        z1_loc = self.z_init_loc     # shape (z_dim,)
        z1_scale = jnp.exp(self.z_init_logscale)  # shape (z_dim,)

        # We'll sample z1 from q(z1|z0=0,h1)
        z0 = jnp.zeros((B, self.z_dim))  # "fake" z_{0}
        h1 = h_seq[:, 0, :]              # shape (B, hidden_dim)
        rngs = jax.random.split(rng, T)

        q1_loc, q1_scale = self.combiner(z0, h1)
        z1 = self.reparam_sample(rngs[0], q1_loc, q1_scale)

        # log p(z1)
        lp_z1 = self.log_normal_diag(z1, z1_loc[None,:], z1_scale[None,:])
        # log q(z1)
        lq_z1 = self.log_normal_diag(z1, q1_loc, q1_scale)
        log_w = jnp.sum(lp_z1 - lq_z1)  # sum over batch

        # log p(x1|z1)
        x1 = x_seq[:, 0, :]
        x1_loc, x1_scale = self.emitter(z1)
        lp_x1 = self.log_normal_diag(x1, x1_loc, x1_scale)
        log_w += jnp.sum(lp_x1)

        z_prev = z1
        for t in range(1, T):
            # sample z_t from q
            h_t = h_seq[:, t, :]
            qt_loc, qt_scale = self.combiner(z_prev, h_t)
            z_t = self.reparam_sample(rngs[t], qt_loc, qt_scale)

            # p(z_t|z_{t-1})
            pt_loc, pt_scale = self.transition(z_prev)
            lp_zt = self.log_normal_diag(z_t, pt_loc, pt_scale)
            lq_zt = self.log_normal_diag(z_t, qt_loc, qt_scale)
            log_w += jnp.sum(lp_zt - lq_zt)

            # p(x_t|z_t)
            x_t = x_seq[:, t, :]
            xt_loc, xt_scale = self.emitter(z_t)
            lp_xt = self.log_normal_diag(x_t, xt_loc, xt_scale)
            log_w += jnp.sum(lp_xt)

            z_prev = z_t

        # average across batch => negative ELBO
        neg_elbo = -log_w / B
        return neg_elbo


# -----------------------
# 4) Train
# -----------------------
@eqx.filter_jit
def loss_fn(model, x_seq, rng):
    return model(x_seq, rng)

@eqx.filter_jit
def make_step(model, x_seq, optimizer, opt_state, rng):
    val, grads = eqx.filter_value_and_grad(loss_fn)(model, x_seq, rng)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, val

def train_dmm(model, x_data, n_epochs=100, batch_size=8, lr=1e-3, seed=42):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    N = x_data.shape[0]
    n_batches = int(np.ceil(N / batch_size))
    rng = jax.random.PRNGKey(seed)

    for epoch in range(n_epochs):
        perm = np.random.permutation(N)
        x_shuf = x_data[perm]

        epoch_loss = 0.0
        for i in range(n_batches):
            batch = x_shuf[i * batch_size : (i + 1) * batch_size]
            rng, step_key = jax.random.split(rng)
            model, opt_state, val = make_step(model, batch, optimizer, opt_state, step_key)
            epoch_loss += val

        epoch_loss /= n_batches
        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, -ELBO: {epoch_loss:.4f}")

    return model


# -----------------------
# 5) Main
# -----------------------
def main():
    rng_key = jax.random.PRNGKey(0)
    x_data, z_data = generate_synthetic_sequences(rng_key, n_sequences=64, T=15, x_dim=1, z_dim=2)
    print("x_data shape:", x_data.shape, "z_data shape:", z_data.shape)

    x_dim = 1
    z_dim = 2
    hidden_dim = 16
    init_key = jax.random.PRNGKey(1)
    model = DMM(x_dim, z_dim, hidden_dim, key=init_key)

    trained_model = train_dmm(model, x_data, n_epochs=50, batch_size=16, lr=1e-3, seed=999)

    # Evaluate on a single sequence
    rng_test = jax.random.PRNGKey(123)
    test_seq = x_data[0:1]  # shape (1,T,x_dim)
    neg_elbo_val = trained_model(test_seq, rng_test)
    print(f"Final negative ELBO on one sequence: {neg_elbo_val:.4f}")

    # "Reconstruct" the single sequence by sampling z_t from posterior => p(x_t|z_t).
    def sample_forward(dmm, single_seq, key):
        B, T, _ = single_seq.shape  # B=1
        # h_seq = (1,T,hidden_dim)
        h_seq = dmm.posterior_rnn(single_seq)

        rngs = jax.random.split(key, T)
        z0 = jnp.zeros((B, dmm.z_dim))
        q1_loc, q1_scale = dmm.combiner(z0, h_seq[:,0,:])
        z1 = dmm.reparam_sample(rngs[0], q1_loc, q1_scale)
        x1_loc, x1_scale = dmm.emitter(z1)
        x1_recon = x1_loc  # mean

        z_prev = z1
        recons = [x1_recon]

        for t in range(1, T):
            qt_loc, qt_scale = dmm.combiner(z_prev, h_seq[:,t,:])
            zt = dmm.reparam_sample(rngs[t], qt_loc, qt_scale)
            xt_loc, xt_scale = dmm.emitter(zt)
            x_recon = xt_loc
            recons.append(x_recon)
            z_prev = zt
        # shape (T, x_dim)
        return jnp.concatenate(recons, axis=0)

    recon_seq = sample_forward(trained_model, test_seq, rng_test)
    original = test_seq[0, :, 0]    # shape (T,)
    rec = recon_seq[:, 0]          # shape (T,)

    plt.figure(figsize=(8,4))
    plt.plot(original, "b-o", label="Original")
    plt.plot(rec, "r--o", label="Reconstruction")
    plt.title("DMM Observations: original vs. recon")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
