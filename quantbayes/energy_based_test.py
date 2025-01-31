###########################################################
# EBM_EXAMPLE.PY
# A complete end-to-end example of an Energy-Based Model
# in JAX + Flax with short-run Contrastive Divergence.
#
# We'll do:
#   1) Toy 2D data generation (mixture of Gaussians).
#   2) A Flax "EnergyNetwork" that maps x-> E_\theta(x).
#   3) Training by short-run Langevin MCMC to get "negative samples."
#   4) Visualizing final data vs. model samples.
# 
# Python 3.8+ recommended. 
###########################################################

import jax
import jax.numpy as jnp
from jax import random, grad, jit
from jax import vmap

import flax.linen as nn
from flax.training import train_state
import optax

import matplotlib.pyplot as plt
import numpy as np


##############################################
# 1. Toy 2D Data
##############################################

def make_toy_data(rng_key, n=2000):
    """
    We'll produce a mixture of 2 Gaussians in 2D for demonstration.
    """
    rng = np.random.default_rng(int(rng_key[0]))
    means = np.array([[-2, 0],
                      [ 2, 0]])
    n_clusters = len(means)

    X = []
    for i in range(n):
        c = rng.integers(0, n_clusters)
        center = means[c]
        point = center + 0.6*rng.standard_normal(2)
        X.append(point)
    X = np.array(X)
    return jnp.array(X)  # shape (n,2)


##############################################
# 2. Flax "EnergyNetwork"
##############################################

class EnergyNetwork(nn.Module):
    """
    A small MLP that outputs scalar energy E(x).
    We'll define input_dim=2 for 2D data; you can adjust as needed.
    hidden_dim is the MLP dimension.
    """
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        """
        x: shape (batch, 2) if 2D data
        returns: shape (batch,) a scalar energy per sample
        """
        tanh = nn.tanh
        # MLP
        h = nn.Dense(self.hidden_dim)(x)
        h = tanh(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = tanh(h)
        # final: a single scalar per sample
        e = nn.Dense(1)(h)
        # shape (batch,1) => flatten
        return jnp.squeeze(e, axis=-1)


##############################################
# 3. EBM Training with Contrastive Divergence
##############################################

def init_model(rng_key, hidden_dim=64):
    """
    Initialize the EnergyNetwork parameters with random key.
    We'll do a dummy forward pass to get the param dictionary.
    """
    init_batch = jnp.zeros((1,2))  # (batch=1, 2D)
    model = EnergyNetwork(hidden_dim=hidden_dim)
    params = model.init(rng_key, init_batch)
    return model, params

def energy_apply(model, params, x):
    """
    Utility to apply the EnergyNetwork: E_\theta(x).
    x: shape (...,2)
    returns: shape (...,) energy
    """
    return model.apply(params, x)

def short_run_mcmc(rng_key, model, params, init_x, step_size=0.1, n_steps=20):
    """
    We'll run short-run Langevin dynamics (SGLD).
    This yields "negative samples" from our unnormalized model exp(-E).
    
    init_x: shape (batch, 2), random init
    returns final_x: shape (batch, 2)
    """
    def grad_energy(x):
        return grad(lambda z: jnp.mean(energy_apply(model, params, z)))(x)
    
    x_cur = init_x
    key = rng_key
    for i in range(n_steps):
        key, subkey = random.split(key)
        # gradient of E wrt x
        g = grad_energy(x_cur)
        # SGLD update: x_cur = x_cur - step_size*g + sqrt(2*step_size)*noise
        noise = random.normal(subkey, shape=x_cur.shape)
        x_cur = x_cur - 0.5*step_size*g + jnp.sqrt(step_size)*noise
    return x_cur

def cd_loss_fn(model, params, rng_key, x_data):
    """
    Contrastive Divergence Loss = E_\theta(x_data) - E_\theta(x_neg)
    where x_neg ~ short-run MCMC from random init.
    We'll do a batch-level average.
    """
    # data term
    e_data = energy_apply(model, params, x_data)  # shape (batch,)
    data_term = jnp.mean(e_data)

    # negative samples: random normal init
    key, subkey = random.split(rng_key)
    init_x_neg = random.normal(subkey, shape=x_data.shape)*3.0  # shape (batch,2)
    # run short-run MCMC
    key, subkey = random.split(key)
    x_neg = short_run_mcmc(subkey, model, params, init_x_neg)

    # negative term
    e_neg = energy_apply(model, params, x_neg)    # shape (batch,)
    neg_term = jnp.mean(e_neg)

    # final loss = data_term - neg_term
    # Typically we do: L = data_term - neg_term. Minimizing L => tries E(data)<E(neg)
    return data_term - neg_term

@jit
def train_step(state, rng_key, x_data):
    """
    A single step of gradient-based update for EBM with CD.
    state: train_state (holding parameters and optimizer states)
    x_data: shape (batch,2)
    returns new_state, loss
    """
    def loss_fn(params):
        return cd_loss_fn(state.apply_fn, params, rng_key, x_data)
    grads = jax.grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    loss_val = loss_fn(new_state.params)
    return new_state, loss_val

def train_ebm(rng_key, x_data, hidden_dim=64, batch_size=64, n_steps=2001, step_size=1e-3):
    """
    Full training loop with mini-batch CD. 
    We'll shuffle x_data, do random batches, etc.
    We'll keep a train_state with an optax optimizer.
    """
    # 1) init
    model, params = init_model(rng_key, hidden_dim=hidden_dim)
    tx = optax.adam(step_size)
    state = train_state.TrainState.create(
        apply_fn=model, params=params, tx=tx
    )
    n_data = x_data.shape[0]
    n_batches = n_data // batch_size
    rng = np.random.default_rng(int(rng_key[1]))  # for data shuffling

    # 2) main loop
    losses = []
    for step_i in range(n_steps):
        # shuffle each epoch if desired
        if step_i % n_batches == 0:
            indices = np.arange(n_data)
            rng.shuffle(indices)
            x_data = x_data[indices]

        # pick batch
        batch_idx = step_i % n_batches
        start_i = batch_idx*batch_size
        end_i = start_i + batch_size
        x_batch = x_data[start_i:end_i]

        # we need a new RNG for each train step
        rng_key, subkey = random.split(rng_key)
        state, loss_val = train_step(state, subkey, x_batch)
        losses.append(loss_val)

        if step_i % 500 == 0:
            print(f"[Step {step_i}] CD-Loss = {loss_val:.4f}")

    return state, jnp.array(losses)


########################################
# 4. Demo: Putting It All Together
########################################

def main():
    # 1) generate toy 2D data
    rng_key = random.PRNGKey(0)
    x_data = make_toy_data(rng_key, n=2000)  # shape (2000, 2)
    print("x_data shape:", x_data.shape)

    # 2) train EBM
    rng_key, subkey = random.split(rng_key)
    state, losses = train_ebm(subkey, x_data, hidden_dim=64, batch_size=64, n_steps=3001, step_size=1e-3)

    # 3) plot the training curve
    plt.figure()
    plt.plot(losses, label="CD-Loss")
    plt.xlabel("iteration")
    plt.ylabel("loss = E(data) - E(model)")
    plt.legend()
    plt.title("EBM Training (CD-Loss)")
    plt.show()

    # 4) visualize final samples from the EBM vs. data
    # We'll do short-run MCMC from random noise. 
    def sample_ebm(rng_key, model_state, n_samples=2000):
        init_x = random.normal(rng_key, shape=(n_samples, 2))*3.0
        # short-run:
        return short_run_mcmc(rng_key, model_state.apply_fn, model_state.params, init_x,
                              step_size=0.1, n_steps=50)

    rng_key, subkey = random.split(rng_key)
    x_samples = sample_ebm(subkey, state, n_samples=2000)  # shape (2000,2)

    # 5) plot data vs. samples
    plt.figure(figsize=(8,8))
    plt.scatter(x_data[:,0], x_data[:,1], alpha=0.3, label="Real Data", color="blue")
    plt.scatter(np.array(x_samples[:,0]), np.array(x_samples[:,1]),
                alpha=0.3, label="EBM Samples", color="red")
    plt.title("2D Data vs. EBM Samples (after training)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
