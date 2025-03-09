from typing import Union

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
import pandas as pd


def lipswish(x):
    return 0.909 * jnn.silu(x)


class VectorField(eqx.Module):
    scale: Union[int, jnp.ndarray]
    mlp: eqx.nn.MLP

    def __init__(self, hidden_size, width_size, depth, scale, *, key, **kwargs):
        super().__init__(**kwargs)
        scale_key, mlp_key = jr.split(key)
        if scale:
            self.scale = jr.uniform(scale_key, (hidden_size,), minval=0.9, maxval=1.1)
        else:
            self.scale = 1
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + 1,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=mlp_key,
        )

    def __call__(self, t, y, args):
        t = jnp.asarray(t)
        return self.scale * self.mlp(jnp.concatenate([t[None], y]))


class ControlledVectorField(eqx.Module):
    scale: Union[int, jnp.ndarray]
    mlp: eqx.nn.MLP
    control_size: int
    hidden_size: int

    def __init__(
        self, control_size, hidden_size, width_size, depth, scale, *, key, **kwargs
    ):
        super().__init__(**kwargs)
        scale_key, mlp_key = jr.split(key)
        if scale:
            self.scale = jr.uniform(
                scale_key, (hidden_size, control_size), minval=0.9, maxval=1.1
            )
        else:
            self.scale = 1
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + 1,
            out_size=hidden_size * control_size,
            width_size=width_size,
            depth=depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=mlp_key,
        )
        self.control_size = control_size
        self.hidden_size = hidden_size

    def __call__(self, t, y, args):
        t = jnp.asarray(t)
        return self.scale * self.mlp(jnp.concatenate([t[None], y])).reshape(
            self.hidden_size, self.control_size
        )


class NeuralSDE(eqx.Module):
    initial: eqx.nn.MLP
    vf: VectorField  # drift
    cvf: ControlledVectorField  # diffusion
    readout: eqx.nn.Linear
    initial_noise_size: int
    noise_size: int

    def __init__(
        self,
        data_size,
        initial_noise_size,
        noise_size,
        hidden_size,
        width_size,
        depth,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        initial_key, vf_key, cvf_key, readout_key = jr.split(key, 4)

        self.initial = eqx.nn.MLP(
            initial_noise_size, hidden_size, width_size, depth, key=initial_key
        )
        self.vf = VectorField(hidden_size, width_size, depth, scale=True, key=vf_key)
        self.cvf = ControlledVectorField(
            noise_size, hidden_size, width_size, depth, scale=True, key=cvf_key
        )
        self.readout = eqx.nn.Linear(hidden_size, data_size, key=readout_key)

        self.initial_noise_size = initial_noise_size
        self.noise_size = noise_size

    def __call__(self, ts, *, key):
        t0 = ts[0]
        t1 = ts[-1]
        # Very large dt0 for computational speed
        dt0 = 1.0
        init_key, bm_key = jr.split(key, 2)
        init = jr.normal(init_key, (self.initial_noise_size,))
        control = diffrax.VirtualBrownianTree(
            t0=t0, t1=t1, tol=dt0 / 2, shape=(self.noise_size,), key=bm_key
        )
        vf = diffrax.ODETerm(self.vf)  # Drift term
        cvf = diffrax.ControlTerm(self.cvf, control)  # Diffusion term
        terms = diffrax.MultiTerm(vf, cvf)
        # ReversibleHeun is a cheap choice of SDE solver. We could also use Euler etc.
        solver = diffrax.ReversibleHeun()
        y0 = self.initial(init)
        saveat = diffrax.SaveAt(ts=ts)
        sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0, y0, saveat=saveat)
        return jax.vmap(self.readout)(sol.ys)


class NeuralCDE(eqx.Module):
    initial: eqx.nn.MLP
    vf: VectorField
    cvf: ControlledVectorField
    readout: eqx.nn.Linear

    def __init__(self, data_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        initial_key, vf_key, cvf_key, readout_key = jr.split(key, 4)

        self.initial = eqx.nn.MLP(
            data_size + 1, hidden_size, width_size, depth, key=initial_key
        )
        self.vf = VectorField(hidden_size, width_size, depth, scale=False, key=vf_key)
        self.cvf = ControlledVectorField(
            data_size, hidden_size, width_size, depth, scale=False, key=cvf_key
        )
        self.readout = eqx.nn.Linear(hidden_size, 1, key=readout_key)

    def __call__(self, ts, ys):
        # Interpolate data into a continuous path.
        ys = diffrax.linear_interpolation(
            ts, ys, replace_nans_at_start=0.0, fill_forward_nans_at_end=True
        )
        init = jnp.concatenate([ts[0, None], ys[0]])
        control = diffrax.LinearInterpolation(ts, ys)
        vf = diffrax.ODETerm(self.vf)
        cvf = diffrax.ControlTerm(self.cvf, control)
        terms = diffrax.MultiTerm(vf, cvf)
        solver = diffrax.ReversibleHeun()
        t0 = ts[0]
        t1 = ts[-1]
        dt0 = 1.0
        y0 = self.initial(init)
        # Have the discriminator produce an output at both `t0` *and* `t1`.
        # The output at `t0` has only seen the initial point of a sample. This gives
        # additional supervision to the distribution learnt for the initial condition.
        # The output at `t1` has seen the entire path of a sample. This is needed to
        # actually learn the evolving trajectory.
        saveat = diffrax.SaveAt(t0=True, t1=True)
        sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0, y0, saveat=saveat)
        return jax.vmap(self.readout)(sol.ys)

    @eqx.filter_jit
    def clip_weights(self):
        leaves, treedef = jax.tree_util.tree_flatten(
            self, is_leaf=lambda x: isinstance(x, eqx.nn.Linear)
        )
        new_leaves = []
        for leaf in leaves:
            if isinstance(leaf, eqx.nn.Linear):
                lim = 1 / leaf.out_features
                leaf = eqx.tree_at(
                    lambda x: x.weight, leaf, leaf.weight.clip(-lim, lim)
                )
            new_leaves.append(leaf)
        return jax.tree_util.tree_unflatten(treedef, new_leaves)


@jax.jit
@jax.vmap
def get_data(key):
    bm_key, y0_key, drop_key = jr.split(key, 3)

    mu = 0.02
    theta = 0.1
    sigma = 0.4

    t0 = 0
    t1 = 63
    t_size = 64

    def drift(t, y, args):
        return mu * t - theta * y

    def diffusion(t, y, args):
        return 2 * sigma * t / t1

    bm = diffrax.UnsafeBrownianPath(shape=(), key=bm_key)
    drift = diffrax.ODETerm(drift)
    diffusion = diffrax.ControlTerm(diffusion, bm)
    terms = diffrax.MultiTerm(drift, diffusion)
    solver = diffrax.Euler()
    dt0 = 0.1
    y0 = jr.uniform(y0_key, (1,), minval=-1, maxval=1)
    ts = jnp.linspace(t0, t1, t_size)
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        terms, solver, t0, t1, dt0, y0, saveat=saveat, adjoint=diffrax.DirectAdjoint()
    )

    # Make the data irregularly sampled
    to_drop = jr.bernoulli(drop_key, 0.3, (t_size, 1))
    ys = jnp.where(to_drop, jnp.nan, sol.ys)

    return ts, ys


def dataloader(arrays, batch_size, loop, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(
        array.shape[0] == dataset_size for array in arrays
    ), "All arrays must have the same dataset size along the first axis."
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        key = jr.split(key, 1)[0]
        start = 0
        end = batch_size
        while end <= dataset_size:  # Change < to <= to handle edge cases
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size
        if not loop:
            break


@eqx.filter_jit
def loss(generator, discriminator, ts_i, ys_i, key, step=0):
    batch_size, _ = ts_i.shape
    key = jr.fold_in(key, step)
    key = jr.split(key, batch_size)
    fake_ys_i = jax.vmap(generator)(ts_i, key=key)
    real_score = jax.vmap(discriminator)(ts_i, ys_i)
    fake_score = jax.vmap(discriminator)(ts_i, fake_ys_i)
    return jnp.mean(real_score - fake_score)


@eqx.filter_grad
def grad_loss(g_d, ts_i, ys_i, key, step):
    generator, discriminator = g_d
    return loss(generator, discriminator, ts_i, ys_i, key, step)


def increase_update_initial(updates):
    get_initial_leaves = lambda u: jax.tree_util.tree_leaves(u.initial)
    return eqx.tree_at(get_initial_leaves, updates, replace_fn=lambda x: x * 10)


@eqx.filter_jit
def make_step(
    generator,
    discriminator,
    g_opt_state,
    d_opt_state,
    g_optim,
    d_optim,
    ts_i,
    ys_i,
    key,
    step,
):
    g_grad, d_grad = grad_loss((generator, discriminator), ts_i, ys_i, key, step)
    g_updates, g_opt_state = g_optim.update(g_grad, g_opt_state)
    d_updates, d_opt_state = d_optim.update(d_grad, d_opt_state)
    g_updates = increase_update_initial(g_updates)
    d_updates = increase_update_initial(d_updates)
    generator = eqx.apply_updates(generator, g_updates)
    discriminator = eqx.apply_updates(discriminator, d_updates)
    discriminator = discriminator.clip_weights()
    return generator, discriminator, g_opt_state, d_opt_state


def plot_trajectories(df, num_trajectories=30):
    """
    Visualize original and generated trajectories.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Time', 'Original', and 'Generated' columns.
        num_trajectories (int): Max number of trajectories to plot.
    """
    # Reshape 'Original' and 'Generated' to get individual trajectories
    time_steps = df["Time"].unique()
    num_time_steps = len(time_steps)
    total_trajectories = len(df) // num_time_steps

    # Reshape data into (trajectories, time_steps)
    original_trajectories = df["Original"].values.reshape(
        total_trajectories, num_time_steps
    )
    generated_trajectories = df["Generated"].values.reshape(
        total_trajectories, num_time_steps
    )

    # Plot a subset of trajectories
    plt.figure(figsize=(10, 6))
    for i in range(min(num_trajectories, total_trajectories)):
        plt.plot(
            time_steps,
            original_trajectories[i],
            label="Original" if i == 0 else "",
            color="dodgerblue",
            linewidth=0.5,
            alpha=0.7,
        )
        plt.plot(
            time_steps,
            generated_trajectories[i],
            label="Generated" if i == 0 else "",
            color="crimson",
            linewidth=0.5,
            alpha=0.7,
        )

    plt.title(f"Original vs. Generated Trajectories (max {num_trajectories})")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.show()


def run_sde(
    real_data,  # Required parameter: Tuple of (ts, ys)
    initial_noise_size=5,
    noise_size=3,
    hidden_size=16,
    width_size=16,
    depth=1,
    generator_lr=2e-5,
    discriminator_lr=1e-4,
    batch_size=1,  # Adjust to match dataset size
    steps=10,
    steps_per_print=1,
    seed=5678,
    max_plot_trajectories=50,  # Limit number of trajectories in the plot
):
    """
    Runs the Stochastic Differential equation as the Generator in the Generative Adversarial Network
    With correponding Controlled Differential Equation as the discriminator

    :param real_data, Tuple of (ts, ys) where ts has shape (num_sequences, num_timesteps) and ys has shape (num_sequences, num_timesteps, 1)
    """
    if real_data is None:
        raise ValueError("You must provide real-world data as (ts, ys).")

    ts, ys = real_data
    assert ts.shape[0] == ys.shape[0], "Time and observation dimensions must match!"

    key = jr.PRNGKey(seed)
    (
        generator_key,
        discriminator_key,
        dataloader_key,
        evaluate_key,
        sample_key,
    ) = jr.split(key, 5)

    _, _, data_size = ys.shape

    generator = NeuralSDE(
        data_size,
        initial_noise_size,
        noise_size,
        hidden_size,
        width_size,
        depth,
        key=generator_key,
    )
    discriminator = NeuralCDE(
        data_size, hidden_size, width_size, depth, key=discriminator_key
    )

    g_optim = optax.rmsprop(generator_lr)
    d_optim = optax.rmsprop(-discriminator_lr)
    g_opt_state = g_optim.init(eqx.filter(generator, eqx.is_inexact_array))
    d_opt_state = d_optim.init(eqx.filter(discriminator, eqx.is_inexact_array))

    infinite_dataloader = dataloader(
        (ts, ys), batch_size, loop=True, key=dataloader_key
    )

    print("Starting training loop...")
    for step, (ts_i, ys_i) in zip(range(steps), infinite_dataloader):
        generator, discriminator, g_opt_state, d_opt_state = make_step(
            generator,
            discriminator,
            g_opt_state,
            d_opt_state,
            g_optim,
            d_optim,
            ts_i,
            ys_i,
            key,
            step,
        )
        if (step % steps_per_print) == 0 or step == steps - 1:
            total_score = 0
            num_batches = 0
            for ts_i, ys_i in dataloader(
                (ts, ys), batch_size, loop=False, key=evaluate_key
            ):
                score = loss(generator, discriminator, ts_i, ys_i, sample_key)
                total_score += score.item()
                num_batches += 1
            print(f"Step {step}: Loss = {total_score / num_batches}")

    print("Training complete. Generating samples...")

    # Generate and return a DataFrame
    num_samples = ts.shape[0]
    ts_to_plot = ts[:num_samples]
    ys_generated = jax.vmap(generator)(
        ts_to_plot, key=jr.split(sample_key, num_samples)
    )[..., 0]

    generated_df = pd.DataFrame(
        {
            "Time": ts_to_plot.flatten(),
            "Generated": ys_generated.flatten(),
        }
    )

    # Add original data to the DataFrame
    original_df = pd.DataFrame(
        {
            "Time": ts.flatten(),
            "Original": ys.flatten(),
        }
    )

    # Ensure column names are unique and descriptive
    combined_df = pd.concat(
        [original_df.set_index("Time"), generated_df.set_index("Time")], axis=1
    ).reset_index()

    plot_trajectories(combined_df, num_trajectories=max_plot_trajectories)

    return combined_df
