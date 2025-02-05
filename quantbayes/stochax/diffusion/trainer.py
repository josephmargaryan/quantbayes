# score_diffusion/training/trainer.py

import jax
import jax.random as jr
import equinox as eqx
import optax
from quantbayes.stochax.diffusion.sde import batch_loss_fn

@eqx.filter_jit
def make_step(model, weight, int_beta, data, t1, key, opt_state, opt_update):
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, weight, int_beta, data, t1, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def train_model(
    model,
    dataset,
    t1,
    lr,
    num_steps,
    batch_size,
    weight_fn,
    int_beta_fn,
    print_every,
    seed,
    *,
    data_loader_func,
):
    key = jr.PRNGKey(seed)
    model_key, loader_key = jr.split(key)

    opt = optax.adabelief(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    total_value = 0.0
    total_size = 0

    loader = data_loader_func(dataset, batch_size, key=loader_key)

    for step in range(num_steps):
        data_batch = next(loader)
        subkey, model_key = jr.split(model_key)
        loss, model, opt_state = make_step(
            model, weight_fn, int_beta_fn, data_batch, t1, subkey, opt_state, opt.update
        )
        total_value += loss.item()
        total_size += 1
        if (step % print_every) == 0 or step == (num_steps - 1):
            print(f"Step={step}, Loss={total_value/total_size}")
            total_value = 0.0
            total_size = 0

    return model
