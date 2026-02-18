# quantbayes/ball_dp/reconstruction/optim_lbfgs_array.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import optax
import jax.numpy as jnp


@dataclass(frozen=True)
class LBFGSArrayResult:
    theta: jnp.ndarray
    value: float
    grad_norm: float
    n_iters: int
    converged: bool
    stop_reason: str


def solve_lbfgs_array(
    theta0: jnp.ndarray,
    value_and_grad_fn: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
    value_fn: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    max_iters: int = 200,
    grad_tol: float = 1e-7,
    memory_size: int = 20,
    max_linesearch_steps: int = 25,
    slope_rtol: float = 1e-4,
    curv_rtol: float = 0.9,
    # stall safeguard
    min_iters: int = 10,
    stall_patience: int = 30,
    value_tol: float = 1e-12,
    stall_grad_tol: Optional[float] = None,
    # logging
    verbose: bool = False,
    print_every: int = 10,
) -> LBFGSArrayResult:
    """
    L-BFGS on a flat parameter vector with stall safeguards.

    IMPORTANT: value_and_grad_fn and value_fn should already be JIT-compiled and
    must NOT be recreated per target. They can close over X,y but should call a
    module-level jitted function internally.
    """
    if stall_grad_tol is None:
        stall_grad_tol = float(10.0 * grad_tol)

    theta = theta0
    solver = optax.lbfgs(
        memory_size=int(memory_size),
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=int(max_linesearch_steps),
            slope_rtol=float(slope_rtol),
            curv_rtol=float(curv_rtol),
            initial_guess_strategy="one",
        ),
    )
    opt_state = solver.init(theta)

    prev_val = None
    stall = 0
    best_val = jnp.inf
    best_theta = theta

    stop_reason = "max_iters"
    converged = False
    val = jnp.asarray(0.0, dtype=theta.dtype)
    gnorm = jnp.asarray(jnp.inf, dtype=theta.dtype)

    for it in range(1, int(max_iters) + 1):
        val, grad = value_and_grad_fn(theta)
        gnorm = optax.global_norm(grad)

        if val < best_val:
            best_val = val
            best_theta = theta

        if prev_val is not None:
            if jnp.abs(prev_val - val) <= float(value_tol) * (1.0 + jnp.abs(prev_val)):
                stall += 1
            else:
                stall = 0
        prev_val = val

        if verbose and (it == 1 or it % int(print_every) == 0 or it == int(max_iters)):
            print(
                f"[LBFGS-array] iter={it:04d}  value={float(val):.6f}  "
                f"grad_norm={float(gnorm):.3e}  stall={stall}",
                flush=True,
            )

        if float(gnorm) <= float(grad_tol):
            converged = True
            stop_reason = "grad_tol"
            break

        if (
            it >= int(min_iters)
            and stall_patience > 0
            and stall >= int(stall_patience)
            and float(gnorm) <= float(stall_grad_tol)
        ):
            converged = True
            stop_reason = "stall"
            theta = best_theta
            val = best_val
            break

        updates, opt_state = solver.update(
            grad, opt_state, theta, value=val, grad=grad, value_fn=value_fn
        )
        theta = optax.apply_updates(theta, updates)

    return LBFGSArrayResult(
        theta=theta,
        value=float(val),
        grad_norm=float(gnorm),
        n_iters=int(it),
        converged=bool(converged),
        stop_reason=str(stop_reason),
    )


if __name__ == "__main__":
    import jax
    import numpy as np

    rng = np.random.default_rng(0)
    d = 8
    A = jnp.eye(d) * 3.0
    b = jnp.asarray(rng.normal(size=(d,)).astype(np.float32))

    def obj(theta):
        return 0.5 * theta @ (A @ theta) - b @ theta

    obj_j = jax.jit(obj)
    vg_j = jax.jit(jax.value_and_grad(obj))

    theta0 = jnp.zeros((d,), dtype=jnp.float32)

    res = solve_lbfgs_array(
        theta0,
        lambda th: vg_j(th),
        lambda th: obj_j(th),
        max_iters=200,
        grad_tol=1e-9,
        verbose=True,
        print_every=25,
    )
    print(res)
    print("[OK] optim_lbfgs_array smoke.")
