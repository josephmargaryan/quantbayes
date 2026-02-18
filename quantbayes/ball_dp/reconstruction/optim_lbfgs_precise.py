# quantbayes/ball_dp/reconstruction/optim_lbfgs_precise.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax


@dataclass(frozen=True)
class LBFGSResult:
    model: eqx.Module
    value: float
    grad_norm: float
    n_iters: int
    converged: bool
    stop_reason: str


def solve_lbfgs_precise(
    model: eqx.Module,
    objective_fn: Callable[[eqx.Module], jnp.ndarray],
    *,
    max_iters: int = 600,
    grad_tol: float = 1e-7,
    memory_size: int = 20,
    max_linesearch_steps: int = 25,
    slope_rtol: float = 1e-4,
    curv_rtol: float = 0.9,
    deterministic_objective: bool = True,
    # ---- stall safeguards ----
    min_iters: int = 25,
    stall_patience: int = 50,
    value_tol: float = 1e-12,  # relative
    stall_grad_tol: float | None = None,  # if None -> 10*grad_tol
    # ---- logging ----
    verbose: bool = False,
    print_every: int = 25,
) -> LBFGSResult:
    """
    High-precision L-BFGS solve for reconstruction experiments.

    Stops when either:
      (A) grad_norm <= grad_tol
      (B) objective has stalled for `stall_patience` steps AND grad_norm <= stall_grad_tol
      (C) max_iters reached

    This avoids "running forever" at the float32 noise floor.
    """

    params, static = eqx.partition(model, eqx.is_inexact_array)

    def f(p):
        m = eqx.combine(p, static)
        if deterministic_objective:
            m = eqx.nn.inference_mode(m, value=True)
        return objective_fn(m)

    vg = eqx.filter_jit(eqx.filter_value_and_grad(f))
    f_jit = eqx.filter_jit(f)

    linesearch = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=int(max_linesearch_steps),
        slope_rtol=float(slope_rtol),
        curv_rtol=float(curv_rtol),
        initial_guess_strategy="one",
    )
    solver = optax.lbfgs(memory_size=int(memory_size), linesearch=linesearch)
    opt_state = solver.init(params)

    if stall_grad_tol is None:
        stall_grad_tol = float(10.0 * grad_tol)

    prev_val = None
    stall = 0

    best_val = float("inf")
    best_params = params

    stop_reason = "max_iters"
    converged = False
    val = 0.0
    gnorm = float("inf")

    for it in range(1, int(max_iters) + 1):
        val_t, grad = vg(params)
        val = float(val_t)
        gnorm = float(optax.global_norm(grad))

        if val < best_val:
            best_val = val
            best_params = params

        if prev_val is not None:
            # relative stall check
            if abs(prev_val - val) <= float(value_tol) * (1.0 + abs(prev_val)):
                stall += 1
            else:
                stall = 0
        prev_val = val

        if verbose and (it == 1 or it % int(print_every) == 0 or it == int(max_iters)):
            print(
                f"[LBFGS-precise] iter={it:04d}  value={val:.6f}  grad_norm={gnorm:.3e}  stall={stall}"
            )

        # Primary convergence
        if gnorm <= float(grad_tol):
            converged = True
            stop_reason = "grad_tol"
            break

        # Stall stop: after min_iters, if not improving AND gradient already tiny-ish
        if (
            it >= int(min_iters)
            and stall_patience > 0
            and stall >= int(stall_patience)
            and gnorm <= float(stall_grad_tol)
        ):
            converged = True  # "good enough" convergence
            stop_reason = "stall"
            params = best_params
            break

        updates, opt_state = solver.update(
            grad, opt_state, params, value=val_t, grad=grad, value_fn=f_jit
        )
        params = optax.apply_updates(params, updates)

    mdl = eqx.combine(params, static)
    return LBFGSResult(
        model=mdl,
        value=float(val),
        grad_norm=float(gnorm),
        n_iters=int(it),
        converged=bool(converged),
        stop_reason=str(stop_reason),
    )


if __name__ == "__main__":
    # Smoke test: convex quadratic should converge quickly
    import numpy as np

    class Quad(eqx.Module):
        w: jnp.ndarray

        def __init__(self, d: int, *, key):
            self.w = jr.normal(key, (d,))

    rng = np.random.default_rng(0)
    d = 8
    A = jnp.eye(d) * 3.0
    b = jnp.asarray(rng.normal(size=(d,)).astype(np.float32))

    mdl0 = Quad(d, key=jr.PRNGKey(0))

    def obj(m: Quad):
        return 0.5 * (m.w @ (A @ m.w)) - (b @ m.w)

    res = solve_lbfgs_precise(mdl0, obj, max_iters=200, grad_tol=1e-9, verbose=True)
    print(
        "stop_reason:",
        res.stop_reason,
        "iters:",
        res.n_iters,
        "grad_norm:",
        res.grad_norm,
    )
    print("[OK] optim_lbfgs_precise smoke test done.")
