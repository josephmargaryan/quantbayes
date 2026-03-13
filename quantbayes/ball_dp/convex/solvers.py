# quantbayes/ball_dp/convex/solvers.py
from __future__ import annotations

import dataclasses as dc
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from ..config import ConvexOptimizationConfig
from ..types import OptimizationCertificate


@dc.dataclass
class SolverResult:
    model: Any
    cert: OptimizationCertificate


def _fullbatch_objective(
    loss_fn: Callable, lam: float, x: jnp.ndarray, y: jnp.ndarray, key: jr.PRNGKey
) -> Callable[[Any], jnp.ndarray]:
    def obj(model: Any) -> jnp.ndarray:
        loss, _ = loss_fn(model, None, x, y, key, lam=lam)
        return loss

    return obj


def solve_lbfgs_fullbatch(
    model: Any,
    *,
    loss_fn: Callable,
    x: jnp.ndarray,
    y: jnp.ndarray,
    lam: float,
    key: jr.PRNGKey,
    cfg: ConvexOptimizationConfig,
) -> SolverResult:
    params, static = eqx.partition(model, eqx.is_inexact_array)
    objective = _fullbatch_objective(loss_fn, lam, x, y, key)

    def flat_obj(p):
        mdl = eqx.combine(p, static)
        return objective(mdl)

    solver = optax.lbfgs(
        memory_size=10,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=cfg.line_search_steps
        ),
    )
    opt_state = solver.init(params)
    prev_params = params
    prev_obj = float(flat_obj(params))
    converged = False
    grad_norm = float("inf")
    n_iter = 0

    for it in range(1, int(cfg.max_iter) + 1):
        value_and_grad = optax.value_and_grad_from_state(flat_obj)
        value, grad = value_and_grad(params, state=opt_state)
        leaves = jax.tree_util.tree_leaves(grad)
        grad_norm = float(
            jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in leaves if g is not None))
        )
        updates, opt_state = solver.update(
            grad, opt_state, params, value=value, grad=grad, value_fn=flat_obj
        )
        params = optax.apply_updates(params, updates)
        curr_obj = float(flat_obj(params))
        delta_obj = abs(curr_obj - prev_obj)
        delta_param = float(
            jnp.sqrt(
                sum(
                    jnp.sum(jnp.square(a - b))
                    for a, b in zip(
                        jax.tree_util.tree_leaves(params),
                        jax.tree_util.tree_leaves(prev_params),
                    )
                )
            )
        )
        prev_params = params
        prev_obj = curr_obj
        n_iter = it
        if (
            grad_norm <= cfg.grad_tol
            or delta_param <= cfg.param_tol
            or delta_obj <= cfg.objective_tol
        ):
            converged = True
            break

    param_error_bound = (
        float(grad_norm / lam)
        if cfg.certify_approximation and cfg.approximation_mode == "optimality_residual"
        else float("nan")
    )
    sensitivity_addon = (
        2.0 * param_error_bound
        if cfg.certify_approximation and cfg.approximation_mode == "optimality_residual"
        else 0.0
    )
    cert = OptimizationCertificate(
        solver="lbfgs_fullbatch",
        exact_solution=False,
        converged=converged,
        n_iter=n_iter,
        objective_value=float(prev_obj),
        grad_norm=float(grad_norm),
        parameter_error_bound=(
            float(param_error_bound) if param_error_bound == param_error_bound else 0.0
        ),
        sensitivity_addon=float(sensitivity_addon),
        notes=[
            "Deterministic iterative solver. exact_solution=False because this path does not provide a theorem-backed global certificate of exact ERM optimality.",
            "parameter_error_bound=||grad F_D(theta)||/lambda and sensitivity_addon=2*parameter_error_bound are local residual heuristics for the realized dataset only; they are not neighboring-dataset-uniform privacy certificates.",
        ],
    )
    return SolverResult(model=eqx.combine(params, static), cert=cert)


def solve_gd_fullbatch(
    model: Any,
    *,
    loss_fn: Callable,
    x: jnp.ndarray,
    y: jnp.ndarray,
    lam: float,
    key: jr.PRNGKey,
    cfg: ConvexOptimizationConfig,
) -> SolverResult:
    objective = _fullbatch_objective(loss_fn, lam, x, y, key)
    value_and_grad = eqx.filter_value_and_grad(objective)
    opt = optax.sgd(cfg.learning_rate)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = opt.init(params)
    prev_params = params
    prev_obj = float(objective(model))
    grad_norm = float("inf")
    converged = False
    n_iter = 0
    for it in range(1, int(cfg.max_iter) + 1):
        mdl = eqx.combine(params, static)
        value, grads = value_and_grad(mdl)
        leaves = jax.tree_util.tree_leaves(grads)
        grad_norm = float(
            jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in leaves if g is not None))
        )
        updates, opt_state = opt.update(
            eqx.filter(grads, eqx.is_inexact_array), opt_state, params
        )
        params = eqx.apply_updates(params, updates)
        curr_obj = float(objective(eqx.combine(params, static)))
        delta_obj = abs(curr_obj - prev_obj)
        delta_param = float(
            jnp.sqrt(
                sum(
                    jnp.sum(jnp.square(a - b))
                    for a, b in zip(
                        jax.tree_util.tree_leaves(params),
                        jax.tree_util.tree_leaves(prev_params),
                    )
                )
            )
        )
        prev_params = params
        prev_obj = curr_obj
        n_iter = it
        if (
            grad_norm <= cfg.grad_tol
            or delta_param <= cfg.param_tol
            or delta_obj <= cfg.objective_tol
        ):
            converged = True
            break
    param_error_bound = (
        float(grad_norm / lam)
        if cfg.certify_approximation and cfg.approximation_mode == "optimality_residual"
        else 0.0
    )
    cert = OptimizationCertificate(
        solver="gd_fullbatch",
        exact_solution=False,
        converged=converged,
        n_iter=n_iter,
        objective_value=float(prev_obj),
        grad_norm=float(grad_norm),
        parameter_error_bound=float(param_error_bound),
        sensitivity_addon=float(2.0 * param_error_bound),
        notes=[
            "Deterministic iterative solver. exact_solution=False because this path does not provide a theorem-backed global certificate of exact ERM optimality.",
            "parameter_error_bound=||grad F_D(theta)||/lambda and sensitivity_addon=2*parameter_error_bound are local residual heuristics for the realized dataset only; they are not neighboring-dataset-uniform privacy certificates.",
        ],
    )
    return SolverResult(model=eqx.combine(params, static), cert=cert)


def solve_convex_model(
    model: Any,
    *,
    loss_fn: Callable,
    x: jnp.ndarray,
    y: jnp.ndarray,
    lam: float,
    key: jr.PRNGKey,
    cfg: ConvexOptimizationConfig,
) -> SolverResult:
    if cfg.solver == "lbfgs_fullbatch":
        return solve_lbfgs_fullbatch(
            model, loss_fn=loss_fn, x=x, y=y, lam=lam, key=key, cfg=cfg
        )
    if cfg.solver == "gd_fullbatch":
        return solve_gd_fullbatch(
            model, loss_fn=loss_fn, x=x, y=y, lam=lam, key=key, cfg=cfg
        )
    if cfg.theorem_backed_only:
        raise ValueError(
            f"Solver '{cfg.solver}' delegates to existing trainers and is not treated as theorem-backed by default. "
            "Set theorem_backed_only=False if you want the convenience adapter."
        )
    if cfg.solver == "stochax_lbfgs":
        from quantbayes.stochax.trainer.quasi_newton import train_lbfgs

        solved, _, _, _ = train_lbfgs(
            model,
            None,
            x,
            y,
            x,
            y,
            batch_size=int(x.shape[0]),
            num_epochs=int(cfg.max_iter),
            patience=int(cfg.max_iter),
            key=key,
            loss_fn=lambda m, s, xb, yb, k: loss_fn(m, s, xb, yb, k, lam=lam),
            deterministic_objective=True,
        )
        cert = OptimizationCertificate(
            solver="stochax_lbfgs",
            exact_solution=False,
            converged=False,
            n_iter=int(cfg.max_iter),
            objective_value=float(
                _fullbatch_objective(loss_fn, lam, x, y, key)(solved)
            ),
            grad_norm=float("nan"),
            parameter_error_bound=0.0,
            sensitivity_addon=0.0,
            notes=[
                "Convenience adapter to an external trainer. This path is not theorem-backed and does not provide a global optimality or privacy certificate."
            ],
        )
        return SolverResult(model=solved, cert=cert)
    raise ValueError(cfg.solver)
