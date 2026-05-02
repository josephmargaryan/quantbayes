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


def _validate_solver_lam(lam: float) -> None:
    lam_f = float(lam)
    if not bool(jnp.isfinite(jnp.asarray(lam_f))) or lam_f <= 0.0:
        raise ValueError(
            "Convex solvers require lam > 0 for the strong-convexity-based "
            "optimization certificates used downstream."
        )


def _fullbatch_objective(
    loss_fn: Callable,
    lam: float,
    x: jnp.ndarray,
    y: jnp.ndarray,
    key: jr.PRNGKey,
) -> Callable[[Any], jnp.ndarray]:
    def obj(model: Any) -> jnp.ndarray:
        loss, _ = loss_fn(model, None, x, y, key, lam=lam)
        return loss

    return obj


def _tree_l2_norm(tree: Any) -> float:
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(tree) if leaf is not None]
    if not leaves:
        return 0.0
    sq = sum(jnp.sum(jnp.square(jnp.asarray(leaf))) for leaf in leaves)
    return float(jnp.sqrt(sq))


def _tree_diff_l2_norm(tree_a: Any, tree_b: Any) -> float:
    leaves_a = jax.tree_util.tree_leaves(tree_a)
    leaves_b = jax.tree_util.tree_leaves(tree_b)

    if len(leaves_a) != len(leaves_b):
        raise ValueError("Pytrees do not have the same number of leaves.")

    sq_terms = []
    for a, b in zip(leaves_a, leaves_b):
        if a is None or b is None:
            continue
        sq_terms.append(jnp.sum(jnp.square(jnp.asarray(a) - jnp.asarray(b))))

    if not sq_terms:
        return 0.0

    return float(jnp.sqrt(sum(sq_terms)))


def _should_stop(
    cfg: ConvexOptimizationConfig,
    *,
    it: int,
    grad_norm: float,
    delta_param: float,
    delta_obj: float,
) -> tuple[bool, str]:
    if not bool(cfg.early_stop):
        return False, ""

    if int(it) < int(cfg.min_iter):
        return False, ""

    rule = str(cfg.stop_rule).lower()

    grad_hit = False if cfg.grad_tol is None else grad_norm <= float(cfg.grad_tol)
    param_hit = False if cfg.param_tol is None else delta_param <= float(cfg.param_tol)
    obj_hit = (
        False if cfg.objective_tol is None else delta_obj <= float(cfg.objective_tol)
    )

    if rule == "grad_only":
        if cfg.grad_tol is None:
            raise ValueError("stop_rule='grad_only' requires grad_tol to be set.")
        return grad_hit, "grad_tol" if grad_hit else ""

    active = []
    if cfg.grad_tol is not None:
        active.append(("grad_tol", grad_hit))
    if cfg.param_tol is not None:
        active.append(("param_tol", param_hit))
    if cfg.objective_tol is not None:
        active.append(("objective_tol", obj_hit))

    if not active:
        return False, ""

    if rule == "any":
        for name, hit in active:
            if hit:
                return True, name
        return False, ""

    if rule == "all":
        hit = all(flag for _, flag in active)
        return hit, "all_active_tolerances" if hit else ""

    raise ValueError("cfg.stop_rule must be one of {'any', 'all', 'grad_only'}.")


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
    _validate_solver_lam(lam)

    params, static = eqx.partition(model, eqx.is_inexact_array)
    objective = _fullbatch_objective(loss_fn, lam, x, y, key)

    def flat_obj(p):
        mdl = eqx.combine(p, static)
        return objective(mdl)

    solver = optax.lbfgs(
        memory_size=10,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=int(cfg.line_search_steps)
        ),
    )
    value_and_grad = optax.value_and_grad_from_state(flat_obj)
    opt_state = solver.init(params)

    prev_params = params
    prev_obj = float(flat_obj(params))

    converged = False
    termination_reason = "max_iter"
    grad_norm = float("inf")
    delta_param = float("inf")
    delta_obj = float("inf")
    n_iter = 0

    for it in range(1, int(cfg.max_iter) + 1):
        value, grad = value_and_grad(params, state=opt_state)

        updates, opt_state = solver.update(
            grad,
            opt_state,
            params,
            value=value,
            grad=grad,
            value_fn=flat_obj,
        )
        params_next = optax.apply_updates(params, updates)

        curr_obj, curr_grad = jax.value_and_grad(flat_obj)(params_next)
        curr_obj_f = float(curr_obj)

        grad_norm = _tree_l2_norm(curr_grad)
        delta_obj = abs(curr_obj_f - prev_obj)
        delta_param = _tree_diff_l2_norm(params_next, prev_params)

        params = params_next
        prev_params = params
        prev_obj = curr_obj_f
        n_iter = it

        stop, reason = _should_stop(
            cfg,
            it=it,
            grad_norm=grad_norm,
            delta_param=delta_param,
            delta_obj=delta_obj,
        )
        if stop:
            converged = True
            termination_reason = reason
            break

    final_obj, final_grad = jax.value_and_grad(flat_obj)(params)
    final_obj_f = float(final_obj)
    grad_norm = _tree_l2_norm(final_grad)

    if (
        bool(cfg.certify_approximation)
        and cfg.approximation_mode == "optimality_residual"
    ):
        param_error_bound = float(grad_norm / float(lam))
        sensitivity_addon = float(2.0 * param_error_bound)
    else:
        param_error_bound = 0.0
        sensitivity_addon = 0.0

    cert = OptimizationCertificate(
        solver="lbfgs_fullbatch",
        exact_solution=False,
        converged=converged,
        n_iter=int(n_iter),
        objective_value=float(final_obj_f),
        grad_norm=float(grad_norm),
        parameter_error_bound=float(param_error_bound),
        sensitivity_addon=float(sensitivity_addon),
        termination_reason=str(termination_reason),
        final_delta_param=float(delta_param),
        final_delta_obj=float(delta_obj),
        notes=[
            "Deterministic iterative solver. exact_solution=False because this path does not provide a theorem-backed global certificate of exact ERM optimality.",
            "grad_norm and parameter_error_bound are evaluated at the returned parameters.",
            "parameter_error_bound=||grad F_D(theta)||/lambda and sensitivity_addon=2*parameter_error_bound are local residual bounds for the realized dataset only; they are not neighboring-dataset-uniform privacy certificates.",
            f"Stopping configuration: early_stop={bool(cfg.early_stop)}, stop_rule={str(cfg.stop_rule)}, min_iter={int(cfg.min_iter)}, grad_tol={cfg.grad_tol}, param_tol={cfg.param_tol}, objective_tol={cfg.objective_tol}.",
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
    _validate_solver_lam(lam)

    objective = _fullbatch_objective(loss_fn, lam, x, y, key)

    params, static = eqx.partition(model, eqx.is_inexact_array)

    def flat_obj(p):
        mdl = eqx.combine(p, static)
        return objective(mdl)

    value_and_grad = eqx.filter_value_and_grad(objective)

    opt = optax.sgd(float(cfg.learning_rate))
    opt_state = opt.init(params)

    prev_params = params
    prev_obj = float(objective(model))

    grad_norm = float("inf")
    delta_param = float("inf")
    delta_obj = float("inf")
    converged = False
    termination_reason = "max_iter"
    n_iter = 0

    for it in range(1, int(cfg.max_iter) + 1):
        mdl = eqx.combine(params, static)
        _, grads = value_and_grad(mdl)

        updates, opt_state = opt.update(
            eqx.filter(grads, eqx.is_inexact_array),
            opt_state,
            params,
        )
        params_next = eqx.apply_updates(params, updates)

        curr_obj, curr_grad = jax.value_and_grad(flat_obj)(params_next)
        curr_obj_f = float(curr_obj)

        grad_norm = _tree_l2_norm(curr_grad)
        delta_obj = abs(curr_obj_f - prev_obj)
        delta_param = _tree_diff_l2_norm(params_next, prev_params)

        params = params_next
        prev_params = params
        prev_obj = curr_obj_f
        n_iter = it

        stop, reason = _should_stop(
            cfg,
            it=it,
            grad_norm=grad_norm,
            delta_param=delta_param,
            delta_obj=delta_obj,
        )
        if stop:
            converged = True
            termination_reason = reason
            break

    final_obj, final_grad = jax.value_and_grad(flat_obj)(params)
    final_obj_f = float(final_obj)
    grad_norm = _tree_l2_norm(final_grad)

    if (
        bool(cfg.certify_approximation)
        and cfg.approximation_mode == "optimality_residual"
    ):
        param_error_bound = float(grad_norm / float(lam))
        sensitivity_addon = float(2.0 * param_error_bound)
    else:
        param_error_bound = 0.0
        sensitivity_addon = 0.0

    cert = OptimizationCertificate(
        solver="gd_fullbatch",
        exact_solution=False,
        converged=converged,
        n_iter=int(n_iter),
        objective_value=float(final_obj_f),
        grad_norm=float(grad_norm),
        parameter_error_bound=float(param_error_bound),
        sensitivity_addon=float(sensitivity_addon),
        termination_reason=str(termination_reason),
        final_delta_param=float(delta_param),
        final_delta_obj=float(delta_obj),
        notes=[
            "Deterministic iterative solver. exact_solution=False because this path does not provide a theorem-backed global certificate of exact ERM optimality.",
            "grad_norm and parameter_error_bound are evaluated at the returned parameters.",
            "parameter_error_bound=||grad F_D(theta)||/lambda and sensitivity_addon=2*parameter_error_bound are local residual bounds for the realized dataset only; they are not neighboring-dataset-uniform privacy certificates.",
            f"Stopping configuration: early_stop={bool(cfg.early_stop)}, stop_rule={str(cfg.stop_rule)}, min_iter={int(cfg.min_iter)}, grad_tol={cfg.grad_tol}, param_tol={cfg.param_tol}, objective_tol={cfg.objective_tol}.",
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
    _validate_solver_lam(lam)

    if cfg.solver == "lbfgs_fullbatch":
        return solve_lbfgs_fullbatch(
            model,
            loss_fn=loss_fn,
            x=x,
            y=y,
            lam=lam,
            key=key,
            cfg=cfg,
        )

    if cfg.solver == "gd_fullbatch":
        return solve_gd_fullbatch(
            model,
            loss_fn=loss_fn,
            x=x,
            y=y,
            lam=lam,
            key=key,
            cfg=cfg,
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
            termination_reason="external_trainer",
            final_delta_param=float("nan"),
            final_delta_obj=float("nan"),
            notes=[
                "Convenience adapter to an external trainer. This path is not theorem-backed and does not provide a global optimality or privacy certificate."
            ],
        )

        return SolverResult(model=solved, cert=cert)

    raise ValueError(cfg.solver)
