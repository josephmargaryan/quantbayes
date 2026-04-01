from __future__ import annotations

import math
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax


Array = jax.Array


class BallTanhOpNet(eqx.Module):
    """One-hidden-layer tanh MLP with theorem-backed operator-norm control:
         f(x) = (1/sqrt(H)) * a^T tanh(Wx + b)

    Hidden bias is allowed; output bias is omitted.

    This is the dense operator-norm baseline:
      - hidden layer is a standard dense layer
      - theorem/control uses ||W||_op <= Lambda
      - projection onto the operator-norm ball is exact via SVD
    """

    hidden: eqx.nn.Linear
    out: eqx.nn.Linear
    hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        d_in: int,
        hidden_dim: int,
        *,
        key: Array,
        dtype=None,
    ):
        k1, k2 = jr.split(key, 2)
        self.hidden = eqx.nn.Linear(
            d_in,
            hidden_dim,
            use_bias=True,
            dtype=dtype,
            key=k1,
        )
        self.out = eqx.nn.Linear(
            hidden_dim,
            1,
            use_bias=False,
            dtype=dtype,
            key=k2,
        )
        self.hidden_dim = int(hidden_dim)

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        key: Array | None = None,
        state: Any = None,
    ):
        del key
        h = jnp.tanh(self.hidden(jnp.asarray(x)))
        logit = self.out(h)[0] / math.sqrt(self.hidden_dim)
        return logit, state


def make_ball_tanh_op_net(
    d_in: int,
    hidden_dim: int,
    *,
    key: Array,
    dtype=None,
    init_project: bool = False,
    Lambda: float | None = None,
    A: float | None = None,
) -> BallTanhOpNet:
    model = BallTanhOpNet(
        d_in=d_in,
        hidden_dim=hidden_dim,
        key=key,
        dtype=dtype,
    )
    if init_project:
        if Lambda is None or A is None:
            raise ValueError("init_project=True requires both Lambda and A.")
        model = project_ball_tanh_op_params(model, Lambda=float(Lambda), A=float(A))
    return model


def tanh_kappa() -> float:
    return 4.0 / (3.0 * math.sqrt(3.0))


def certified_tanh_mlp_op_constants(
    *,
    A: float,
    Lambda: float,
    B: float,
    H: int,
) -> dict[str, float]:
    """Theorem-backed constants for the dense operator-norm tanh MLP."""
    if A < 0.0 or Lambda < 0.0 or B < 0.0:
        raise ValueError("A, Lambda, B must be nonnegative.")
    if H <= 0:
        raise ValueError("H must be positive.")

    kappa = tanh_kappa()
    sqrt_H = math.sqrt(float(H))

    L_f = (
        1.0
        / sqrt_H
        * math.sqrt(
            Lambda * Lambda
            + (A * kappa * Lambda) ** 2
            + (A * (kappa * Lambda * B + 1.0)) ** 2
        )
    )
    G_f = math.sqrt(1.0 + (A * A * (1.0 + B * B)) / float(H))
    L_z = L_f + (A * Lambda / (4.0 * sqrt_H)) * G_f

    return {
        "kappa_tanh": kappa,
        "L_f": L_f,
        "G_f": G_f,
        "L_z": L_z,
    }


def certified_tanh_mlp_op_lz(
    *,
    A: float,
    Lambda: float,
    B: float,
    H: int,
) -> float:
    return certified_tanh_mlp_op_constants(
        A=A,
        Lambda=Lambda,
        B=B,
        H=H,
    )["L_z"]


def _top_singular_value(weight: jnp.ndarray) -> jnp.ndarray:
    s = jnp.linalg.svd(jnp.asarray(weight), full_matrices=False, compute_uv=False)
    return s[0]


def _project_spectral(weight: jnp.ndarray, radius: float) -> jnp.ndarray:
    """Exact Frobenius projection onto the operator-norm ball {||W||_op <= radius}."""
    if not math.isfinite(radius):
        return weight

    U, s, Vh = jnp.linalg.svd(jnp.asarray(weight), full_matrices=False)
    bound = jnp.asarray(radius, dtype=s.dtype)
    s_proj = jnp.minimum(s, bound)
    out = U @ (s_proj[:, None] * Vh)
    return out.astype(weight.dtype)


def _project_l2(vec: jnp.ndarray, radius: float) -> jnp.ndarray:
    if not math.isfinite(radius):
        return vec
    norm = jnp.linalg.norm(vec)
    scale = jnp.minimum(
        jnp.asarray(1.0, dtype=vec.dtype),
        jnp.asarray(radius, dtype=vec.dtype)
        / jnp.maximum(norm, jnp.asarray(1e-12, dtype=vec.dtype)),
    )
    return vec * scale


def project_ball_tanh_op_params(
    params: BallTanhOpNet,
    *,
    Lambda: float,
    A: float,
) -> BallTanhOpNet:
    """Project the theorem-backed trainable parameters.

    Projects:
      - hidden weight onto the operator-norm ball of radius Lambda
      - output head onto the l2 ball of radius A
    """
    if Lambda < 0.0 or A < 0.0:
        raise ValueError("Lambda and A must be nonnegative.")

    new_hidden_weight = _project_spectral(params.hidden.weight, float(Lambda))
    new_out_weight = _project_l2(params.out.weight.reshape(-1), float(A)).reshape(
        params.out.weight.shape
    )

    params = eqx.tree_at(lambda m: m.hidden.weight, params, new_hidden_weight)
    params = eqx.tree_at(lambda m: m.out.weight, params, new_out_weight)
    return params


def make_ball_tanh_op_projector(*, Lambda: float, A: float) -> Callable[[Any], Any]:
    """Return a post-update projector on the full model."""

    def projector(model: Any) -> Any:
        return project_ball_tanh_op_params(model, Lambda=Lambda, A=A)

    return projector


def make_ball_tanh_op_adam(
    *,
    learning_rate: float,
) -> optax.GradientTransformation:
    """Minimal optimizer helper for the theorem-backed dense operator-norm model."""
    return optax.adam(float(learning_rate))


def check_input_bound(X: jnp.ndarray, *, B: float, atol: float = 1e-6) -> None:
    norms = jnp.linalg.norm(jnp.asarray(X), axis=1)
    max_norm = float(jnp.max(norms))
    if max_norm > float(B) + float(atol):
        raise ValueError(
            f"Input bound violated: max ||x||_2 = {max_norm:.8g} > B = {float(B):.8g}."
        )


def check_ball_tanh_op_constraints(
    model: BallTanhOpNet,
    *,
    Lambda: float,
    A: float,
    atol: float = 1e-6,
) -> None:
    sigma = float(_top_singular_value(model.hidden.weight))
    a_norm = float(jnp.linalg.norm(model.out.weight.reshape(-1)))

    if sigma > float(Lambda) + float(atol):
        raise ValueError(
            f"Hidden operator bound violated: ||W||_op={sigma:.8g} > Lambda={float(Lambda):.8g}."
        )
    if a_norm > float(A) + float(atol):
        raise ValueError(
            f"Output-head bound violated: ||a||_2={a_norm:.8g} > A={float(A):.8g}."
        )


if __name__ == "__main__":
    import numpy as np
    import jax.random as jr
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split

    from quantbayes.ball_dp.api import (
        fit_ball_sgd,
        account_ball_sgd_noise_multiplier,
        calibrate_ball_sgd_noise_multiplier,
        extract_privacy_epsilon,
        evaluate_release_classifier,
        get_release_step_table,
    )
    from quantbayes.ball_dp.plots import plot_release_comparison

    # ------------------------------------------------------------
    # 1) Synthetic binary dataset with a public input bound ||x||_2 <= 1
    # ------------------------------------------------------------
    X, y = make_moons(n_samples=2500, noise=0.18, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int32)

    X_norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(X_norms, 1e-12)
    B = 1.0
    check_input_bound(jnp.asarray(X), B=B)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y
    )
    X_public, X_test, y_public, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=0, stratify=y_tmp
    )

    # ------------------------------------------------------------
    # 2) Theorem-backed dense operator-norm tanh model constants
    # ------------------------------------------------------------
    H = 64
    A = 1.0
    Lambda = 1.0
    r = 1.0
    clip_norm = 1.0

    lz = certified_tanh_mlp_op_lz(A=A, Lambda=Lambda, B=B, H=H)
    projector = make_ball_tanh_op_projector(Lambda=Lambda, A=A)

    def make_model():
        return make_ball_tanh_op_net(
            d_in=X_train.shape[1],
            hidden_dim=H,
            key=jr.PRNGKey(0),
            init_project=True,
            Lambda=Lambda,
            A=A,
        )

    model0 = make_model()
    check_ball_tanh_op_constraints(model0, Lambda=Lambda, A=A)

    print("=" * 80)
    print("Ball-operator-norm-tanh demo setup")
    print("=" * 80)
    print(f"H={H}, A={A}, Lambda={Lambda}, B={B}, r={r}, C={clip_norm}")
    print(f"L_z = {lz:.6f}")
    print(f"L_z * r = {lz * r:.6f}")
    print(f"2C = {2.0 * clip_norm:.6f}")
    print(f"(L_z * r) / (2C) = {(lz * r) / (2.0 * clip_norm):.6f}")
    print(f"Strict Ball sensitivity improvement? {(lz * r) < (2.0 * clip_norm)}")
    print(f"init ||W||_op = {float(_top_singular_value(model0.hidden.weight)):.6f}")
    print(
        f"init ||a||_2  = {float(jnp.linalg.norm(model0.out.weight.reshape(-1))):.6f}"
    )
    print()

    # ------------------------------------------------------------
    # 3) Shared training/accounting hyperparameters
    # ------------------------------------------------------------
    learning_rate = 3e-2
    num_steps = 250
    batch_size = 64
    eval_every = 25
    eval_batch_size = 512
    delta = 1e-5
    epsilon_target = 3.0
    seed = 123
    orders = (2, 3, 4, 5, 8, 16, 32, 64, 128)

    optimizer = make_ball_tanh_op_adam(learning_rate=learning_rate)

    def fit_release(
        *,
        privacy: str,
        noise_multiplier: float,
        epsilon: float | None,
    ):
        return fit_ball_sgd(
            model=make_model(),
            optimizer=optimizer,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_public,
            y_eval=y_public,
            radius=r,
            lz=(lz if privacy.startswith("ball") else None),
            privacy=privacy,
            epsilon=epsilon,
            delta=delta,
            num_steps=num_steps,
            batch_size=batch_size,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            orders=orders,
            loss_name="binary_logistic",
            checkpoint_selection="best_public_eval_accuracy",
            eval_every=eval_every,
            eval_batch_size=eval_batch_size,
            warn_if_ball_equals_standard=True,
            seed=seed,
            key=jr.PRNGKey(seed),
            param_projector=projector,
        )

    # ------------------------------------------------------------
    # 4) Same-noise comparison (accountant only)
    # ------------------------------------------------------------
    shared_noise_multiplier = 1.0

    same_noise_ball = account_ball_sgd_noise_multiplier(
        dataset_size=len(X_train),
        radius=r,
        lz=lz,
        num_steps=num_steps,
        batch_size=batch_size,
        clip_norm=clip_norm,
        noise_multiplier=shared_noise_multiplier,
        delta=delta,
        privacy="ball_rdp",
        orders=orders,
    )
    same_noise_std = account_ball_sgd_noise_multiplier(
        dataset_size=len(X_train),
        radius=r,
        lz=None,
        num_steps=num_steps,
        batch_size=batch_size,
        clip_norm=clip_norm,
        noise_multiplier=shared_noise_multiplier,
        delta=delta,
        privacy="standard_rdp",
        orders=orders,
    )

    print("=" * 80)
    print("Same-noise comparison (accountant only)")
    print("=" * 80)
    print(f"shared noise multiplier = {shared_noise_multiplier:.6f}")
    print(f"Ball epsilon     = {float(same_noise_ball['epsilon']):.6f}")
    print(f"Standard epsilon = {float(same_noise_std['epsilon']):.6f}")
    print()

    # ------------------------------------------------------------
    # 5) Matched-epsilon calibration (accountant only)
    # ------------------------------------------------------------
    calib_ball = calibrate_ball_sgd_noise_multiplier(
        dataset_size=len(X_train),
        radius=r,
        lz=lz,
        num_steps=num_steps,
        batch_size=batch_size,
        clip_norm=clip_norm,
        target_epsilon=epsilon_target,
        delta=delta,
        privacy="ball_rdp",
        orders=orders,
        lower=1e-3,
        upper=0.25,
        max_upper=128.0,
        num_bisection_steps=10,
    )
    calib_std = calibrate_ball_sgd_noise_multiplier(
        dataset_size=len(X_train),
        radius=r,
        lz=None,
        num_steps=num_steps,
        batch_size=batch_size,
        clip_norm=clip_norm,
        target_epsilon=epsilon_target,
        delta=delta,
        privacy="standard_rdp",
        orders=orders,
        lower=1e-3,
        upper=0.25,
        max_upper=128.0,
        num_bisection_steps=10,
    )

    nm_ball = float(calib_ball["noise_multiplier"])
    nm_std = float(calib_std["noise_multiplier"])

    print("=" * 80)
    print("Matched-epsilon calibration (accountant only)")
    print("=" * 80)
    print(f"target epsilon = {epsilon_target:.6f}, delta = {delta:.2e}")
    print(f"Ball calibrated noise multiplier     = {nm_ball:.6f}")
    print(f"Standard calibrated noise multiplier = {nm_std:.6f}")
    print()

    # ------------------------------------------------------------
    # 6) Final matched-epsilon training runs
    # ------------------------------------------------------------
    release_ball = fit_release(
        privacy="ball_dp",
        noise_multiplier=nm_ball,
        epsilon=epsilon_target,
    )
    release_std = fit_release(
        privacy="standard_dp",
        noise_multiplier=nm_std,
        epsilon=epsilon_target,
    )

    eps_ball = extract_privacy_epsilon(release_ball, accounting_view="ball")
    eps_std = extract_privacy_epsilon(release_std, accounting_view="standard")

    test_ball = evaluate_release_classifier(
        release_ball,
        X_test,
        y_test,
        key=jr.PRNGKey(999),
        batch_size=eval_batch_size,
    )
    test_std = evaluate_release_classifier(
        release_std,
        X_test,
        y_test,
        key=jr.PRNGKey(999),
        batch_size=eval_batch_size,
    )

    print("=" * 80)
    print("Matched-epsilon comparison")
    print("=" * 80)
    print(f"target epsilon = {epsilon_target:.6f}, delta = {delta:.2e}")
    print(f"Ball noise multiplier     = {nm_ball:.6f}")
    print(f"Standard noise multiplier = {nm_std:.6f}")
    print(f"Ball achieved epsilon     = {eps_ball:.6f}")
    print(f"Standard achieved epsilon = {eps_std:.6f}")
    print(
        f"Ball public-eval accuracy = "
        f"{release_ball.utility_metrics.get('public_eval_accuracy', float('nan')):.6f}"
    )
    print(
        f"Std  public-eval accuracy = "
        f"{release_std.utility_metrics.get('public_eval_accuracy', float('nan')):.6f}"
    )
    print(f"Ball test accuracy        = {test_ball['accuracy']:.6f}")
    print(f"Std  test accuracy        = {test_std['accuracy']:.6f}")
    print()

    # ------------------------------------------------------------
    # 7) Internal release diagnostics
    # ------------------------------------------------------------
    print("=" * 80)
    print("First 3 Ball release step rows")
    print("=" * 80)
    for row in get_release_step_table(release_ball)[:3]:
        print(row)
    print()

    # ------------------------------------------------------------
    # 8) Visual comparison
    # ------------------------------------------------------------
    plot_release_comparison(
        release_ball,
        release_std,
        labels=(
            "Ball-DP operator-norm (matched ε)",
            "Standard DP operator-norm (matched ε)",
        ),
        metric_keys=("public_eval_accuracy", "public_eval_loss"),
    )
