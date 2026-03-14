# quantbayes/ball_dp/nonconvex/models/ball_net.py
from __future__ import annotations

import math
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


Array = jax.Array


class BallTanhNet(eqx.Module):
    """One-hidden-layer tanh MLP matching the theorem:
         f(x) = (1/sqrt(H)) * a^T tanh(Wx + b)

    hidden bias is allowed; output bias is omitted.
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


def make_ball_tanh_net(
    d_in: int,
    hidden_dim: int,
    *,
    key: Array,
    dtype=None,
    init_project: bool = False,
    S: float | None = None,
    A: float | None = None,
) -> BallTanhNet:
    model = BallTanhNet(
        d_in=d_in,
        hidden_dim=hidden_dim,
        key=key,
        dtype=dtype,
    )
    if init_project:
        if S is None or A is None:
            raise ValueError("init_project=True requires both S and A.")
        model = project_ball_tanh_params(model, S=float(S), A=float(A))
    return model


def tanh_kappa() -> float:
    return 4.0 / (3.0 * math.sqrt(3.0))


def certified_tanh_mlp_constants(
    *,
    A: float,
    S: float,
    B: float,
    H: int,
) -> dict[str, float]:
    if A < 0.0 or S < 0.0 or B < 0.0:
        raise ValueError("A, S, B must be nonnegative.")
    if H <= 0:
        raise ValueError("H must be positive.")

    kappa = tanh_kappa()
    sqrt_H = math.sqrt(float(H))

    L_f = (
        1.0
        / sqrt_H
        * math.sqrt(S * S + (A * kappa * S) ** 2 + (A * (kappa * S * B + 1.0)) ** 2)
    )
    G_f = math.sqrt(1.0 + (A * A * (1.0 + B * B)) / float(H))
    L_z = L_f + (A * S / (4.0 * sqrt_H)) * G_f

    return {
        "kappa_tanh": kappa,
        "L_f": L_f,
        "G_f": G_f,
        "L_z": L_z,
    }


def certified_tanh_mlp_lz(
    *,
    A: float,
    S: float,
    B: float,
    H: int,
) -> float:
    return certified_tanh_mlp_constants(A=A, S=S, B=B, H=H)["L_z"]


def _project_fro(weight: jnp.ndarray, radius: float) -> jnp.ndarray:
    if not math.isfinite(radius):
        return weight
    norm = jnp.linalg.norm(weight)
    scale = jnp.minimum(
        jnp.asarray(1.0, dtype=weight.dtype),
        jnp.asarray(radius, dtype=weight.dtype)
        / jnp.maximum(norm, jnp.asarray(1e-12, dtype=weight.dtype)),
    )
    return weight * scale


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


def project_ball_tanh_params(params: BallTanhNet, *, S: float, A: float) -> BallTanhNet:
    """Project either a full BallTanhNet or its parameter pytree."""
    if S < 0.0 or A < 0.0:
        raise ValueError("S and A must be nonnegative.")

    new_hidden_weight = _project_fro(params.hidden.weight, float(S))
    new_out_weight = _project_l2(params.out.weight.reshape(-1), float(A)).reshape(
        params.out.weight.shape
    )

    params = eqx.tree_at(lambda m: m.hidden.weight, params, new_hidden_weight)
    params = eqx.tree_at(lambda m: m.out.weight, params, new_out_weight)
    return params


def make_ball_tanh_projector(*, S: float, A: float) -> Callable[[Any], Any]:
    def projector(params: Any) -> Any:
        return project_ball_tanh_params(params, S=S, A=A)

    return projector


def check_input_bound(X: jnp.ndarray, *, B: float, atol: float = 1e-6) -> None:
    norms = jnp.linalg.norm(jnp.asarray(X), axis=1)
    max_norm = float(jnp.max(norms))
    if max_norm > float(B) + float(atol):
        raise ValueError(
            f"Input bound violated: max ||x||_2 = {max_norm:.8g} > B = {float(B):.8g}."
        )


if __name__ == "__main__":
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import equinox as eqx
    import optax
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split

    from quantbayes.ball_dp.config import NonconvexReleaseConfig
    from quantbayes.ball_dp.types import ArrayDataset
    from quantbayes.ball_dp.nonconvex.sgd import (
        run_ball_sgd_rdp,
        run_ball_sgd_dp,
        run_standard_sgd_rdp,
        run_standard_sgd_dp,
    )

    # ----------------------------
    # Small helpers
    # ----------------------------

    def _extract_epsilon(artifact, view: str) -> float:
        ledger = artifact.privacy.ball if view == "ball" else artifact.privacy.standard
        if not ledger.dp_certificates:
            return math.inf
        return float(ledger.dp_certificates[0].epsilon)

    def _extract_history(artifact, key: str):
        hist = artifact.extra.get("public_curve_history", [])
        xs, ys = [], []
        for row in hist:
            if key in row:
                xs.append(int(row["step"]))
                ys.append(float(row[key]))
        return xs, ys

    def _test_accuracy(model, X, y, *, key):
        model_eval = eqx.nn.inference_mode(model, value=True)
        Xj = jnp.asarray(X, dtype=jnp.float32)
        keys = jr.split(key, Xj.shape[0])

        logits = jax.vmap(
            lambda x, k: model_eval(x, key=k, state=None)[0],
            in_axes=(0, 0),
        )(Xj, keys)
        preds = (np.asarray(logits) > 0.0).astype(np.int32)
        return float(np.mean(preds == np.asarray(y, dtype=np.int32)))

    def _make_cfg(*, noise_multiplier: float, lz_value, epsilon_value=None, seed=0):
        # Keep Ball and standard runs identical except for the accounting view and lz.
        return NonconvexReleaseConfig(
            epsilon=epsilon_value,
            delta=delta,
            radius=r,
            lz=lz_value,
            num_steps=num_steps,
            batch_sizes=batch_size,
            clip_norms=clip_norm,
            noise_multipliers=noise_multiplier,
            orders=orders,
            loss_name="binary_logistic",
            normalize_noisy_sum_by="batch_size",
            eval_every=eval_every,
            eval_batch_size=eval_batch_size,
            checkpoint_selection="best_public_eval_accuracy",
            warn_if_ball_equals_standard=True,
            seed=seed,
        )

    def _fresh_model():
        # Same initialization every time for a fair comparison.
        return make_ball_tanh_net(
            d_in=X_train.shape[1],
            hidden_dim=H,
            key=jr.PRNGKey(model_seed),
            init_project=True,
            S=S,
            A=A,
        )

    def _run_rdp(view: str, noise_multiplier: float):
        cfg = _make_cfg(
            noise_multiplier=noise_multiplier,
            lz_value=(lz if view == "ball" else None),
            epsilon_value=None,
        )
        model = _fresh_model()
        if view == "ball":
            return run_ball_sgd_rdp(
                dataset=train_ds,
                cfg=cfg,
                model=model,
                optimizer=optax.adam(learning_rate),
                public_eval_dataset=public_eval_ds,
                key=jr.PRNGKey(train_seed),
                param_projector=projector,
            )
        elif view == "standard":
            return run_standard_sgd_rdp(
                dataset=train_ds,
                cfg=cfg,
                model=model,
                optimizer=optax.adam(learning_rate),
                public_eval_dataset=public_eval_ds,
                key=jr.PRNGKey(train_seed),
                param_projector=projector,
            )
        else:
            raise ValueError(f"Unknown view={view!r}")

    def _run_dp(view: str, noise_multiplier: float):
        cfg = _make_cfg(
            noise_multiplier=noise_multiplier,
            lz_value=(lz if view == "ball" else None),
            epsilon_value=epsilon_target,
        )
        model = _fresh_model()
        if view == "ball":
            return run_ball_sgd_dp(
                dataset=train_ds,
                cfg=cfg,
                model=model,
                optimizer=optax.adam(learning_rate),
                public_eval_dataset=public_eval_ds,
                key=jr.PRNGKey(train_seed),
                param_projector=projector,
            )
        elif view == "standard":
            return run_standard_sgd_dp(
                dataset=train_ds,
                cfg=cfg,
                model=model,
                optimizer=optax.adam(learning_rate),
                public_eval_dataset=public_eval_ds,
                key=jr.PRNGKey(train_seed),
                param_projector=projector,
            )
        else:
            raise ValueError(f"Unknown view={view!r}")

    def _calibrate_noise_multiplier(view: str, target_epsilon: float):
        # Privacy epsilon depends only on the accountant inputs, not on optimization
        # outcomes, but we use the actual trainer here as a simple tutorial path.
        lo, hi = 1e-3, 0.25

        # Expand upper bracket until epsilon <= target.
        while True:
            art = _run_rdp(view, hi)
            eps = _extract_epsilon(art, view)
            if eps <= target_epsilon:
                break
            hi *= 2.0
            if hi > 128.0:
                raise RuntimeError(
                    f"Could not bracket a noise multiplier for view={view!r}."
                )

        # Bisection for a reasonably tight calibration.
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            art = _run_rdp(view, mid)
            eps = _extract_epsilon(art, view)
            if eps <= target_epsilon:
                hi = mid
            else:
                lo = mid

        return float(hi)

    # ----------------------------
    # Synthetic binary dataset
    # ----------------------------
    X, y = make_moons(n_samples=2500, noise=0.18, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int32)

    # Public preprocessing to enforce ||x||_2 <= 1.
    X_norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(X_norms, 1e-12)
    B = 1.0
    check_input_bound(jnp.asarray(X), B=B)

    # Train / public-eval / held-out test split.
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y
    )
    X_public, X_test, y_public, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=0, stratify=y_tmp
    )

    train_ds = ArrayDataset(X_train, y_train)
    public_eval_ds = ArrayDataset(X_public, y_public)

    # ----------------------------
    # Theorem / training hyperparameters
    # ----------------------------
    H = 64
    A = 1.0
    S = 1.0
    r = 1.0
    clip_norm = 1.0

    # This is the theorem-backed Ball constant for THIS model family.
    lz = certified_tanh_mlp_lz(A=A, S=S, B=B, H=H)

    learning_rate = 3e-2
    num_steps = 250
    batch_size = 64
    eval_every = 25
    eval_batch_size = 512
    orders = tuple(range(2, 65))
    delta = 1e-5
    epsilon_target = 3.0

    model_seed = 0
    train_seed = 123

    projector = make_ball_tanh_projector(S=S, A=A)

    print("=" * 80)
    print("Ball-tanh demo setup")
    print("=" * 80)
    print(f"H={H}, A={A}, S={S}, B={B}, r={r}, C={clip_norm}")
    print(f"L_z = {lz:.6f}")
    print(f"L_z * r = {lz * r:.6f}")
    print(f"2C = {2.0 * clip_norm:.6f}")
    print(f"(L_z * r) / (2C) = {(lz * r) / (2.0 * clip_norm):.6f}")
    print(f"Strict Ball sensitivity improvement? {(lz * r) < (2.0 * clip_norm)}")
    print()

    # ------------------------------------------------------------------
    # 1) Same-noise comparison: identical training mechanism, different
    #    privacy accounting. This shows the certificate gain directly.
    # ------------------------------------------------------------------
    shared_noise_multiplier = 1.0

    artifact_ball_same_noise = _run_rdp("ball", shared_noise_multiplier)
    artifact_std_same_noise = _run_rdp("standard", shared_noise_multiplier)

    eps_ball_same_noise = _extract_epsilon(artifact_ball_same_noise, "ball")
    eps_std_same_noise = _extract_epsilon(artifact_std_same_noise, "standard")

    print("=" * 80)
    print("Same-noise comparison")
    print("=" * 80)
    print(f"shared noise multiplier = {shared_noise_multiplier:.6f}")
    print(f"Ball epsilon     = {eps_ball_same_noise:.6f}")
    print(f"Standard epsilon = {eps_std_same_noise:.6f}")
    print("At the same noise, Ball should certify a smaller (or equal) epsilon.")
    print()

    # ------------------------------------------------------------------
    # 2) Matched-epsilon comparison: calibrate noise separately for Ball
    #    and standard, then compare utility at the same target epsilon.
    # ------------------------------------------------------------------
    nm_ball = _calibrate_noise_multiplier("ball", epsilon_target)
    nm_std = _calibrate_noise_multiplier("standard", epsilon_target)

    artifact_ball = _run_dp("ball", nm_ball)
    artifact_std = _run_dp("standard", nm_std)

    eps_ball = _extract_epsilon(artifact_ball, "ball")
    eps_std = _extract_epsilon(artifact_std, "standard")

    acc_test_ball = _test_accuracy(
        artifact_ball.payload,
        X_test,
        y_test,
        key=jr.PRNGKey(999),
    )
    acc_test_std = _test_accuracy(
        artifact_std.payload,
        X_test,
        y_test,
        key=jr.PRNGKey(999),
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
        f"Ball public-eval accuracy = {artifact_ball.utility_metrics.get('public_eval_accuracy', float('nan')):.6f}"
    )
    print(
        f"Std  public-eval accuracy = {artifact_std.utility_metrics.get('public_eval_accuracy', float('nan')):.6f}"
    )
    print(f"Ball test accuracy        = {acc_test_ball:.6f}")
    print(f"Std  test accuracy        = {acc_test_std:.6f}")
    print()

    # do use run_standard_sgd_* to compare with normal dp/rdp
    # Do NOT compare to "normal DP" by setting lz = 2*C inside the Ball runner.
    # The standard baseline is run_standard_sgd_dp / run_standard_sgd_rdp.

    # ----------------------------
    # Visualizations
    # ----------------------------
    steps_ball_acc, ball_acc_curve = _extract_history(
        artifact_ball, "public_eval_accuracy"
    )
    steps_std_acc, std_acc_curve = _extract_history(
        artifact_std, "public_eval_accuracy"
    )
    steps_ball_loss, ball_loss_curve = _extract_history(
        artifact_ball, "public_eval_loss"
    )
    steps_std_loss, std_loss_curve = _extract_history(artifact_std, "public_eval_loss")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(steps_ball_acc, ball_acc_curve, label="Ball-DP (matched ε)")
    axes[0].plot(steps_std_acc, std_acc_curve, label="Standard DP (matched ε)")
    axes[0].set_title("Public-eval accuracy")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps_ball_loss, ball_loss_curve, label="Ball-DP (matched ε)")
    axes[1].plot(steps_std_loss, std_loss_curve, label="Standard DP (matched ε)")
    axes[1].set_title("Public-eval loss")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Optional extra plot: same-noise certificate comparison.
    plt.figure(figsize=(5, 4))
    plt.bar(
        ["Ball\n(same noise)", "Standard\n(same noise)"],
        [eps_ball_same_noise, eps_std_same_noise],
    )
    plt.ylabel("Certified epsilon")
    plt.title("Same-noise privacy certificates")
    plt.tight_layout()
    plt.show()
