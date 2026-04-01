# quantbayes/ball_dp/nonconvex/models/ball_net_svd.py

from __future__ import annotations

import math
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from quantbayes.stochax.utils.optim_util import make_freeze_mask


Array = jax.Array


class SVDDense(eqx.Module):
    """Fixed-basis rectangular SVD dense used by the Ball-DP theorem model.

    This is intentionally theorem-specific:
      W = U diag(s) V^T
    with U and V frozen orthonormal factors and only s trainable.
    """

    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)

    U: jnp.ndarray  # (out_features, rank)
    V: jnp.ndarray  # (in_features,  rank)
    s: jnp.ndarray  # (rank,)
    bias: jnp.ndarray  # (out_features,)

    def __init__(self, U, V, s, bias):
        U = jnp.asarray(U)
        V = jnp.asarray(V)
        s = jnp.asarray(s)
        bias = jnp.asarray(bias)

        r = int(s.shape[0])
        if U.shape != (bias.shape[0], r):
            raise ValueError(
                f"Expected U.shape={(bias.shape[0], r)}, got {tuple(U.shape)}."
            )
        if V.shape[1] != r:
            raise ValueError(f"Expected V.shape[1]={r}, got {V.shape[1]}.")

        self.in_features = int(V.shape[0])
        self.out_features = int(U.shape[0])
        self.rank = r

        self.U = U
        self.V = V
        self.s = s
        self.bias = bias

    @classmethod
    def from_linear(cls, lin: eqx.nn.Linear, *, rank: int | None = None):
        """Exact SVD warm-start from an eqx.nn.Linear."""
        W = jnp.asarray(lin.weight)  # (out, in)
        U, s, Vh = jnp.linalg.svd(W, full_matrices=False)

        if rank is not None:
            r = int(rank)
            if r <= 0:
                raise ValueError("rank must be positive.")
            r = min(r, int(s.shape[0]))
            U = U[:, :r]
            s = s[:r]
            Vh = Vh[:r, :]

        V = Vh.T
        bias = (
            jnp.asarray(lin.bias)
            if getattr(lin, "bias", None) is not None
            else jnp.zeros((W.shape[0],), dtype=W.dtype)
        )
        return cls(U=U, V=V, s=s, bias=bias)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        U = jax.lax.stop_gradient(self.U)
        V = jax.lax.stop_gradient(self.V)

        z = x @ V
        z = z * self.s
        y = z @ U.T
        return y + self.bias

    def __operator_norm_hint__(self):
        return jnp.max(jnp.abs(self.s)).astype(jnp.float32)


# This theorem-specific layer uses a genuinely fixed basis.
# Keep an explicit alias to distinguish it conceptually from the generic
# stochax SVDDense layer, which is intended for broader spectral experiments.
FixedBasisSVDDense = SVDDense


def full_hidden_rank(d_in: int, hidden_dim: int) -> int:
    """Maximum rank available for the hidden map W ∈ R^{hidden_dim × d_in}."""
    return min(int(d_in), int(hidden_dim))


class BallSVDTanhNet(eqx.Module):
    """One-hidden-layer tanh MLP with fixed-basis SVD hidden layer:

         f(x) = (1/sqrt(H)) * a^T tanh(U diag(s) V^T x + b)

    The theorem-backed trainable parameters are:
      - hidden singular values s
      - hidden bias b
      - output head a
    while U and V remain fixed.
    """

    hidden: SVDDense
    out: eqx.nn.Linear
    hidden_dim: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)

    def __init__(
        self,
        d_in: int,
        hidden_dim: int,
        *,
        rank: int | None = None,
        key: Array,
        dtype=None,
    ):
        k1, k2 = jr.split(key, 2)

        dense_hidden = eqx.nn.Linear(
            d_in,
            hidden_dim,
            use_bias=True,
            dtype=dtype,
            key=k1,
        )
        self.hidden = SVDDense.from_linear(dense_hidden, rank=rank)
        self.out = eqx.nn.Linear(
            hidden_dim,
            1,
            use_bias=False,
            dtype=dtype,
            key=k2,
        )

        self.hidden_dim = int(hidden_dim)
        self.rank = int(self.hidden.rank)

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


def make_ball_svd_tanh_net(
    d_in: int,
    hidden_dim: int,
    *,
    rank: int | None = None,
    key: Array,
    dtype=None,
    init_project: bool = False,
    Lambda: float | None = None,
    A: float | None = None,
) -> BallSVDTanhNet:
    model = BallSVDTanhNet(
        d_in=d_in,
        hidden_dim=hidden_dim,
        rank=rank,
        key=key,
        dtype=dtype,
    )
    if init_project:
        if Lambda is None or A is None:
            raise ValueError("init_project=True requires both Lambda and A.")
        model = project_ball_svd_params(model, Lambda=float(Lambda), A=float(A))
    return model


def make_ball_svd_tanh_net_from_dense(
    dense_model: Any,
    *,
    rank: int | None = None,
    init_project: bool = False,
    Lambda: float | None = None,
    A: float | None = None,
) -> BallSVDTanhNet:
    """Build a fixed-basis Ball-SVD tanh net from a trained dense tanh model.

    Intended use:
      1) train a public-only dense BallTanhNet (or any duck-typed equivalent)
      2) compute the SVD of its hidden layer
      3) freeze U,V and private-train only s,b,a

    The output head is copied from the dense model.
    """
    hidden_weight = jnp.asarray(dense_model.hidden.weight)
    d_in = int(hidden_weight.shape[1])
    hidden_dim = int(hidden_weight.shape[0])
    dtype = hidden_weight.dtype

    model = BallSVDTanhNet(
        d_in=d_in,
        hidden_dim=hidden_dim,
        rank=rank,
        key=jr.PRNGKey(0),
        dtype=dtype,
    )
    hidden = FixedBasisSVDDense.from_linear(dense_model.hidden, rank=rank)

    model = eqx.tree_at(lambda m: m.hidden, model, hidden)
    model = eqx.tree_at(lambda m: m.out, model, dense_model.out)

    if init_project:
        if Lambda is None or A is None:
            raise ValueError("init_project=True requires both Lambda and A.")
        model = project_ball_svd_params(model, Lambda=float(Lambda), A=float(A))

    return model


def svd_frobenius_energy_fraction(weight: jnp.ndarray, rank: int | None) -> float:
    """Return the retained Frobenius-energy fraction of the rank-r truncated SVD."""
    s = jnp.linalg.svd(jnp.asarray(weight), compute_uv=False)
    r_eff = int(s.shape[0]) if rank is None else min(int(rank), int(s.shape[0]))
    num = jnp.sum(s[:r_eff] ** 2)
    den = jnp.sum(s**2)
    frac = num / jnp.maximum(den, jnp.asarray(1e-12, dtype=s.dtype))
    return float(frac)


def tanh_kappa() -> float:
    return 4.0 / (3.0 * math.sqrt(3.0))


def certified_tanh_svd_mlp_constants(
    *,
    A: float,
    Lambda: float,
    B: float,
    H: int,
) -> dict[str, float]:
    """The theorem-backed constants for the fixed-basis SVD tanh MLP.

    Note: the explicit bound depends on H, A, Lambda, B, but not directly on rank.
    """
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


def certified_tanh_svd_mlp_lz(
    *,
    A: float,
    Lambda: float,
    B: float,
    H: int,
) -> float:
    return certified_tanh_svd_mlp_constants(
        A=A,
        Lambda=Lambda,
        B=B,
        H=H,
    )["L_z"]


def _project_linf(vec: jnp.ndarray, radius: float) -> jnp.ndarray:
    if not math.isfinite(radius):
        return vec
    bound = jnp.asarray(radius, dtype=vec.dtype)
    return jnp.clip(vec, -bound, bound)


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


def project_ball_svd_params(
    params: BallSVDTanhNet,
    *,
    Lambda: float,
    A: float,
) -> BallSVDTanhNet:
    """Project the theorem-backed trainable parameters.

    Projects:
      - hidden singular values s onto the l_infinity ball of radius Lambda
      - output head a onto the l2 ball of radius A

    Leaves hidden U, V, and b unchanged.
    """
    if Lambda < 0.0 or A < 0.0:
        raise ValueError("Lambda and A must be nonnegative.")

    new_s = _project_linf(params.hidden.s, float(Lambda))
    new_out_weight = _project_l2(params.out.weight.reshape(-1), float(A)).reshape(
        params.out.weight.shape
    )

    params = eqx.tree_at(lambda m: m.hidden.s, params, new_s)
    params = eqx.tree_at(lambda m: m.out.weight, params, new_out_weight)
    return params


def make_ball_svd_projector(*, Lambda: float, A: float) -> Callable[[Any], Any]:
    """Return a post-update projector on the full model."""

    def projector(model: Any) -> Any:
        return project_ball_svd_params(model, Lambda=Lambda, A=A)

    return projector


def make_ball_svd_freeze_mask(model: Any):
    """Freeze the hidden SVD basis matrices U and V."""
    return make_freeze_mask(model, names=("U", "V"))


def make_ball_svd_adam(
    model: Any,
    *,
    learning_rate: float,
) -> optax.GradientTransformation:
    """Minimal theorem-safe optimizer for BallSVDTanhNet.

    This zeroes updates to U and V before Adam. For AdamW or more elaborate
    setups, use your build_optimizer(..., prepend=...) path with the same mask.
    """
    freeze_mask = make_ball_svd_freeze_mask(model)
    return optax.chain(
        optax.masked(optax.set_to_zero(), freeze_mask),
        optax.adam(float(learning_rate)),
    )


def check_input_bound(X: jnp.ndarray, *, B: float, atol: float = 1e-6) -> None:
    norms = jnp.linalg.norm(jnp.asarray(X), axis=1)
    max_norm = float(jnp.max(norms))
    if max_norm > float(B) + float(atol):
        raise ValueError(
            f"Input bound violated: max ||x||_2 = {max_norm:.8g} > B = {float(B):.8g}."
        )


def check_hidden_basis_orthonormal(
    model: Any,
    *,
    atol: float = 1e-5,
) -> None:
    """Check that model.hidden.U and model.hidden.V are column-orthonormal."""
    U = jnp.asarray(model.hidden.U)
    V = jnp.asarray(model.hidden.V)

    Iu = U.T @ U
    Iv = V.T @ V

    err_u = float(jnp.max(jnp.abs(Iu - jnp.eye(U.shape[1], dtype=Iu.dtype))))
    err_v = float(jnp.max(jnp.abs(Iv - jnp.eye(V.shape[1], dtype=Iv.dtype))))

    if err_u > float(atol):
        raise ValueError(
            f"Hidden U is not orthonormal to tolerance {atol:.3g}: max error={err_u:.6g}."
        )
    if err_v > float(atol):
        raise ValueError(
            f"Hidden V is not orthonormal to tolerance {atol:.3g}: max error={err_v:.6g}."
        )


def check_ball_svd_constraints(
    model: BallSVDTanhNet,
    *,
    Lambda: float,
    A: float,
    atol: float = 1e-6,
) -> None:
    sigma = float(jnp.max(jnp.abs(model.hidden.s)))
    a_norm = float(jnp.linalg.norm(model.out.weight.reshape(-1)))

    if sigma > float(Lambda) + float(atol):
        raise ValueError(
            f"Hidden operator bound violated: ||s||_inf={sigma:.8g} > Lambda={float(Lambda):.8g}."
        )
    if a_norm > float(A) + float(atol):
        raise ValueError(
            f"Output-head bound violated: ||a||_2={a_norm:.8g} > A={float(A):.8g}."
        )


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification
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
    X, y = make_classification(
        n_samples=2500,
        n_features=64,
        n_informative=20,
        n_redundant=10,
        n_repeated=0,
        n_clusters_per_class=2,
        class_sep=1.25,
        flip_y=0.03,
        random_state=0,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int32)

    # Public preprocessing to enforce the theorem assumption ||x||_2 <= B.
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
    # 2) Theorem-backed Ball-SVD-tanh model constants
    # ------------------------------------------------------------
    H = 64
    rank = 32
    A = 1.0
    Lambda = 1.0
    r = 1.0
    clip_norm = 1.0

    lz = certified_tanh_svd_mlp_lz(A=A, Lambda=Lambda, B=B, H=H)
    projector = make_ball_svd_projector(Lambda=Lambda, A=A)

    def make_model():
        return make_ball_svd_tanh_net(
            d_in=X_train.shape[1],
            hidden_dim=H,
            rank=rank,
            key=jr.PRNGKey(0),
            init_project=True,
            Lambda=Lambda,
            A=A,
        )

    model0 = make_model()
    check_hidden_basis_orthonormal(model0)
    check_ball_svd_constraints(model0, Lambda=Lambda, A=A)

    print("=" * 80)
    print("Ball-SVD-tanh demo setup")
    print("=" * 80)
    full_rank = full_hidden_rank(X_train.shape[1], H)
    print(
        f"H={H}, rank={model0.rank}, full_rank={full_rank}, "
        f"A={A}, Lambda={Lambda}, B={B}, r={r}, C={clip_norm}"
    )
    print(f"L_z = {lz:.6f}")
    print(f"L_z * r = {lz * r:.6f}")
    print(f"2C = {2.0 * clip_norm:.6f}")
    print(f"(L_z * r) / (2C) = {(lz * r) / (2.0 * clip_norm):.6f}")
    print(f"Strict Ball sensitivity improvement? {(lz * r) < (2.0 * clip_norm)}")
    print(f"init ||s||_inf = {float(jnp.max(jnp.abs(model0.hidden.s))):.6f}")
    print(
        f"init ||a||_2   = {float(jnp.linalg.norm(model0.out.weight.reshape(-1))):.6f}"
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

    optimizer = make_ball_svd_adam(model0, learning_rate=learning_rate)

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
    # 4) Same-noise comparison (accountant only; no training needed)
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

    eps_ball_same_noise = float(same_noise_ball["epsilon"])
    eps_std_same_noise = float(same_noise_std["epsilon"])

    print("=" * 80)
    print("Same-noise comparison (accountant only)")
    print("=" * 80)
    print(f"shared noise multiplier = {shared_noise_multiplier:.6f}")
    print(f"Ball epsilon     = {eps_ball_same_noise:.6f}")
    print(f"Standard epsilon = {eps_std_same_noise:.6f}")
    print("At the same noise, Ball should certify a smaller (or equal) epsilon.")
    print()

    # ------------------------------------------------------------
    # 5) Matched-epsilon calibration (accountant only; no training needed)
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
    # 8) Visual comparison of matched-epsilon runs
    # ------------------------------------------------------------
    plot_release_comparison(
        release_ball,
        release_std,
        labels=("Ball-DP SVD (matched ε)", "Standard DP SVD (matched ε)"),
        metric_keys=("public_eval_accuracy", "public_eval_loss"),
    )
