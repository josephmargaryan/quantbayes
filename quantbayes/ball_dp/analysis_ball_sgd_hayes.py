# analysis_ball_sgd_hayes.py
#
# Assumes:
#   - you already trained with fit_ball_sgd(...) and have a `release`
#   - you have X_train (or any reference array with the same feature shape)
#
# This script adds:
#   1) full ReRo curve vs kappa
#   2) gap curve Gamma(kappa) - kappa
#   3) stepwise sensitivity diagnostics
#   4) prefix Ball-ReRo-vs-utility trajectory
#   5) multi-run privacy-utility frontier

from __future__ import annotations

import dataclasses as dc
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from quantbayes.ball_dp import (
    ball_rero,
    get_public_curve_history,
    get_release_step_table,
    make_uniform_ball_prior,
)


def assert_hayes_theorem_conditions(release) -> None:
    cfg = dict(release.training_config)

    sampler = str(
        cfg.get("resolved_batch_sampler", cfg.get("batch_sampler", ""))
    ).lower()
    if sampler != "poisson":
        raise ValueError(
            "mode='ball_sgd_hayes' is theorem-backed only for actual Poisson subsampling."
        )

    if bool(cfg.get("fixed_batch_schedule_present", False)):
        raise ValueError(
            "mode='ball_sgd_hayes' is not theorem-backed when a fixed batch schedule was used."
        )

    norm = str(cfg.get("normalize_noisy_sum_by", "batch_size")).lower()
    if norm not in {"batch_size", "none"}:
        raise ValueError(
            "mode='ball_sgd_hayes' supports normalize_noisy_sum_by in {'batch_size', 'none'}."
        )


def flattened_feature_dim(X: np.ndarray) -> int:
    X = np.asarray(X)
    if X.ndim < 2:
        raise ValueError("X must have shape (n, ...).")
    return int(X.reshape(len(X), -1).shape[1])


def make_continuous_uniform_prior_for_release(release, X_reference: np.ndarray):
    """
    Build the certified continuous uniform-ball prior used for the publication plots.

    Note:
      For this prior family, kappa depends only on radius and ambient dimension,
      not on the center, so a zero center is fine here.
    """
    dim = flattened_feature_dim(X_reference)
    radius = float(release.training_config["radius"])
    center = np.zeros(dim, dtype=np.float32)
    prior = make_uniform_ball_prior(center=center, radius=radius, dimension=dim)
    return prior, radius, dim


def make_eta_grid_for_uniform_ball(
    radius: float,
    *,
    lo_ratio: float = 0.85,
    hi_ratio: float = 0.999,
    n: int = 80,
) -> tuple[float, ...]:
    """
    Dense eta-grid near the boundary eta/r -> 1, which is where kappa stays nontrivial
    in moderate/high dimension.
    """
    ratios = np.linspace(lo_ratio, hi_ratio, int(n), dtype=np.float64)
    return tuple(float(radius * r) for r in ratios)


def eta_for_uniform_ball_kappa(radius: float, dimension: int, kappa: float) -> float:
    """
    For the uniform L2-ball prior:
        kappa(eta) = (eta / r)^d
    so
        eta = r * kappa^(1/d).
    """
    kappa = float(kappa)
    if not (0.0 < kappa <= 1.0):
        raise ValueError("kappa must lie in (0, 1].")
    return float(radius * (kappa ** (1.0 / float(dimension))))


def _sorted_report_arrays(report, which: str = "ball"):
    if which not in {"ball", "standard"}:
        raise ValueError("which must be 'ball' or 'standard'.")

    xs = np.asarray([float(p.kappa) for p in report.points], dtype=np.float64)
    if which == "ball":
        ys = np.asarray([float(p.gamma_ball) for p in report.points], dtype=np.float64)
    else:
        ys = np.asarray(
            [
                np.nan if p.gamma_standard is None else float(p.gamma_standard)
                for p in report.points
            ],
            dtype=np.float64,
        )

    mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0.0)
    xs = xs[mask]
    ys = ys[mask]

    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    # Remove duplicate kappas so interpolation is well-defined.
    xs_unique, idx = np.unique(xs, return_index=True)
    ys_unique = ys[idx]
    return xs_unique, ys_unique


def gamma_at_kappa(report, target_kappa: float, *, which: str = "ball") -> float:
    """
    Interpolate gamma at a common target kappa.
    Useful for cross-run privacy-utility frontiers.
    """
    x, y = _sorted_report_arrays(report, which=which)
    if x.size == 0:
        raise ValueError(f"report has no finite {which!r} curve values.")

    target_kappa = float(target_kappa)
    if not (x[0] <= target_kappa <= x[-1]):
        raise ValueError(
            f"target_kappa={target_kappa:g} lies outside the report range [{x[0]:g}, {x[-1]:g}]."
        )
    return float(np.interp(target_kappa, x, y))


def plot_ball_regime_threshold(release) -> None:
    rows = get_release_step_table(release)

    steps = np.asarray([int(r["step"]) for r in rows], dtype=np.int64)
    clip_norm = np.asarray([float(r["clip_norm"]) for r in rows], dtype=np.float64)

    lz = release.training_config.get("lz", None)
    radius = float(release.training_config["radius"])
    if lz is None:
        raise ValueError(
            "release.training_config['lz'] is missing; Ball threshold cannot be formed."
        )

    ball_raw = float(lz) * radius  # L_z r
    two_c = 2.0 * clip_norm  # 2 C_t

    plt.figure(figsize=(7, 4))
    plt.plot(steps, two_c, marker="o", label=r"$2C_t$")
    plt.plot(
        steps,
        np.full_like(steps, ball_raw, dtype=np.float64),
        linestyle="--",
        label=r"$L_z r$",
    )
    plt.xlabel("step")
    plt.ylabel("sensitivity scale")
    plt.title(r"Ball regime threshold: compare $L_z r$ to $2C_t$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_hayes_curve(
    report, *, show_gap: bool = False, title: str | None = None
) -> None:
    """
    Publication-friendly plot against kappa.
    show_gap=False: plot Gamma(kappa)
    show_gap=True : plot Gamma(kappa) - kappa
    """
    k_ball, g_ball = _sorted_report_arrays(report, which="ball")
    k_std, g_std = _sorted_report_arrays(report, which="standard")

    if show_gap:
        g_ball = g_ball - k_ball
        if k_std.size:
            g_std = g_std - k_std
        ylabel = r"$\Gamma(\kappa) - \kappa$"
    else:
        ylabel = r"$\Gamma(\kappa)$"

    plt.figure(figsize=(6, 4))
    plt.semilogx(k_ball, g_ball, marker="o", label="ball")
    if k_std.size:
        plt.semilogx(k_std, g_std, marker="o", label="standard")
    if not show_gap:
        plt.semilogx(
            k_ball,
            k_ball,
            linestyle="--",
            label=r"baseline $\Gamma(\kappa)=\kappa$",
        )
    plt.xlabel(r"$\kappa$")
    plt.ylabel(ylabel)
    plt.title(title or f"ReRo curve ({report.mode})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def truncate_release_for_direct_modes(release, upto_step: int):
    """
    Create a prefix release artifact by truncating the per-step schedules.

    This is appropriate for:
      - mode='ball_sgd_hayes'
      - mode='ball_sgd_direct'

    It is NOT appropriate for prefix 'rdp' or 'dp' unless you also recompute
    the accountant for the truncated run.
    """
    upto_step = int(upto_step)
    if upto_step <= 0:
        raise ValueError("upto_step must be >= 1.")

    cfg = dict(release.training_config)
    extra = dict(release.extra)
    attack_metadata = dict(release.attack_metadata)

    def _truncate_sequence(x):
        if x is None:
            return None
        if isinstance(x, tuple):
            return tuple(x[:upto_step])
        if isinstance(x, list):
            return list(x[:upto_step])
        return x

    for key in (
        "batch_sizes",
        "sample_rates",
        "clip_norms",
        "noise_multipliers",
        "effective_noise_stds",
    ):
        if key in cfg:
            cfg[key] = _truncate_sequence(cfg[key])

    cfg["num_steps"] = min(int(cfg.get("num_steps", upto_step)), upto_step)

    step_delta_ball = release.sensitivity.step_delta_ball
    step_delta_std = release.sensitivity.step_delta_std
    step_delta_ball_prefix = (
        None if step_delta_ball is None else list(step_delta_ball[:upto_step])
    )
    step_delta_std_prefix = (
        None if step_delta_std is None else list(step_delta_std[:upto_step])
    )

    sensitivity = dc.replace(
        release.sensitivity,
        delta_ball=(
            None
            if step_delta_ball_prefix is None or len(step_delta_ball_prefix) == 0
            else float(max(step_delta_ball_prefix))
        ),
        delta_std=(
            None
            if step_delta_std_prefix is None or len(step_delta_std_prefix) == 0
            else float(max(step_delta_std_prefix))
        ),
        step_delta_ball=step_delta_ball_prefix,
        step_delta_std=step_delta_std_prefix,
    )

    if "public_curve_history" in extra and extra["public_curve_history"] is not None:
        extra["public_curve_history"] = [
            dict(row)
            for row in extra["public_curve_history"]
            if int(row.get("step", 10**18)) <= upto_step
        ]

    if "operator_norm_history" in extra and extra["operator_norm_history"] is not None:
        extra["operator_norm_history"] = [
            dict(row)
            for row in extra["operator_norm_history"]
            if int(row.get("step", 10**18)) <= upto_step
        ]

    for key in ("rho_by_step", "ball_to_standard_sensitivity_ratio_by_step"):
        if key in extra and extra[key] is not None:
            extra[key] = list(extra[key][:upto_step])

    if "selected_checkpoint_step" in extra:
        extra["selected_checkpoint_step"] = min(
            int(extra["selected_checkpoint_step"]),
            upto_step,
        )

    if "selected_checkpoint_step" in attack_metadata:
        attack_metadata["selected_checkpoint_step"] = min(
            int(attack_metadata["selected_checkpoint_step"]),
            upto_step,
        )

    return dc.replace(
        release,
        training_config=cfg,
        sensitivity=sensitivity,
        extra=extra,
        attack_metadata=attack_metadata,
    )


def compute_hayes_prefix_trajectory(
    release,
    *,
    prior,
    eta: float,
    checkpoints: Sequence[int] | None = None,
):
    """
    Compute the prefix trajectory:
        t -> Gamma_{1:t}(kappa(eta))
    """
    assert_hayes_theorem_conditions(release)

    if checkpoints is None:
        hist = get_public_curve_history(release)
        if hist:
            checkpoints = sorted({int(row["step"]) for row in hist})
        else:
            T = int(release.training_config["num_steps"])
            checkpoints = np.unique(
                np.linspace(1, T, num=min(T, 25), dtype=np.int64)
            ).tolist()

    rows = []
    for step in checkpoints:
        prefix_release = truncate_release_for_direct_modes(release, int(step))
        report = ball_rero(
            prefix_release,
            prior=prior,
            eta_grid=(float(eta),),
            mode="ball_sgd_hayes",
        )
        point = report.points[0]
        rows.append(
            {
                "step": int(step),
                "eta": float(point.eta),
                "kappa": float(point.kappa),
                "gamma_ball": float(point.gamma_ball),
                "gamma_standard": (
                    None
                    if point.gamma_standard is None
                    else float(point.gamma_standard)
                ),
            }
        )
    return rows


def plot_prefix_privacy_vs_utility(
    release,
    *,
    prior,
    eta: float,
    utility_key: str = "public_eval_accuracy",
) -> None:
    """
    Overlay public utility with the prefix privacy gap Gamma(kappa)-kappa.
    Uses exactly the steps at which public evaluation was recorded.
    """
    hist = [
        dict(row) for row in get_public_curve_history(release) if utility_key in row
    ]
    if not hist:
        raise ValueError(
            f"release has no public curve history with key={utility_key!r}."
        )

    checkpoints = [int(row["step"]) for row in hist]
    traj = compute_hayes_prefix_trajectory(
        release,
        prior=prior,
        eta=float(eta),
        checkpoints=checkpoints,
    )

    step_to_priv = {int(row["step"]): row for row in traj}
    steps = np.asarray(checkpoints, dtype=np.int64)
    utility = np.asarray([float(row[utility_key]) for row in hist], dtype=np.float64)
    gap = np.asarray(
        [
            float(step_to_priv[s]["gamma_ball"] - step_to_priv[s]["kappa"])
            for s in steps
        ],
        dtype=np.float64,
    )

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(steps, utility, marker="o", label=utility_key)
    ax1.set_xlabel("step")
    ax1.set_ylabel(utility_key)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(steps, gap, marker="s", label=r"$\Gamma(\kappa)-\kappa$")
    ax2.set_ylabel(r"$\Gamma(\kappa)-\kappa$")

    selected = release.extra.get(
        "selected_checkpoint_step",
        release.attack_metadata.get("selected_checkpoint_step", None),
    )
    if selected is not None:
        ax1.axvline(int(selected), linestyle="--", label="selected checkpoint")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")

    fig.suptitle(f"Utility vs prefix Ball-ReRo gap at eta={eta:g}")
    fig.tight_layout()
    plt.show()


def plot_sensitivity_diagnostics(release) -> None:
    """
    Mechanism-side figures:
      1) Delta_ball / Delta_std
      2) rho_t = Lz r / (2 C_t)
      3) c_t = Delta_t / nu_t
    """
    rows = get_release_step_table(release)
    steps = np.asarray([int(r["step"]) for r in rows], dtype=np.int64)

    ratio = np.asarray(
        [
            (
                np.nan
                if r["ball_to_standard_ratio"] is None
                else float(r["ball_to_standard_ratio"])
            )
            for r in rows
        ],
        dtype=np.float64,
    )
    rho = np.asarray(
        [np.nan if r["rho"] is None else float(r["rho"]) for r in rows],
        dtype=np.float64,
    )
    c_ball = np.asarray(
        [
            np.nan if r["direct_c_ball"] is None else float(r["direct_c_ball"])
            for r in rows
        ],
        dtype=np.float64,
    )
    c_std = np.asarray(
        [
            np.nan if r["direct_c_standard"] is None else float(r["direct_c_standard"])
            for r in rows
        ],
        dtype=np.float64,
    )

    plt.figure(figsize=(7, 4))
    plt.plot(steps, ratio, marker="o", label=r"$\Delta_t^{ball}/\Delta_t^{std}$")
    plt.axhline(1.0, linestyle="--", label="no Ball advantage")
    plt.xlabel("step")
    plt.ylabel("ratio")
    plt.title("Ball-vs-standard sensitivity ratio by step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(steps, rho, marker="o", label=r"$\rho_t = L_z r / (2 C_t)$")
    plt.axhline(1.0, linestyle="--", label=r"$\rho_t = 1$")
    plt.xlabel("step")
    plt.ylabel(r"$\rho_t$")
    plt.title("Ball saturation diagnostic by step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(steps, c_ball, marker="o", label=r"$c_t^{ball} = \Delta_t^{ball}/\nu_t$")
    plt.plot(steps, c_std, marker="o", label=r"$c_t^{std} = \Delta_t^{std}/\nu_t$")
    plt.xlabel("step")
    plt.ylabel(r"$c_t$")
    plt.title("Direct-profile step parameters")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_privacy_utility_frontier(
    experiments: Mapping[str, Mapping[str, object]],
    *,
    target_kappa: float,
    utility_key: str = "public_eval_accuracy",
) -> None:
    """
    experiments[name] should contain:
      {
        "release": release,
        "prior": prior,
        "eta_grid": eta_grid,
        # optional:
        "utility": scalar utility value
      }

    If "utility" is omitted, this function falls back to release.utility_metrics[utility_key].
    """
    xs = []
    ys = []
    labels = []

    for name, spec in experiments.items():
        release = spec["release"]
        prior = spec["prior"]
        eta_grid = spec["eta_grid"]

        report = ball_rero(
            release,
            prior=prior,
            eta_grid=eta_grid,
            mode="ball_sgd_hayes",
        )
        gamma_ball = gamma_at_kappa(report, float(target_kappa), which="ball")

        utility = spec.get("utility", None)
        if utility is None:
            utility = release.utility_metrics.get(utility_key, None)
        if utility is None:
            raise ValueError(
                f"Experiment {name!r} has no provided utility and release.utility_metrics "
                f"does not contain {utility_key!r}."
            )

        xs.append(float(utility))
        ys.append(float(gamma_ball - float(target_kappa)))
        labels.append(str(name))

    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys)
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y))
    plt.xlabel(utility_key)
    plt.ylabel(r"$\Gamma(\kappa) - \kappa$")
    plt.title(f"Privacy-utility frontier at kappa={target_kappa:g}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import numpy as np
    import jax.random as jr
    from sklearn.model_selection import train_test_split
    from quantbayes.fake_data import generate_binary_classification_data
    from quantbayes.ball_dp.utils import (
        build_same_label_finite_support,
        choose_target_index,
    )
    from quantbayes.ball_dp.theorem import (
        TheoremBounds,
        TheoremModelSpec,
        make_model,
        TrainConfig,
        fit_release,
    )
    from quantbayes.ball_dp.attacks import DPSGDTraceRecorder

    df = generate_binary_classification_data()
    preferred_target_index = 0
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    spec = TheoremModelSpec(
        d_in=X.shape[1],
        hidden_dim=250,
        task="binary",
        parameterization="dense",
        constraint="op",
    )
    target_index = choose_target_index(
        y_train,
        y_test,
        preferred=preferred_target_index,
        min_pool_per_label=8,
    )
    target_label = int(y_train[target_index])
    x_target = np.asarray(X_train[target_index], dtype=np.float32).reshape(-1)

    X_candidates, y_candidates = build_same_label_finite_support(
        x_target,
        target_label,
        X_test,
        y_test,
        max_candidates=16,
    )

    bounds = TheoremBounds(
        B=6.3,
        A=10.0,
        Lambda=10.0,
    )
    model = make_model(
        spec,
        key=jr.PRNGKey(0),
        init_project=True,
        bounds=bounds,
    )
    train_cfg = TrainConfig(
        radius=0.5,
        privacy="ball_rdp",
        # delta=target_delta,
        num_steps=3000,
        batch_size=256,
        batch_sampler="poisson",
        accountant_subsampling="match_sampler",
        clip_norm=100,
        noise_multiplier=1.1,
        learning_rate=0.01,
        eval_every=50,
        normalize_noisy_sum_by="batch_size",
        seed=0,
    )
    recorder = DPSGDTraceRecorder(
        capture_every=1,
        keep_models=True,
        keep_batch_indices=True,
    )

    release = fit_release(
        model,
        spec,
        bounds,
        X_train,
        y_train,
        X_eval=X_test,
        y_eval=y_test,
        train_cfg=train_cfg,
        orders=tuple(range(2, 13)),
        trace_recorder=recorder,
        record_operator_norms=True,
        operator_norms_every=1,
    )
    # Assume you already have:
    #   release = fit_ball_sgd(...)
    #   X_train, y_train available
    plot_ball_regime_threshold(release)
    assert_hayes_theorem_conditions(release)

    # Continuous prior for the publication-quality curve.
    prior, radius, dim = make_continuous_uniform_prior_for_release(release, X_train)

    # Dense eta-grid near the ball boundary so the induced kappa-grid is informative.
    eta_grid = make_eta_grid_for_uniform_ball(
        radius, lo_ratio=0.90, hi_ratio=0.999, n=80
    )

    # Final global/product theorem-backed report.
    report = ball_rero(
        release,
        prior=prior,
        eta_grid=eta_grid,
        mode="ball_sgd_hayes",
    )

    # 1) Full Gamma(kappa) curve
    plot_hayes_curve(report, show_gap=False, title="Final Ball-ReRo curve")

    # 2) Excess-over-baseline curve Gamma(kappa) - kappa
    plot_hayes_curve(report, show_gap=True, title="Final excess testing success")

    # 3) Mechanism-side diagnostics
    plot_sensitivity_diagnostics(release)

    # 4) Prefix privacy-vs-utility trajectory at one chosen operating point.
    #    Here I choose a fixed kappa and map it back to eta using the closed form
    #    for the uniform-ball prior.
    target_kappa = 1e-2
    eta_star = eta_for_uniform_ball_kappa(
        radius=radius, dimension=dim, kappa=target_kappa
    )

    plot_prefix_privacy_vs_utility(
        release,
        prior=prior,
        eta=eta_star,
        utility_key="public_eval_accuracy",  # or "public_eval_loss"
    )

    ### For finite prior
    from quantbayes.ball_dp import make_finite_identification_prior

    # X_candidates has shape (m_candidates, dim)
    finite_prior = make_finite_identification_prior(X_candidates)

    # Any eta < 1 gives the same certified kappa in this prior.
    report_finite = ball_rero(
        release,
        prior=finite_prior,
        eta_grid=(0.5,),
        mode="ball_sgd_hayes",
    )

    p = report_finite.points[0]
    print(
        {
            "kappa": p.kappa,
            "gamma_ball": p.gamma_ball,
            "gamma_standard": p.gamma_standard,
        }
    )
