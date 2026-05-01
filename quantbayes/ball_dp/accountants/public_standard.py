# quantbayes/ball_dp/accountants/public_standard.py
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from ..types import DpCertificate, RdpCurve


def default_public_orders() -> tuple[float, ...]:
    return tuple([1.1 + 0.1 * i for i in range(99)] + [float(i) for i in range(11, 64)])


def _get_rdp_privacy_accountant_module():
    try:
        from dp_accounting import rdp
    except Exception as e:
        raise ImportError(
            "Public-standard Poisson accounting requires the official "
            "`dp-accounting` package. Install it with `pip install dp-accounting`."
        ) from e
    return rdp.rdp_privacy_accountant


def public_poisson_rdp_curve_from_schedule(
    *,
    step_batch_sizes: Sequence[int],
    dataset_size: int,
    step_clip_norms: Sequence[float],
    step_noise_stds: Sequence[float],
    orders: Optional[Sequence[float]] = None,
    source: str = "public_poisson_standard_sgd",
) -> RdpCurve:
    if not (len(step_batch_sizes) == len(step_clip_norms) == len(step_noise_stds)):
        raise ValueError("Step schedules must have the same length.")

    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive.")

    rpa = _get_rdp_privacy_accountant_module()
    orders = tuple(default_public_orders() if orders is None else orders)
    total_rdp = np.zeros(len(orders), dtype=np.float64)

    for batch_size, clip_norm, noise_std in zip(
        step_batch_sizes, step_clip_norms, step_noise_stds
    ):
        q = float(batch_size) / float(dataset_size)
        c = float(clip_norm)
        sigma = float(noise_std)

        if q == 0.0 or c == 0.0:
            continue

        noise_multiplier = sigma / c
        step_rdp = np.asarray(
            rpa._compute_rdp_poisson_subsampled_gaussian(
                q,
                noise_multiplier,
                orders,
            ),
            dtype=np.float64,
        )
        total_rdp += step_rdp

    return RdpCurve(
        orders=tuple(float(a) for a in orders),
        epsilons=tuple(float(v) for v in total_rdp.tolist()),
        source=source,
        radius=None,
    )


def public_poisson_dp_from_schedule(
    *,
    step_batch_sizes: Sequence[int],
    dataset_size: int,
    step_clip_norms: Sequence[float],
    step_noise_stds: Sequence[float],
    delta: float,
    orders: Optional[Sequence[float]] = None,
    source: str = "public_poisson_standard_sgd",
) -> dict[str, Any]:
    curve = public_poisson_rdp_curve_from_schedule(
        step_batch_sizes=step_batch_sizes,
        dataset_size=dataset_size,
        step_clip_norms=step_clip_norms,
        step_noise_stds=step_noise_stds,
        orders=orders,
        source=source,
    )

    rpa = _get_rdp_privacy_accountant_module()
    eps, alpha = rpa.compute_epsilon(curve.orders, curve.epsilons, float(delta))

    cert = DpCertificate(
        epsilon=float(eps),
        delta=float(delta),
        source=source,
        radius=None,
        order_opt=float(alpha),
        note=(
            "Public-standard accounting: Poisson-sampled Gaussian DP-SGD with "
            "add/remove adjacency, using the official dp-accounting RDP routines."
        ),
    )

    noise_multipliers = []
    for c, sigma in zip(step_clip_norms, step_noise_stds):
        c = float(c)
        sigma = float(sigma)
        noise_multipliers.append(0.0 if c == 0.0 else float(sigma / c))

    return {
        "rdp_curve": curve,
        "dp_certificate": cert,
        "orders": tuple(curve.orders),
        "epsilons": tuple(curve.epsilons),
        "noise_multipliers": tuple(noise_multipliers),
        "sample_rates": tuple(float(b) / float(dataset_size) for b in step_batch_sizes),
    }


def public_poisson_dp_from_release(
    release: Any,
    *,
    delta: float,
    dataset_size: Optional[int] = None,
    orders: Optional[Sequence[float]] = None,
    source: str = "public_poisson_standard_sgd",
) -> dict[str, Any]:
    cfg = release.training_config
    n = int(
        release.dataset_metadata["n_total"] if dataset_size is None else dataset_size
    )

    return public_poisson_dp_from_schedule(
        step_batch_sizes=tuple(int(v) for v in cfg["batch_sizes"]),
        dataset_size=n,
        step_clip_norms=tuple(float(v) for v in cfg["clip_norms"]),
        step_noise_stds=tuple(float(v) for v in cfg["noise_stds"]),
        delta=float(delta),
        orders=orders,
        source=source,
    )
