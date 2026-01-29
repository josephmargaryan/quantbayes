from __future__ import annotations

from typing import Literal
import numpy as np

ScoreType = Literal["dot", "neg_l2"]


def bounded_replacement_radius(B: float) -> float:
    """
    If ||v|| <= B (public bound) then any replacement satisfies ||v - v'|| <= 2B.
    """
    B = float(B)
    if B <= 0:
        raise ValueError("B must be > 0.")
    return 2.0 * B


def score_sensitivity(
    score: ScoreType, *, r: float, q_norm_bound: float = 1.0
) -> float:
    """
    Sensitivity of per-index utility u(C,i) across ball-neighbor corpora C ~_r C',
    where one record moves by at most ||v - v'|| <= r.

    - dot: u = <q, v_i> => Δu <= ||q|| * r, so require a bound/clip on ||q||.
    - neg_l2: u = -||q - v_i|| => |u(v) - u(v')| <= ||v - v'|| <= r (triangle inequality),
              independent of q (nice).
    """
    r = float(r)
    if r < 0:
        raise ValueError("r must be >= 0.")
    q_norm_bound = float(q_norm_bound)
    if q_norm_bound <= 0:
        raise ValueError("q_norm_bound must be > 0.")

    if score == "dot":
        return q_norm_bound * r
    if score == "neg_l2":
        return r

    raise ValueError(f"Unknown score type: {score}")
