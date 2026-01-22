# cohort_dp/accountant.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PrivacyAccountant:
    """
    Minimal privacy accountant (basic composition).
    - eps == 0 is allowed (no-op), which is useful for non-private baselines.
    Replace with advanced composition / RDP later if needed.
    """

    eps_budget: float
    eps_spent: float = 0.0

    def can_spend(self, eps: float) -> bool:
        return (self.eps_spent + eps) <= self.eps_budget

    def spend(self, eps: float) -> None:
        if eps < 0:
            raise ValueError("eps must be >= 0.")
        if eps == 0:
            return
        if not self.can_spend(eps):
            raise RuntimeError(
                f"Privacy budget exceeded: need {eps:.4f}, remaining {self.eps_budget - self.eps_spent:.4f}."
            )
        self.eps_spent += eps
