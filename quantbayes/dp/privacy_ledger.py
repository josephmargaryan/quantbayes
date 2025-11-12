from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Sequence
import numpy as np

from .accounting import zcdp_to_epsdelta
from .accounting_rdp import rdp_to_eps
from .accounting_rdp_subsampled import rdp_to_eps as rdp_to_eps_sub


@dataclass
class PrivacyLedger:
    """
    Minimal ledger that composes either:
      - zCDP: sum rho, convert on request
      - RDP: store (alpha, rdp_total) pairs and sum per alpha
      - subsampled-RDP: same as RDP (uses same conversion)
    """

    rho_events: List[float] = field(default_factory=list)
    rdp_orders: List[float] = field(default_factory=list)
    rdp_values: List[float] = field(default_factory=list)  # aligned with rdp_orders

    def add_zcdp(self, rho: float) -> None:
        assert rho >= 0.0
        self.rho_events.append(rho)

    def add_rdp(self, order: float, value: float) -> None:
        assert order > 1.0 and value >= 0.0
        self.rdp_orders.append(order)
        self.rdp_values.append(value)

    def eps_delta_from_zcdp(self, delta: float) -> float:
        rho = float(sum(self.rho_events))
        return zcdp_to_epsdelta(rho, delta)

    def eps_delta_from_rdp(self, delta: float) -> Tuple[float, float]:
        if not self.rdp_orders:
            return 0.0, 2.0
        # group by order and sum
        orders = np.array(self.rdp_orders)
        values = np.array(self.rdp_values)
        uniq = np.unique(orders)
        tot = []
        for a in uniq:
            tot.append(values[orders == a].sum())
        eps, a_star = rdp_to_eps(np.array(tot), np.array(uniq), delta)
        return float(eps), float(a_star)
