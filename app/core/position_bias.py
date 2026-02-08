"""Position Bias Correction using Inverse Propensity Weighting (IPW)."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class DebiasedMetrics:
    raw_clicks: int
    raw_impressions: int
    debiased_clicks: float
    debiased_ctr: float
    average_position: float
    confidence: float


class PositionBiasCorrector:
    """IPW-based position bias correction."""

    DEFAULT_PROPENSITIES = {
        1: 1.00,
        2: 0.70,
        3: 0.50,
        4: 0.35,
        5: 0.25,
        6: 0.18,
        7: 0.13,
        8: 0.10,
        9: 0.08,
        10: 0.06,
        11: 0.05,
        12: 0.04,
        13: 0.035,
        14: 0.03,
        15: 0.025,
        16: 0.02,
        17: 0.018,
        18: 0.016,
        19: 0.014,
        20: 0.012,
    }

    def __init__(
        self,
        propensities: Optional[Dict[int, float]] = None,
        default_propensity: float = 0.01,
        max_ipw_weight: float = 20.0,
    ):
        self.propensities = propensities or self.DEFAULT_PROPENSITIES
        self.default_propensity = default_propensity
        self.max_ipw_weight = max_ipw_weight

    def get_propensity(self, position: int) -> float:
        return self.propensities.get(position, self.default_propensity)

    def compute_ipw_weight(self, position: int) -> float:
        propensity = self.get_propensity(position)
        weight = 1.0 / max(propensity, 1e-6)
        return min(weight, self.max_ipw_weight)

    def debias_click(self, clicked: bool, position: int) -> float:
        if not clicked:
            return 0.0
        return self.compute_ipw_weight(position)

    def compute_debiased_ctr(self, clicks_by_position: List[Tuple[int, bool]]) -> DebiasedMetrics:
        """Compute debiased CTR from (position, clicked) pairs."""
        if not clicks_by_position:
            return DebiasedMetrics(0, 0, 0.0, 0.5, 0.0, 0.0)

        raw_clicks = sum(1 for _, clicked in clicks_by_position if clicked)
        raw_impressions = len(clicks_by_position)

        debiased_clicks = sum(
            self.debias_click(clicked, position) for position, clicked in clicks_by_position
        )

        total_weight = sum(self.compute_ipw_weight(position) for position, _ in clicks_by_position)

        debiased_ctr = debiased_clicks / max(total_weight, 1e-6)
        average_position = sum(p for p, _ in clicks_by_position) / raw_impressions
        confidence = min(1.0, math.sqrt(raw_impressions) / 10)

        return DebiasedMetrics(
            raw_clicks=raw_clicks,
            raw_impressions=raw_impressions,
            debiased_clicks=debiased_clicks,
            debiased_ctr=min(1.0, debiased_ctr),
            average_position=average_position,
            confidence=confidence,
        )

    def compute_simplified_debiased_ctr(
        self, clicks: int, impressions: int, position_sum: int
    ) -> float:
        """Fast debiased CTR using average position."""
        if impressions == 0:
            return 0.5

        avg_position = position_sum / impressions
        avg_propensity = self.get_propensity(round(avg_position))
        raw_ctr = clicks / impressions
        debiased_ctr = raw_ctr / max(avg_propensity, 0.01)

        return min(1.0, debiased_ctr)


def calibrate_propensities_from_data(
    click_data: List[Tuple[int, bool]], smoothing: float = 1.0
) -> Dict[int, float]:
    """Calibrate propensity model from click data."""
    from collections import defaultdict

    position_clicks = defaultdict(int)
    position_impressions = defaultdict(int)

    for position, clicked in click_data:
        position_impressions[position] += 1
        if clicked:
            position_clicks[position] += 1

    propensities = {}
    for position in position_impressions:
        clicks = position_clicks[position] + smoothing
        impressions = position_impressions[position] + 2 * smoothing
        propensities[position] = clicks / impressions

    if 1 in propensities and propensities[1] > 0:
        max_prop = propensities[1]
        propensities = {p: v / max_prop for p, v in propensities.items()}

    return propensities
