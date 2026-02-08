"""Thompson Sampling for exploration/exploitation."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ThompsonSample:
    item_id: str
    sampled_ctr: float
    mean_ctr: float
    variance: float
    exploration_bonus: float


class ThompsonSampler:
    """Beta-Bernoulli Thompson Sampling for CTR estimation."""

    def __init__(
        self, prior_alpha: float = 1.0, prior_beta: float = 1.0, exploration_boost: float = 0.1
    ):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.exploration_boost = exploration_boost

    def sample(self, clicks: int, impressions: int) -> ThompsonSample:
        """Sample CTR from Beta posterior."""
        alpha = self.prior_alpha + clicks
        beta = self.prior_beta + max(impressions - clicks, 0)

        sampled_ctr = np.random.beta(alpha, beta)
        mean_ctr = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        exploration_bonus = self.exploration_boost * np.sqrt(variance)

        return ThompsonSample(
            item_id="",
            sampled_ctr=sampled_ctr,
            mean_ctr=mean_ctr,
            variance=variance,
            exploration_bonus=exploration_bonus,
        )

    def sample_batch(self, items: List[Tuple[str, int, int]]) -> List[ThompsonSample]:
        """Sample CTRs for batch of (item_id, clicks, impressions)."""
        results = []
        for item_id, clicks, impressions in items:
            sample = self.sample(clicks, impressions)
            sample.item_id = item_id
            results.append(sample)
        return results

    def compute_exploration_score(
        self, clicks: int, impressions: int, use_ucb: bool = True
    ) -> float:
        """Compute exploration score (UCB or Thompson sample)."""
        alpha = self.prior_alpha + clicks
        beta = self.prior_beta + max(impressions - clicks, 0)

        if use_ucb:
            mean = alpha / (alpha + beta)
            variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            return min(1.0, mean + 2 * np.sqrt(variance))
        return np.random.beta(alpha, beta)

    @staticmethod
    def compute_beta_parameters(
        clicks: int, impressions: int, prior_alpha: float = 1.0, prior_beta: float = 1.0
    ) -> Tuple[float, float]:
        """Compute Beta distribution parameters."""
        return prior_alpha + clicks, prior_beta + max(impressions - clicks, 0)


def get_exploration_tier(impressions: int) -> str:
    """Categorize item by exploration need: cold/warm/hot."""
    if impressions < 10:
        return "cold"
    elif impressions < 100:
        return "warm"
    return "hot"
