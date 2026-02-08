"""Composite scoring with position bias correction and Thompson Sampling."""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.config import get_settings
from app.core.position_bias import PositionBiasCorrector
from app.core.thompson_sampling import ThompsonSampler

logger = logging.getLogger(__name__)
settings = get_settings()


class RankingService:
    """Multi-signal ranking: semantic + popularity + exploration + freshness."""

    def __init__(self):
        self.thompson_sampler = ThompsonSampler(
            prior_alpha=settings.thompson_prior_alpha,
            prior_beta=settings.thompson_prior_beta,
        )
        self.bias_corrector = PositionBiasCorrector(
            propensities=settings.position_propensities,
            default_propensity=settings.default_propensity,
        )

    def compute_composite_score(
        self,
        semantic_score: float,
        click_count: int,
        impression_count: int,
        position_sum: int,
        created_at: datetime,
        max_clicks: int = 1,
        use_thompson: bool = True,
    ) -> Dict[str, float]:
        """Compute weighted composite score."""
        semantic = semantic_score

        if impression_count > 0:
            debiased_ctr = self.bias_corrector.compute_simplified_debiased_ctr(
                clicks=click_count, impressions=impression_count, position_sum=position_sum
            )
            popularity = min(debiased_ctr, 1.0)
        else:
            popularity = 0.5

        exploration = self.thompson_sampler.compute_exploration_score(
            clicks=click_count, impressions=impression_count, use_ucb=not use_thompson
        )

        if created_at:
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - created_at).total_seconds() / 86400
            freshness = math.exp(-settings.freshness_decay_rate * age_days)
        else:
            freshness = 0.5

        composite = (
            settings.weight_semantic * semantic
            + settings.weight_popularity * popularity
            + settings.weight_exploration * exploration
            + settings.weight_freshness * freshness
        )

        return {
            "semantic_score": semantic,
            "popularity_score": popularity,
            "exploration_score": exploration,
            "freshness_score": freshness,
            "composite_score": composite,
        }

    def rank_candidates(
        self, candidates: List[Dict[str, Any]], limit: int = None
    ) -> List[Dict[str, Any]]:
        """Rank candidates by composite score."""
        limit = limit or settings.composite_pool_size

        if not candidates:
            return []

        max_clicks = max(c.get("click_count", 0) for c in candidates) or 1

        scored = []
        for candidate in candidates:
            scores = self.compute_composite_score(
                semantic_score=candidate.get("semantic_score", 0),
                click_count=candidate.get("click_count", 0),
                impression_count=candidate.get("impression_count", 0),
                position_sum=candidate.get("position_sum", 0),
                created_at=candidate.get("created_at"),
                max_clicks=max_clicks,
            )
            scored.append({**candidate, **scores})

        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        logger.info(f"Ranked {len(candidates)} candidates, returning top {min(limit, len(scored))}")
        return scored[:limit]
