"""Statistics service for aggregation and analysis.

Handles:
- CTR computation and updates
- Position bias calibration
- Global statistics for normalization
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import Interaction, ItemStatistics

logger = logging.getLogger(__name__)
settings = get_settings()


class StatisticsService:
    """
    Service for statistics aggregation and analysis.
    """

    def __init__(self, db: AsyncSession):
        """Initialize statistics service."""
        self.db = db

    async def get_global_stats(self) -> Dict[str, Any]:
        """
        Get global statistics for normalization.

        Returns:
            Dict with max_clicks, avg_clicks, total_items, etc.
        """
        result = await self.db.execute(
            select(
                func.max(ItemStatistics.click_count).label("max_clicks"),
                func.avg(ItemStatistics.click_count).label("avg_clicks"),
                func.sum(ItemStatistics.click_count).label("total_clicks"),
                func.sum(ItemStatistics.impression_count).label("total_impressions"),
                func.count(ItemStatistics.output_id).label("total_items"),
            )
        )
        row = result.one()

        return {
            "max_clicks": row.max_clicks or 0,
            "avg_clicks": float(row.avg_clicks or 0),
            "total_clicks": row.total_clicks or 0,
            "total_impressions": row.total_impressions or 0,
            "total_items": row.total_items or 0,
            "global_ctr": (
                row.total_clicks / row.total_impressions if row.total_impressions else 0
            ),
        }

    async def update_ctr_estimates(self, batch_size: int = 1000) -> int:
        """
        Update CTR estimates for all items.

        Should be run periodically (e.g., every hour) to update
        pre-computed CTR values.

        Args:
            batch_size: Number of items to update per batch

        Returns:
            Number of items updated
        """
        # Update CTR estimate using Bayesian average
        alpha = settings.thompson_prior_alpha
        beta = settings.thompson_prior_beta

        # Raw SQL for efficiency
        update_sql = f"""
            UPDATE item_statistics
            SET
                ctr_estimate = ({alpha} + click_count)::float /
                              ({alpha} + {beta} + impression_count),
                ctr_variance = (({alpha} + click_count) * ({beta} + impression_count - click_count))::float /
                              ((({alpha} + {beta} + impression_count)::float) ^ 2 *
                               ({alpha} + {beta} + impression_count + 1)),
                stats_updated_at = NOW()
            WHERE impression_count > 0
        """

        result = await self.db.execute(text(update_sql))
        await self.db.commit()

        updated_count = result.rowcount
        logger.info(f"Updated CTR estimates for {updated_count} items")
        return updated_count

    async def get_position_click_distribution(
        self,
        days: int = 30,
    ) -> Dict[int, Dict[str, int]]:
        """
        Get click distribution by position for bias calibration.

        Args:
            days: Number of days to look back

        Returns:
            Dict mapping position -> {clicks, impressions}
        """
        cutoff = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff = cutoff.replace(day=cutoff.day - days)

        # Get click counts by position
        clicks_result = await self.db.execute(
            select(
                Interaction.position_shown,
                func.count(Interaction.id).label("count"),
            )
            .where(Interaction.action_type == "click")
            .where(Interaction.created_at >= cutoff)
            .where(Interaction.position_shown > 0)
            .where(Interaction.position_shown <= 20)
            .group_by(Interaction.position_shown)
        )
        clicks_by_position = {row.position_shown: row.count for row in clicks_result}

        # Get impression counts by position
        impressions_result = await self.db.execute(
            select(
                Interaction.position_shown,
                func.count(Interaction.id).label("count"),
            )
            .where(Interaction.action_type.in_(["click", "impression", "skip"]))
            .where(Interaction.created_at >= cutoff)
            .where(Interaction.position_shown > 0)
            .where(Interaction.position_shown <= 20)
            .group_by(Interaction.position_shown)
        )
        impressions_by_position = {row.position_shown: row.count for row in impressions_result}

        # Combine
        distribution = {}
        for pos in range(1, 21):
            distribution[pos] = {
                "clicks": clicks_by_position.get(pos, 0),
                "impressions": impressions_by_position.get(pos, 0),
            }

        return distribution

    async def calibrate_position_propensities(
        self,
        days: int = 30,
    ) -> Dict[int, float]:
        """
        Calibrate position bias propensities from click data.

        Computes: propensity[pos] = CTR[pos] / CTR[1]

        Args:
            days: Number of days of data to use

        Returns:
            Dict mapping position -> propensity
        """
        distribution = await self.get_position_click_distribution(days)

        # Compute CTR per position
        ctrs = {}
        for pos, data in distribution.items():
            if data["impressions"] > 0:
                ctrs[pos] = data["clicks"] / data["impressions"]
            else:
                ctrs[pos] = 0

        # Normalize by position 1 CTR
        if ctrs.get(1, 0) > 0:
            base_ctr = ctrs[1]
            propensities = {pos: ctr / base_ctr for pos, ctr in ctrs.items()}
        else:
            # Fall back to defaults if no data
            propensities = settings.position_propensities.copy()

        logger.info(f"Calibrated propensities: {propensities}")
        return propensities

    async def get_top_items(
        self,
        limit: int = 100,
        metric: str = "clicks",
    ) -> List[Dict[str, Any]]:
        """
        Get top items by engagement metric.

        Args:
            limit: Number of items to return
            metric: Metric to sort by (clicks, impressions, ctr)

        Returns:
            List of top items with stats
        """
        from app.models import AudioOutput, Song

        order_col = {
            "clicks": ItemStatistics.click_count,
            "impressions": ItemStatistics.impression_count,
            "ctr": ItemStatistics.ctr_estimate,
        }.get(metric, ItemStatistics.click_count)

        result = await self.db.execute(
            select(
                Song.title,
                AudioOutput.id.label("output_id"),
                AudioOutput.audio_url,
                ItemStatistics.click_count,
                ItemStatistics.impression_count,
                ItemStatistics.ctr_estimate,
            )
            .join(AudioOutput, Song.id == AudioOutput.song_id)
            .join(ItemStatistics, AudioOutput.id == ItemStatistics.output_id)
            .order_by(order_col.desc())
            .limit(limit)
        )

        return [
            {
                "title": row.title,
                "output_id": str(row.output_id),
                "audio_url": row.audio_url,
                "clicks": row.click_count,
                "impressions": row.impression_count,
                "ctr": row.ctr_estimate,
            }
            for row in result
        ]
