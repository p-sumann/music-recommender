"""Feedback API router.

Implements POST /feedback/{output_id} for recording user interactions.
Thread-safe with atomic counter updates.
"""

import logging
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.feedback import FeedbackRequest, FeedbackResponse, StatsResponse
from app.services.feedback import FeedbackService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("/{output_id}", response_model=FeedbackResponse)
async def record_feedback(
    output_id: UUID,
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Record user feedback (click, like, skip, etc.)

    This endpoint is **thread-safe**. Multiple concurrent requests
    will correctly increment counters using PostgreSQL atomic operations.

    **Effect on ranking:**
    - Clicks increment the output's click_count
    - Higher click counts improve popularity_score
    - Changes are reflected immediately in the next /search call

    **Position bias tracking:**
    - Provide position_shown to help calibrate position bias correction
    - This data is used to compute IPW weights for debiased ranking
    """
    feedback_service = FeedbackService(db)

    try:
        result = await feedback_service.record_interaction(
            output_id=output_id,
            action=request.action,
            position_shown=request.position_shown or 0,
            session_id=request.session_id,
            context=request.context,
        )

        # Get updated stats
        stats = await feedback_service.get_output_stats(output_id)

        return FeedbackResponse(
            success=True,
            interaction_id=UUID(result["interaction_id"]),
            output_id=output_id,
            action=request.action,
            recorded_at=datetime.fromisoformat(result["recorded_at"]),
            current_clicks=stats["click_count"] if stats else None,
            current_impressions=stats["impression_count"] if stats else None,
        )

    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record feedback: {str(e)}",
        )


@router.get("/{output_id}/stats", response_model=StatsResponse)
async def get_stats(
    output_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get engagement statistics for an output.
    """
    feedback_service = FeedbackService(db)
    stats = await feedback_service.get_output_stats(output_id)

    if stats is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No statistics found for output {output_id}",
        )

    return StatsResponse(
        output_id=UUID(stats["output_id"]),
        click_count=stats["click_count"],
        impression_count=stats["impression_count"],
        like_count=stats.get("like_count", 0),
        ctr_estimate=stats["ctr_estimate"],
        average_position=stats.get("average_position"),
        last_interaction=datetime.fromisoformat(stats["last_interaction"])
        if stats.get("last_interaction")
        else None,
    )


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "feedback"}
