"""Feedback API schemas."""

from datetime import datetime
from typing import Any, Dict, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """Feedback request payload."""

    action: Literal["click", "like", "skip", "play_complete"] = Field(
        ..., description="Type of user action"
    )
    position_shown: Optional[int] = Field(
        0, ge=0, le=100, description="Position where item was shown (for position bias tracking)"
    )
    session_id: Optional[str] = Field(None, max_length=100, description="User session identifier")
    context: Optional[Dict[str, Any]] = Field(
        None, description="Additional context (device, search query, etc.)"
    )


class FeedbackResponse(BaseModel):
    """Feedback response."""

    success: bool = Field(..., description="Whether feedback was recorded")
    interaction_id: UUID = Field(..., description="Created interaction ID")
    output_id: UUID = Field(..., description="Audio output ID")
    action: str = Field(..., description="Recorded action type")
    recorded_at: datetime = Field(..., description="Timestamp when recorded")

    # Updated stats (optional)
    current_clicks: Optional[int] = Field(None, description="Current click count")
    current_impressions: Optional[int] = Field(None, description="Current impression count")


class StatsResponse(BaseModel):
    """Statistics response."""

    output_id: UUID
    click_count: int
    impression_count: int
    like_count: int
    ctr_estimate: float
    average_position: Optional[float] = None
    last_interaction: Optional[datetime] = None
