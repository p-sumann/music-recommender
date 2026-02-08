"""User interaction logging model."""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Interaction(Base):
    """User interaction log (clicks, likes, impressions)."""

    __tablename__ = "interactions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    output_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("audio_outputs.id", ondelete="CASCADE"), nullable=False
    )

    action_type: Mapped[str] = mapped_column(String(50), nullable=False)
    position_shown: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    session_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    context: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("idx_interactions_output_id", output_id),
        Index("idx_interactions_created_at", created_at.desc()),
        Index(
            "idx_interactions_position_clicks",
            position_shown,
            postgresql_where=(action_type == "click"),
        ),
        Index("idx_interactions_session", session_id),
        Index("idx_interactions_action_type", action_type),
    )

    def __repr__(self) -> str:
        return f"<Interaction(output_id={self.output_id}, action={self.action_type})>"
