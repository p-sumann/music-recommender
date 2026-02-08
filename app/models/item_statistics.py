"""Item engagement statistics model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Index, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.audio_output import AudioOutput


class ItemStatistics(Base):
    """Engagement statistics for audio outputs (separate table for fast updates)."""

    __tablename__ = "item_statistics"

    output_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("audio_outputs.id", ondelete="CASCADE"),
        primary_key=True,
    )

    impression_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    click_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    like_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    position_sum: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)

    ctr_estimate: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    ctr_variance: Mapped[float] = mapped_column(Float, default=0.25, nullable=False)

    last_interaction: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    stats_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    audio_output: Mapped["AudioOutput"] = relationship("AudioOutput", back_populates="statistics")

    __table_args__ = (
        Index("idx_stats_ctr_estimate", ctr_estimate.desc()),
        Index("idx_stats_click_count", click_count.desc()),
        Index("idx_stats_last_interaction", last_interaction.desc()),
    )

    def __repr__(self) -> str:
        return f"<ItemStatistics(output_id={self.output_id}, clicks={self.click_count})>"

    @property
    def average_position(self) -> float:
        if self.impression_count == 0:
            return 0.0
        return self.position_sum / self.impression_count

    def compute_ctr(self, alpha: float = 1.0, beta: float = 1.0) -> float:
        """Smoothed CTR using Beta prior."""
        a = alpha + self.click_count
        b = beta + max(self.impression_count - self.click_count, 0)
        return a / (a + b)

    def compute_variance(self, alpha: float = 1.0, beta: float = 1.0) -> float:
        """CTR variance for Thompson Sampling."""
        a = alpha + self.click_count
        b = beta + max(self.impression_count - self.click_count, 0)
        total = a + b
        return (a * b) / (total * total * (total + 1))
