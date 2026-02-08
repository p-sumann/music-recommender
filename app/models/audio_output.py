"""AudioOutput model for individual audio files."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.item_statistics import ItemStatistics
    from app.models.song import Song


class AudioOutput(Base):
    """Audio output file from a song generation."""

    __tablename__ = "audio_outputs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    song_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("songs.id", ondelete="CASCADE"), nullable=False
    )

    output_number: Mapped[int] = mapped_column(Integer, nullable=False)
    audio_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    sounds_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    song: Mapped["Song"] = relationship("Song", back_populates="audio_outputs")
    statistics: Mapped[Optional["ItemStatistics"]] = relationship(
        "ItemStatistics", back_populates="audio_output", uselist=False,
        cascade="all, delete-orphan", lazy="joined"
    )

    __table_args__ = (
        Index("idx_audio_outputs_song_id", song_id),
        Index("idx_audio_outputs_created_at", created_at.desc()),
    )

    def __repr__(self) -> str:
        return f"<AudioOutput(id={self.id}, output={self.output_number})>"
