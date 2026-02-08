"""Song model with vector embeddings."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.audio_output import AudioOutput


class Song(Base):
    """Song representing a music generation request."""

    __tablename__ = "songs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    lyrics: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    acoustic_prompt_descriptive: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1536), nullable=True)

    bpm: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    musical_key: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    primary_genre: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    primary_mood: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    format: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    primary_context: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    vocal_gender: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    extended_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    audio_outputs: Mapped[List["AudioOutput"]] = relationship(
        "AudioOutput", back_populates="song", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        Index(
            "idx_songs_embedding_hnsw", embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 24, "ef_construction": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("idx_songs_extended_metadata_gin", extended_metadata, postgresql_using="gin"),
        Index("idx_songs_bpm", bpm),
        Index("idx_songs_primary_genre", primary_genre),
        Index("idx_songs_primary_mood", primary_mood),
        Index("idx_songs_format", format),
        Index("idx_songs_created_at", created_at.desc()),
    )

    def __repr__(self) -> str:
        return f"<Song(id={self.id}, title='{self.title[:30]}...')>"
