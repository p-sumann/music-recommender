"""Initial schema with pgvector, songs, audio_outputs, statistics, interactions.

Revision ID: 001
Revises:
Create Date: 2024-02-02

Schema Design:
- songs: Core metadata with vector embeddings (HNSW indexed)
- audio_outputs: Individual audio files (1 song -> N outputs)
- item_statistics: Hot statistics table (TikTok Monolith pattern)
- interactions: User action logs for bias analysis

Key Indexes:
- HNSW on songs.embedding for ANN search
- GIN on songs.extended_metadata for JSONB tag queries
- B-tree on structured filter columns
"""

from typing import Sequence, Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create songs table
    op.create_table(
        "songs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=True),
        sa.Column("lyrics", sa.Text(), nullable=True),
        sa.Column("acoustic_prompt_descriptive", sa.Text(), nullable=True),
        sa.Column("embedding", Vector(1536), nullable=True),  # OpenAI text-embedding-3-small
        sa.Column("bpm", sa.Integer(), nullable=True),
        sa.Column("musical_key", sa.String(50), nullable=True),
        sa.Column("primary_genre", sa.String(100), nullable=True),
        sa.Column("primary_mood", sa.String(100), nullable=True),
        sa.Column("format", sa.String(50), nullable=True),
        sa.Column("primary_context", sa.String(100), nullable=True),
        sa.Column("vocal_gender", sa.String(50), nullable=True),
        sa.Column("extended_metadata", postgresql.JSONB(), nullable=True, default={}),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Create audio_outputs table
    op.create_table(
        "audio_outputs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "song_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("songs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("output_number", sa.Integer(), nullable=False),
        sa.Column("audio_url", sa.String(1000), nullable=False),
        sa.Column("sounds_description", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Create item_statistics table (hot statistics - TikTok pattern)
    op.create_table(
        "item_statistics",
        sa.Column(
            "output_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("audio_outputs.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("impression_count", sa.BigInteger(), default=0, nullable=False),
        sa.Column("click_count", sa.BigInteger(), default=0, nullable=False),
        sa.Column("like_count", sa.BigInteger(), default=0, nullable=False),
        sa.Column("position_sum", sa.BigInteger(), default=0, nullable=False),
        sa.Column("ctr_estimate", sa.Float(), default=0.5, nullable=False),
        sa.Column("ctr_variance", sa.Float(), default=0.25, nullable=False),
        sa.Column("last_interaction", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "stats_updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Create interactions table (user action logs)
    op.create_table(
        "interactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "output_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("audio_outputs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("action_type", sa.String(50), nullable=False),
        sa.Column("position_shown", sa.Integer(), default=0, nullable=False),
        sa.Column("session_id", sa.String(100), nullable=True),
        sa.Column("context", postgresql.JSONB(), nullable=True, default={}),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Create indexes

    # HNSW index for vector similarity search (the key index!)
    op.execute("""
        CREATE INDEX idx_songs_embedding_hnsw ON songs
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 24, ef_construction = 100)
    """)

    # GIN index for JSONB tag search
    op.execute("""
        CREATE INDEX idx_songs_extended_metadata_gin ON songs
        USING gin (extended_metadata jsonb_path_ops)
    """)

    # B-tree indexes for structured filters
    op.create_index("idx_songs_bpm", "songs", ["bpm"])
    op.create_index("idx_songs_primary_genre", "songs", ["primary_genre"])
    op.create_index("idx_songs_primary_mood", "songs", ["primary_mood"])
    op.create_index("idx_songs_format", "songs", ["format"])
    op.create_index("idx_songs_created_at", "songs", [sa.text("created_at DESC")])

    # Audio outputs indexes
    op.create_index("idx_audio_outputs_song_id", "audio_outputs", ["song_id"])
    op.create_index("idx_audio_outputs_created_at", "audio_outputs", [sa.text("created_at DESC")])

    # Statistics indexes (hot path)
    op.create_index("idx_stats_ctr_estimate", "item_statistics", [sa.text("ctr_estimate DESC")])
    op.create_index("idx_stats_click_count", "item_statistics", [sa.text("click_count DESC")])
    op.create_index(
        "idx_stats_last_interaction", "item_statistics", [sa.text("last_interaction DESC")]
    )

    # Interaction indexes
    op.create_index("idx_interactions_output_id", "interactions", ["output_id"])
    op.create_index("idx_interactions_created_at", "interactions", [sa.text("created_at DESC")])
    op.create_index("idx_interactions_session", "interactions", ["session_id"])
    op.create_index("idx_interactions_action_type", "interactions", ["action_type"])

    # Partial index for position bias analysis (clicks only)
    op.execute("""
        CREATE INDEX idx_interactions_position_clicks ON interactions (position_shown)
        WHERE action_type = 'click'
    """)


def downgrade() -> None:
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table("interactions")
    op.drop_table("item_statistics")
    op.drop_table("audio_outputs")
    op.drop_table("songs")

    # Drop extension
    op.execute("DROP EXTENSION IF EXISTS vector")
