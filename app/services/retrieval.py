"""HNSW vector search retrieval."""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import AudioOutput, ItemStatistics, Song
from app.services.cache import EmbeddingCache
from app.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)
settings = get_settings()


class RetrievalService:
    """HNSW-based candidate retrieval with optional filtering."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedding_service = EmbeddingService()
        self.embedding_cache = EmbeddingCache()

    async def retrieve_candidates(
        self,
        query: str,
        limit: int = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve candidates using HNSW vector search."""
        limit = limit or settings.candidate_pool_size
        filters = filters or {}

        query_embedding = await self.embedding_cache.get_or_compute(
            query, self.embedding_service.embed_query
        )
        query_embedding_list = query_embedding.tolist()

        filter_conditions = []
        if filters.get("genre"):
            filter_conditions.append(Song.primary_genre == filters["genre"])
        if filters.get("mood"):
            filter_conditions.append(Song.primary_mood == filters["mood"])
        if filters.get("format"):
            filter_conditions.append(Song.format == filters["format"])
        if filters.get("bpm_min"):
            filter_conditions.append(Song.bpm >= filters["bpm_min"])
        if filters.get("bpm_max"):
            filter_conditions.append(Song.bpm <= filters["bpm_max"])

        cosine_distance = Song.embedding.cosine_distance(query_embedding_list)

        base_query = (
            select(
                Song.id.label("song_id"),
                Song.title,
                Song.acoustic_prompt_descriptive,
                Song.embedding,
                Song.bpm,
                Song.musical_key,
                Song.primary_genre,
                Song.primary_mood,
                Song.format,
                Song.primary_context,
                Song.created_at,
                AudioOutput.id.label("output_id"),
                AudioOutput.output_number,
                AudioOutput.audio_url,
                AudioOutput.sounds_description,
                ItemStatistics.click_count,
                ItemStatistics.impression_count,
                ItemStatistics.like_count,
                ItemStatistics.position_sum,
                ItemStatistics.ctr_estimate,
                ItemStatistics.ctr_variance,
                (1 - cosine_distance).label("semantic_score"),
            )
            .join(AudioOutput, Song.id == AudioOutput.song_id)
            .outerjoin(ItemStatistics, AudioOutput.id == ItemStatistics.output_id)
        )

        if filter_conditions:
            base_query = base_query.where(and_(*filter_conditions))

        base_query = base_query.order_by(cosine_distance).limit(limit)
        result = await self.db.execute(base_query)
        rows = result.fetchall()

        candidates = []
        for row in rows:
            candidates.append(
                {
                    "song_id": str(row.song_id),
                    "output_id": str(row.output_id),
                    "title": row.title,
                    "acoustic_prompt_descriptive": row.acoustic_prompt_descriptive,
                    "embedding": row.embedding,
                    "bpm": row.bpm,
                    "musical_key": row.musical_key,
                    "primary_genre": row.primary_genre,
                    "primary_mood": row.primary_mood,
                    "format": row.format,
                    "primary_context": row.primary_context,
                    "created_at": row.created_at,
                    "output_number": row.output_number,
                    "audio_url": row.audio_url,
                    "sounds_description": row.sounds_description,
                    "click_count": row.click_count or 0,
                    "impression_count": row.impression_count or 0,
                    "like_count": row.like_count or 0,
                    "position_sum": row.position_sum or 0,
                    "ctr_estimate": row.ctr_estimate or 0.5,
                    "ctr_variance": row.ctr_variance or 0.25,
                    "semantic_score": float(row.semantic_score) if row.semantic_score else 0.0,
                }
            )

        logger.info(f"Retrieved {len(candidates)} candidates")
        return candidates

    async def retrieve_by_ids(self, output_ids: List[UUID]) -> List[Dict[str, Any]]:
        """Retrieve specific outputs by ID."""
        query = (
            select(
                Song.id.label("song_id"),
                Song.title,
                Song.embedding,
                Song.primary_genre,
                Song.primary_mood,
                AudioOutput.id.label("output_id"),
                AudioOutput.audio_url,
                AudioOutput.sounds_description,
                ItemStatistics.click_count,
                ItemStatistics.impression_count,
            )
            .join(AudioOutput, Song.id == AudioOutput.song_id)
            .outerjoin(ItemStatistics, AudioOutput.id == ItemStatistics.output_id)
            .where(AudioOutput.id.in_(output_ids))
        )

        result = await self.db.execute(query)
        return [
            {
                "song_id": str(row.song_id),
                "output_id": str(row.output_id),
                "title": row.title,
                "embedding": row.embedding,
                "primary_genre": row.primary_genre,
                "primary_mood": row.primary_mood,
                "audio_url": row.audio_url,
                "sounds_description": row.sounds_description,
                "click_count": row.click_count or 0,
                "impression_count": row.impression_count or 0,
            }
            for row in result.fetchall()
        ]
