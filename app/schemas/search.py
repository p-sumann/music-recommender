"""Search API schemas."""

from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    """Search filter options."""

    genre: Optional[str] = Field(None, description="Filter by primary genre")
    mood: Optional[str] = Field(None, description="Filter by primary mood")
    format: Optional[str] = Field(None, description="Filter by format (MUSIC, SFX)")
    bpm_min: Optional[int] = Field(None, ge=20, le=300, description="Minimum BPM")
    bpm_max: Optional[int] = Field(None, ge=20, le=300, description="Maximum BPM")


class SearchRequest(BaseModel):
    """Search request payload."""

    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    filters: Optional[SearchFilters] = Field(None, description="Optional filters")
    limit: Optional[int] = Field(20, ge=1, le=100, description="Number of results")
    include_scores: Optional[bool] = Field(False, description="Include score breakdown")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")


class ScoreBreakdown(BaseModel):
    """Score component breakdown."""

    semantic_score: float = Field(..., description="Semantic similarity (0-1)")
    popularity_score: float = Field(..., description="Popularity score (0-1)")
    exploration_score: float = Field(..., description="Exploration bonus (0-1)")
    freshness_score: float = Field(..., description="Freshness decay (0-1)")
    composite_score: float = Field(..., description="Weighted composite score")
    neural_score: Optional[float] = Field(None, description="Cross-encoder score")
    final_score: Optional[float] = Field(None, description="Final blended score")
    mmr_score: Optional[float] = Field(None, description="MMR diversity score")
    redundancy_score: Optional[float] = Field(None, description="Redundancy to selected")


class SearchResult(BaseModel):
    """Individual search result."""

    output_id: UUID = Field(..., description="Audio output ID")
    song_id: UUID = Field(..., description="Parent song ID")
    title: str = Field(..., description="Song title")
    audio_url: str = Field(..., description="Audio file URL")

    # Metadata
    primary_genre: Optional[str] = Field(None, description="Primary genre")
    primary_mood: Optional[str] = Field(None, description="Primary mood")
    bpm: Optional[int] = Field(None, description="BPM")
    musical_key: Optional[str] = Field(None, description="Musical key")

    # Descriptions
    sounds_description: Optional[str] = Field(None, description="Sound description")
    acoustic_prompt_descriptive: Optional[str] = Field(None, description="Acoustic description")

    # Engagement (optional)
    click_count: Optional[int] = Field(None, description="Click count")

    # Scores (optional, controlled by include_scores)
    scores: Optional[ScoreBreakdown] = Field(None, description="Score breakdown")

    # Position in results
    position: int = Field(..., description="Position in search results (1-indexed)")


class SearchResponse(BaseModel):
    """Search response."""

    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="Search results")
    total_candidates: int = Field(..., description="Total candidates retrieved")

    # Timing
    retrieval_ms: Optional[float] = Field(None, description="Retrieval stage time (ms)")
    ranking_ms: Optional[float] = Field(None, description="Ranking stage time (ms)")
    rerank_ms: Optional[float] = Field(None, description="Neural rerank time (ms)")
    diversity_ms: Optional[float] = Field(None, description="Diversity stage time (ms)")
    total_ms: Optional[float] = Field(None, description="Total processing time (ms)")

    # Metadata
    filters_applied: Optional[SearchFilters] = Field(None, description="Filters applied")
