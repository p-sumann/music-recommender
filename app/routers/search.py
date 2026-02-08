"""Search API router.

Implements POST /search with multi-stage ranking pipeline:
1. Candidate Retrieval (HNSW)
2. Composite Scoring (IPW + Thompson)
3. Neural Reranking (FlashRank)
4. Diversity (MMR)
"""

import logging
import time
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.schemas.search import (
    ScoreBreakdown,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.services.diversity import DiversityService
from app.services.neural_reranker import NeuralReranker
from app.services.ranking import RankingService
from app.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])
settings = get_settings()


@router.post("", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Search for music with multi-stage ranking.

    Pipeline stages:
    1. **Retrieval**: HNSW ANN search for semantic similarity (500 candidates)
    2. **Ranking**: Composite scoring with IPW debiasing and Thompson Sampling (50 candidates)
    3. **Reranking**: Neural cross-encoder reranking with FlashRank (30 candidates)
    4. **Diversity**: MMR diversification with genre slot allocation (20 results)

    The ranking formula:
    ```
    composite_score = 0.5 * semantic + 0.25 * popularity + 0.15 * exploration + 0.1 * freshness
    final_score = 0.6 * neural_rerank + 0.4 * composite_score
    ```
    """
    total_start = time.perf_counter()
    timings = {}

    # Build filters dict
    filters = {}
    if request.filters:
        filters = request.filters.model_dump(exclude_none=True)

    # Stage 1: Retrieval
    stage_start = time.perf_counter()
    retrieval_service = RetrievalService(db)
    candidates = await retrieval_service.retrieve_candidates(
        query=request.query,
        filters=filters,
    )
    timings["retrieval_ms"] = (time.perf_counter() - stage_start) * 1000
    total_candidates = len(candidates)

    if not candidates:
        return SearchResponse(
            query=request.query,
            results=[],
            total_candidates=0,
            **timings,
            filters_applied=request.filters,
        )

    # Stage 2: Ranking
    stage_start = time.perf_counter()
    ranking_service = RankingService()
    ranked = ranking_service.rank_candidates(candidates)
    timings["ranking_ms"] = (time.perf_counter() - stage_start) * 1000

    # Stage 2.5: Neural Reranking
    stage_start = time.perf_counter()
    reranker = NeuralReranker()
    reranked = await reranker.rerank(
        query=request.query,
        candidates=ranked,
    )
    timings["rerank_ms"] = (time.perf_counter() - stage_start) * 1000

    # Stage 3: Diversity
    stage_start = time.perf_counter()
    diversity_service = DiversityService()
    final = diversity_service.diversify(
        candidates=reranked,
        k=request.limit or settings.final_result_size,
    )
    timings["diversity_ms"] = (time.perf_counter() - stage_start) * 1000

    # Build response
    results = []
    for i, item in enumerate(final):
        scores = None
        if request.include_scores:
            scores = ScoreBreakdown(
                semantic_score=item.get("semantic_score", 0),
                popularity_score=item.get("popularity_score", 0),
                exploration_score=item.get("exploration_score", 0),
                freshness_score=item.get("freshness_score", 0),
                composite_score=item.get("composite_score", 0),
                neural_score=item.get("neural_score"),
                final_score=item.get("final_score"),
                mmr_score=item.get("mmr_score"),
                redundancy_score=item.get("redundancy_score"),
            )

        results.append(
            SearchResult(
                output_id=UUID(item["output_id"]),
                song_id=UUID(item["song_id"]),
                title=item["title"],
                audio_url=item["audio_url"],
                primary_genre=item.get("primary_genre"),
                primary_mood=item.get("primary_mood"),
                bpm=item.get("bpm"),
                musical_key=item.get("musical_key"),
                sounds_description=item.get("sounds_description"),
                acoustic_prompt_descriptive=item.get("acoustic_prompt_descriptive"),
                click_count=item.get("click_count") if request.include_scores else None,
                scores=scores,
                position=i + 1,
            )
        )

    timings["total_ms"] = (time.perf_counter() - total_start) * 1000

    logger.info(
        f"Search completed: query='{request.query[:30]}...', "
        f"results={len(results)}, total_ms={timings['total_ms']:.1f}"
    )

    return SearchResponse(
        query=request.query,
        results=results,
        total_candidates=total_candidates,
        filters_applied=request.filters,
        **timings,
    )


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "search"}
