"""MMR-based diversity reranking."""

import logging
from typing import Any, Dict, List

import numpy as np

from app.config import get_settings
from app.core.mmr import MMRCandidate, MMRDiversifier, allocate_genre_slots

logger = logging.getLogger(__name__)
settings = get_settings()


class DiversityService:
    """MMR diversity reranking to avoid redundant results."""

    def __init__(self, lambda_relevance: float = None):
        self.lambda_relevance = lambda_relevance or settings.mmr_lambda
        self.diversifier = MMRDiversifier(lambda_relevance=self.lambda_relevance)

    def diversify(
        self,
        candidates: List[Dict[str, Any]],
        k: int = None,
        use_genre_slots: bool = True,
    ) -> List[Dict[str, Any]]:
        """Apply MMR diversification to candidates."""
        k = k or settings.final_result_size

        if not candidates:
            return []

        if len(candidates) <= k:
            return candidates

        genre_slots = None
        if use_genre_slots:
            genre_slots = allocate_genre_slots(
                candidates=candidates,
                total_slots=k,
                min_per_genre=2,
                genre_key="primary_genre",
            )
            logger.debug(f"Genre slots: {genre_slots}")

        mmr_candidates = []
        for candidate in candidates:
            embedding = candidate.get("embedding")
            if embedding is None:
                continue

            if isinstance(embedding, list):
                embedding = np.array(embedding)

            relevance_score = candidate.get("final_score") or candidate.get("composite_score", 0)

            mmr_candidates.append(
                MMRCandidate(
                    id=str(candidate.get("output_id", "")),
                    relevance_score=relevance_score,
                    embedding=embedding,
                    metadata=candidate,
                )
            )

        mmr_results = self.diversifier.diversify(
            candidates=mmr_candidates,
            k=k,
            genre_slots=genre_slots,
        )

        result_ids = {r.id: r for r in mmr_results}

        output = []
        for candidate in candidates:
            output_id = str(candidate.get("output_id", ""))
            if output_id in result_ids:
                mmr_result = result_ids[output_id]
                candidate_copy = candidate.copy()
                candidate_copy["mmr_score"] = mmr_result.mmr_score
                candidate_copy["mmr_rank"] = mmr_result.rank
                candidate_copy["redundancy_score"] = mmr_result.redundancy_score
                output.append(candidate_copy)

        output.sort(key=lambda x: x.get("mmr_rank", 999))
        logger.info(f"Diversified {len(candidates)} to {len(output)} results")
        return output
