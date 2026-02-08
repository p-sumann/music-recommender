"""Neural reranking using FlashRank (TinyBERT cross-encoder)."""

import asyncio
import logging
from typing import Any, Dict, List

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_reranker = None


def get_reranker():
    """Get or create FlashRank reranker (singleton)."""
    global _reranker
    if _reranker is None:
        try:
            from flashrank import Ranker

            logger.info("Loading FlashRank reranker...")
            _reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
            logger.info("FlashRank loaded successfully")
        except ImportError:
            logger.warning("FlashRank not installed, neural reranking disabled")
        except Exception as e:
            logger.error(f"Failed to load FlashRank: {e}")
    return _reranker


class NeuralReranker:
    """Cross-encoder reranking using FlashRank."""

    def __init__(self, enabled: bool = None):
        self.enabled = enabled if enabled is not None else settings.enable_neural_rerank
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None and self.enabled:
            self._reranker = get_reranker()
        return self._reranker

    def _build_passage_text(self, candidate: Dict[str, Any]) -> str:
        """Build passage text from candidate for reranking."""
        parts = []
        if candidate.get("title"):
            parts.append(candidate["title"])
        if candidate.get("acoustic_prompt_descriptive"):
            parts.append(candidate["acoustic_prompt_descriptive"])
        if candidate.get("sounds_description"):
            parts.append(candidate["sounds_description"])

        metadata = []
        if candidate.get("primary_genre"):
            metadata.append(f"Genre: {candidate['primary_genre']}")
        if candidate.get("primary_mood"):
            metadata.append(f"Mood: {candidate['primary_mood']}")
        if candidate.get("bpm"):
            metadata.append(f"BPM: {candidate['bpm']}")
        if metadata:
            parts.append(". ".join(metadata))

        return ". ".join(parts)

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = None,
        blend_weight: float = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using cross-encoder."""
        top_k = top_k or settings.rerank_pool_size
        blend_weight = blend_weight if blend_weight is not None else settings.rerank_blend_weight

        if not candidates:
            return []

        if not self.enabled or self.reranker is None:
            logger.debug("Neural reranking disabled")
            return candidates[:top_k]

        # Skip for small candidate sets
        if len(candidates) < 10:
            logger.debug(f"Skipping rerank for small set ({len(candidates)} < 10)")
            for candidate in candidates:
                candidate["neural_score"] = None
                candidate["final_score"] = candidate.get("composite_score", 0.0)
            return candidates[:top_k]

        try:
            from flashrank import RerankRequest

            passages = [
                {"id": str(c.get("output_id", i)), "text": self._build_passage_text(c)}
                for i, c in enumerate(candidates)
            ]

            loop = asyncio.get_event_loop()
            request = RerankRequest(query=query, passages=passages)
            results = await loop.run_in_executor(None, self.reranker.rerank, request)

            neural_scores = {r["id"]: r["score"] for r in results}

            for candidate in candidates:
                output_id = str(candidate.get("output_id", ""))
                neural_score = neural_scores.get(output_id, 0.0)
                composite_score = candidate.get("composite_score", 0.0)

                # Normalize neural score to [0, 1]
                normalized_neural = (neural_score + 10) / 20
                normalized_neural = max(0, min(1, normalized_neural))

                candidate["neural_score"] = neural_score
                candidate["normalized_neural_score"] = normalized_neural
                candidate["final_score"] = (
                    blend_weight * normalized_neural + (1 - blend_weight) * composite_score
                )

            candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            logger.info(f"Reranked {len(candidates)} candidates")
            return candidates[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            candidates.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
            return candidates[:top_k]
