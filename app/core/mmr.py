"""Maximal Marginal Relevance (MMR) for result diversity."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class MMRCandidate:
    id: str
    relevance_score: float
    embedding: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class MMRResult:
    id: str
    relevance_score: float
    mmr_score: float
    redundancy_score: float
    rank: int


class MMRDiversifier:
    """MMR diversifier with optional genre slot allocation."""

    def __init__(self, lambda_relevance: float = 0.7, similarity_fn: Optional[Callable] = None):
        self.lambda_relevance = lambda_relevance
        self.similarity_fn = similarity_fn or self._cosine_similarity

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def compute_redundancy(self, candidate: MMRCandidate, selected: List[MMRCandidate]) -> float:
        """Max similarity to already-selected items."""
        if not selected:
            return 0.0
        return max(self.similarity_fn(candidate.embedding, s.embedding) for s in selected)

    def compute_mmr_score(self, relevance: float, redundancy: float) -> float:
        """MMR = λ * relevance - (1-λ) * redundancy"""
        return self.lambda_relevance * relevance - (1 - self.lambda_relevance) * redundancy

    def diversify(
        self,
        candidates: List[MMRCandidate],
        k: int = 20,
        genre_slots: Optional[Dict[str, int]] = None,
    ) -> List[MMRResult]:
        """Select top-k diverse items using MMR."""
        if not candidates:
            return []

        selected: List[MMRCandidate] = []
        results: List[MMRResult] = []
        remaining = candidates.copy()
        genre_counts: Dict[str, int] = {g: 0 for g in (genre_slots or {})}

        while len(selected) < k and remaining:
            best_score = float("-inf")
            best_idx = 0
            best_redundancy = 0.0

            for i, candidate in enumerate(remaining):
                if genre_slots:
                    genre = candidate.metadata.get("primary_genre", "other")
                    if genre in genre_counts and genre_counts[genre] >= genre_slots[genre]:
                        continue

                redundancy = self.compute_redundancy(candidate, selected)
                mmr_score = self.compute_mmr_score(candidate.relevance_score, redundancy)

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
                    best_redundancy = redundancy

            if best_score == float("-inf"):
                break

            chosen = remaining.pop(best_idx)
            selected.append(chosen)

            if genre_slots:
                genre = chosen.metadata.get("primary_genre", "other")
                if genre in genre_counts:
                    genre_counts[genre] += 1

            results.append(
                MMRResult(
                    id=chosen.id,
                    relevance_score=chosen.relevance_score,
                    mmr_score=best_score,
                    redundancy_score=best_redundancy,
                    rank=len(results) + 1,
                )
            )

        return results


def allocate_genre_slots(
    candidates: List[Dict[str, Any]],
    total_slots: int = 20,
    min_per_genre: int = 2,
    genre_key: str = "primary_genre",
) -> Dict[str, int]:
    """Allocate result slots by genre distribution."""
    from collections import Counter

    genre_counts = Counter(item.get(genre_key, "other") for item in candidates)
    total_items = len(candidates)
    slots: Dict[str, int] = {}
    remaining_slots = total_slots

    for genre in genre_counts:
        allocated = min(min_per_genre, remaining_slots)
        slots[genre] = allocated
        remaining_slots -= allocated
        if remaining_slots <= 0:
            break

    if remaining_slots > 0:
        for genre, count in genre_counts.most_common():
            proportion = count / total_items
            bonus = int(remaining_slots * proportion)
            slots[genre] = slots.get(genre, 0) + bonus
            remaining_slots -= bonus
            if remaining_slots <= 0:
                break

    return slots
