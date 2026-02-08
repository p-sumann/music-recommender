#!/usr/bin/env python3
"""
Diversity proof test for MusicGPT ranking engine.

Demonstrates that MMR (Maximal Marginal Relevance) prevents genre domination
in search results. Runs queries and proves genre spread in top-10.

Usage:
    python scripts/test_diversity.py [--url http://localhost:8000] [--output results/diversity_proof.json]
"""

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

DIVERSITY_QUERIES = [
    {"query": "Pop", "description": "Generic genre query - should NOT return all-pop results"},
    {"query": "Christmas holiday festive", "description": "Seasonal query - should NOT be dominated by one subgenre"},
    {"query": "energetic dance beat", "description": "Mood+tempo query - should surface multiple genres"},
    {"query": "chill ambient relaxing", "description": "Mood query - should mix electronic, folk, world"},
    {"query": "upbeat pop", "description": "Specific genre - MMR should still inject diversity"},
]


async def run_diversity_test(client: httpx.AsyncClient, query: str, top_k: int = 10) -> dict:
    """Run a single diversity test for a query."""
    resp = await client.post(
        f"{BASE_URL}/search",
        json={"query": query, "limit": top_k, "include_scores": True},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for r in data["results"][:top_k]:
        scores = r.get("scores", {})
        results.append({
            "position": r["position"],
            "title": r["title"],
            "genre": r.get("primary_genre"),
            "mood": r.get("primary_mood"),
            "semantic_score": round(scores.get("semantic_score", 0), 4),
            "composite_score": round(scores.get("composite_score", 0), 4),
            "mmr_score": round(scores.get("mmr_score", 0), 4) if scores.get("mmr_score") is not None else None,
            "redundancy_score": round(scores.get("redundancy_score", 0), 4) if scores.get("redundancy_score") is not None else None,
        })

    genres = [r["genre"] for r in results if r["genre"]]
    genre_dist = dict(Counter(genres))
    unique_genres = len(genre_dist)

    moods = [r["mood"] for r in results if r["mood"]]
    mood_dist = dict(Counter(moods))
    unique_moods = len(mood_dist)

    # Check if any single genre dominates (>60% of results)
    max_genre_pct = max(genre_dist.values()) / len(genres) * 100 if genres else 0
    dominated = max_genre_pct > 60

    return {
        "query": query,
        "total_candidates": data.get("total_candidates", 0),
        "results_shown": len(results),
        "results": results,
        "genre_distribution": genre_dist,
        "unique_genres": unique_genres,
        "mood_distribution": mood_dist,
        "unique_moods": unique_moods,
        "max_genre_pct": round(max_genre_pct, 1),
        "genre_dominated": dominated,
        "diversity_pass": unique_genres >= 3 and not dominated,
    }


async def main():
    parser = argparse.ArgumentParser(description="Diversity proof test")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", default="results/diversity_proof.json", help="Output JSON path")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.url

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running diversity tests against {BASE_URL}")
    logger.info("=" * 70)

    report = {"tests": [], "summary": {}}
    pass_count = 0

    async with httpx.AsyncClient() as client:
        # Warmup
        await client.post(f"{BASE_URL}/search", json={"query": "warmup", "limit": 1}, timeout=30.0)

        for item in DIVERSITY_QUERIES:
            logger.info(f"\nQuery: \"{item['query']}\"")
            logger.info(f"  Expectation: {item['description']}")

            result = await run_diversity_test(client, item["query"])
            result["description"] = item["description"]
            report["tests"].append(result)

            # Print results
            logger.info(f"  Results: {result['results_shown']} | Candidates: {result['total_candidates']}")
            logger.info(f"  Genre distribution: {result['genre_distribution']}")
            logger.info(f"  Unique genres: {result['unique_genres']} | Max genre %: {result['max_genre_pct']}%")

            for r in result["results"][:5]:
                redund = f"{r['redundancy_score']:.4f}" if r["redundancy_score"] is not None else "N/A"
                mmr = f"{r['mmr_score']:.4f}" if r["mmr_score"] is not None else "N/A"
                logger.info(
                    f"    #{r['position']:2d} | {str(r['genre']):20s} | {str(r['mood']):12s} "
                    f"| sem={r['semantic_score']:.4f} | mmr={mmr} | redund={redund} | {r['title']}"
                )

            if result["diversity_pass"]:
                logger.info(f"  PASS: {result['unique_genres']} genres, no domination")
                pass_count += 1
            else:
                logger.warning(f"  FAIL: dominated={result['genre_dominated']}, genres={result['unique_genres']}")

    report["summary"] = {
        "total_tests": len(DIVERSITY_QUERIES),
        "passed": pass_count,
        "failed": len(DIVERSITY_QUERIES) - pass_count,
        "all_passed": pass_count == len(DIVERSITY_QUERIES),
    }

    # Save JSON
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    logger.info("\n" + "=" * 70)
    logger.info(f"Diversity Tests: {pass_count}/{len(DIVERSITY_QUERIES)} passed")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
