#!/usr/bin/env python3
"""
Benchmark suite for MusicGPT ranking engine.

Runs latency measurements, score breakdowns, feedback tests,
concurrent click tests, and ranking improvement proofs.

Usage:
    python scripts/benchmark.py [--url http://localhost:8000] [--output results/benchmark_results.json] [--runs 3]
"""

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

SEARCH_QUERIES = [
    {"query": "Christmas", "label": "Christmas (holiday)"},
    {"query": "upbeat pop", "label": "Upbeat pop (genre)"},
    {"query": "chill ambient relaxing", "label": "Chill ambient (mood)"},
    {"query": "energetic dance beat", "label": "Energetic dance (tempo)"},
    {"query": "sad piano ballad", "label": "Sad piano ballad (instrument)"},
    {"query": "hip hop rap confident", "label": "Hip hop rap (genre+mood)"},
    {"query": "folk acoustic guitar", "label": "Folk acoustic (genre+instr)"},
    {"query": "electronic synthwave neon", "label": "Synthwave (subgenre)"},
]

FILTER_QUERIES = [
    {"query": "energetic dance", "filters": {"genre": "pop"}, "label": "Genre filter (pop)"},
    {
        "query": "mellow vibes",
        "filters": {"bpm_min": 80, "bpm_max": 100},
        "label": "BPM filter (80-100)",
    },
    {"query": "dark cinematic", "filters": {"format": "MUSIC"}, "label": "Format filter (MUSIC)"},
]


async def search(
    client: httpx.AsyncClient, query: str, filters: dict = None, limit: int = 20
) -> dict:
    """Execute a search request."""
    body = {"query": query, "limit": limit, "include_scores": True}
    if filters:
        body["filters"] = filters
    resp = await client.post(f"{BASE_URL}/search", json=body, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


async def bench_search(
    client: httpx.AsyncClient, query: str, filters: dict = None, runs: int = 3
) -> tuple:
    """Benchmark a search query over multiple runs."""
    latencies = []
    last_data = None
    for _ in range(runs):
        start = time.perf_counter()
        data = await search(client, query, filters)
        latencies.append((time.perf_counter() - start) * 1000)
        last_data = data
    return latencies, last_data


def extract_top_results(data: dict, n: int = 5) -> list:
    """Extract top-N results with scores."""
    results = []
    for r in data["results"][:n]:
        entry = {
            "position": r["position"],
            "title": r["title"],
            "genre": r.get("primary_genre"),
            "mood": r.get("primary_mood"),
            "bpm": r.get("bpm"),
        }
        if r.get("scores"):
            entry["semantic"] = round(r["scores"]["semantic_score"], 4)
            entry["popularity"] = round(r["scores"]["popularity_score"], 4)
            entry["composite"] = round(r["scores"]["composite_score"], 4)
            if r["scores"].get("final_score") is not None:
                entry["final"] = round(r["scores"]["final_score"], 4)
            if r["scores"].get("mmr_score") is not None:
                entry["mmr"] = round(r["scores"]["mmr_score"], 4)
            if r["scores"].get("redundancy_score") is not None:
                entry["redundancy"] = round(r["scores"]["redundancy_score"], 4)
        results.append(entry)
    return results


def latency_stats(latencies: list) -> dict:
    """Compute latency statistics."""
    return {
        "runs": len(latencies),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "avg_ms": round(statistics.mean(latencies), 1),
        "median_ms": round(statistics.median(latencies), 1),
    }


async def bench_feedback(client: httpx.AsyncClient, output_id: str, runs: int = 10) -> dict:
    """Benchmark feedback endpoint latency."""
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        resp = await client.post(
            f"{BASE_URL}/feedback/{output_id}",
            json={"action": "click", "position_shown": 1},
            timeout=10.0,
        )
        latencies.append((time.perf_counter() - start) * 1000)
        resp.raise_for_status()
    return {"output_id": output_id, "latency": latency_stats(latencies)}


async def bench_concurrent_feedback(output_id: str, n: int = 50) -> dict:
    """Benchmark concurrent feedback requests."""
    async with httpx.AsyncClient() as client:
        # Get click count before
        stats_before = await client.get(f"{BASE_URL}/feedback/{output_id}/stats", timeout=10.0)
        clicks_before = stats_before.json()["click_count"] if stats_before.status_code == 200 else 0

        start = time.perf_counter()
        tasks = [
            client.post(
                f"{BASE_URL}/feedback/{output_id}",
                json={"action": "click", "position_shown": 1},
                timeout=30.0,
            )
            for _ in range(n)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall_ms = (time.perf_counter() - start) * 1000

        ok = sum(1 for r in results if isinstance(r, httpx.Response) and r.status_code == 200)

        # Get click count after
        stats_after = await client.get(f"{BASE_URL}/feedback/{output_id}/stats", timeout=10.0)
        clicks_after = stats_after.json()["click_count"] if stats_after.status_code == 200 else 0
        actual_increase = clicks_after - clicks_before

    return {
        "concurrent_requests": n,
        "successful_responses": ok,
        "clicks_before": clicks_before,
        "clicks_after": clicks_after,
        "actual_increase": actual_increase,
        "expected_increase": ok,
        "data_loss": ok - actual_increase,
        "wall_ms": round(wall_ms, 1),
        "avg_per_request_ms": round(wall_ms / n, 1),
        "thread_safe": actual_increase == ok,
    }


async def bench_ranking_improvement(
    client: httpx.AsyncClient, query: str, clicks: int = 50, target_pos: int = 6
) -> dict:
    """Prove that clicks improve ranking position."""
    # Search before
    data_before = await search(client, query)
    if len(data_before["results"]) < target_pos:
        return {"error": f"Not enough results ({len(data_before['results'])} < {target_pos})"}

    target = data_before["results"][target_pos - 1]
    target_id = target["output_id"]
    initial_pos = target["position"]
    scores_before = target.get("scores", {})

    # Record clicks
    for i in range(clicks):
        await client.post(
            f"{BASE_URL}/feedback/{target_id}",
            json={"action": "click", "position_shown": initial_pos},
            timeout=10.0,
        )

    # Get updated stats
    stats_resp = await client.get(f"{BASE_URL}/feedback/{target_id}/stats", timeout=10.0)
    stats = stats_resp.json() if stats_resp.status_code == 200 else {}

    # Search after
    data_after = await search(client, query)
    new_pos = None
    scores_after = {}
    for r in data_after["results"]:
        if r["output_id"] == target_id:
            new_pos = r["position"]
            scores_after = r.get("scores", {})
            break

    positions_gained = (initial_pos - new_pos) if new_pos else None

    return {
        "query": query,
        "target_output_id": target_id,
        "target_title": target["title"],
        "clicks_sent": clicks,
        "clicks_recorded": stats.get("click_count", 0),
        "before": {
            "position": initial_pos,
            "semantic": round(scores_before.get("semantic_score", 0), 4),
            "popularity": round(scores_before.get("popularity_score", 0), 4),
            "composite": round(scores_before.get("composite_score", 0), 4),
        },
        "after": {
            "position": new_pos,
            "semantic": round(scores_after.get("semantic_score", 0), 4),
            "popularity": round(scores_after.get("popularity_score", 0), 4),
            "composite": round(scores_after.get("composite_score", 0), 4),
        },
        "positions_gained": positions_gained,
        "result": "PASS" if positions_gained and positions_gained > 0 else "FAIL",
    }


async def main():
    parser = argparse.ArgumentParser(description="Benchmark MusicGPT ranking engine")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument(
        "--output", default="results/benchmark_results.json", help="Output JSON path"
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per search query")
    parser.add_argument(
        "--clicks", type=int, default=50, help="Clicks for ranking improvement test"
    )
    parser.add_argument("--concurrent", type=int, default=50, help="Concurrent feedback requests")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.url

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "config": {
            "base_url": BASE_URL,
            "runs_per_query": args.runs,
            "clicks": args.clicks,
            "concurrent": args.concurrent,
        },
        "search_benchmarks": [],
        "filter_benchmarks": [],
        "feedback_latency": {},
        "concurrent_feedback": {},
        "ranking_improvement": {},
    }

    async with httpx.AsyncClient() as client:
        # Warmup
        logger.info(f"Benchmarking {BASE_URL}")
        await search(client, "warmup", limit=1)

        # --- Search latency + scores ---
        logger.info("\n=== Search Latency Benchmarks ===")
        for item in SEARCH_QUERIES:
            lats, data = await bench_search(client, item["query"], runs=args.runs)
            stage = {
                "retrieval_ms": round(data.get("retrieval_ms", 0), 1),
                "ranking_ms": round(data.get("ranking_ms", 0), 1),
                "rerank_ms": round(data.get("rerank_ms", 0), 1),
                "diversity_ms": round(data.get("diversity_ms", 0), 1),
            }
            entry = {
                "label": item["label"],
                "query": item["query"],
                "total_candidates": data["total_candidates"],
                "results_returned": len(data["results"]),
                "latency": latency_stats(lats),
                "stages": stage,
                "top_5": extract_top_results(data, 5),
            }
            report["search_benchmarks"].append(entry)
            logger.info(
                f"  {item['label']:30s} | median={entry['latency']['median_ms']}ms | results={entry['results_returned']}"
            )

        # --- Filtered search ---
        logger.info("\n=== Filter Benchmarks ===")
        for item in FILTER_QUERIES:
            lats, data = await bench_search(
                client, item["query"], filters=item.get("filters"), runs=args.runs
            )
            entry = {
                "label": item["label"],
                "query": item["query"],
                "filters": item.get("filters"),
                "results_returned": len(data["results"]),
                "latency": latency_stats(lats),
                "top_3": extract_top_results(data, 3),
            }
            report["filter_benchmarks"].append(entry)
            logger.info(
                f"  {item['label']:30s} | median={entry['latency']['median_ms']}ms | results={entry['results_returned']}"
            )

        # --- Feedback latency ---
        logger.info("\n=== Feedback Latency ===")
        first_result = (await search(client, "pop", limit=5))["results"][0]
        fb = await bench_feedback(client, first_result["output_id"])
        report["feedback_latency"] = fb
        logger.info(
            f"  Single click: median={fb['latency']['median_ms']}ms, min={fb['latency']['min_ms']}ms"
        )

    # --- Concurrent feedback ---
    logger.info("\n=== Concurrent Feedback Test ===")
    conc = await bench_concurrent_feedback(first_result["output_id"], n=args.concurrent)
    report["concurrent_feedback"] = conc
    logger.info(
        f"  {conc['concurrent_requests']} concurrent: {conc['successful_responses']} ok, "
        f"data_loss={conc['data_loss']}, wall={conc['wall_ms']}ms, "
        f"thread_safe={conc['thread_safe']}"
    )

    # --- Ranking improvement ---
    logger.info("\n=== Ranking Improvement Test ===")
    async with httpx.AsyncClient() as client:
        ranking = await bench_ranking_improvement(client, query="Christmas", clicks=args.clicks)
        report["ranking_improvement"] = ranking
        if ranking.get("positions_gained"):
            logger.info(
                f"  {ranking['target_title']}: pos {ranking['before']['position']} -> {ranking['after']['position']} "
                f"(+{ranking['positions_gained']} up) | {ranking['result']}"
            )
        else:
            logger.info(f"  {ranking}")

    # Save
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
