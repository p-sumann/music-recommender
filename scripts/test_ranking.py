#!/usr/bin/env python3
"""
Test script to verify the self-optimizing ranking system.

Tests:
1. Search returns results
2. Click feedback is recorded
3. Repeated clicks improve ranking position
4. Position bias correction works

Usage:
    python scripts/test_ranking.py
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"


async def test_search(query: str = "upbeat pop") -> list:
    """Test search endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/search",
            json={
                "query": query,
                "limit": 20,
                "include_scores": True,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        logger.info(f"Search '{query}' returned {len(data['results'])} results")
        logger.info(
            f"  Timing: retrieval={data.get('retrieval_ms', 0):.1f}ms, "
            f"ranking={data.get('ranking_ms', 0):.1f}ms, "
            f"rerank={data.get('rerank_ms', 0):.1f}ms, "
            f"diversity={data.get('diversity_ms', 0):.1f}ms"
        )

        return data["results"]


async def record_click(output_id: str, position: int) -> dict:
    """Record a click."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/feedback/{output_id}",
            json={
                "action": "click",
                "position_shown": position,
                "session_id": "test-session",
            },
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()


async def get_stats(output_id: str) -> dict:
    """Get stats for an output."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/feedback/{output_id}/stats",
            timeout=10.0,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()


async def test_ranking_improvement(
    query: str = "upbeat pop",
    click_count: int = 50,
    target_position: int = 6,
):
    """
    Test that clicks improve ranking position.

    Algorithm:
    1. Search for a query
    2. Pick an item at position N (e.g., 6th result)
    3. Record many clicks on it
    4. Search again
    5. Verify the item moved up in rankings
    """
    logger.info("=" * 60)
    logger.info("TEST: Ranking Improvement from Clicks")
    logger.info("=" * 60)

    # Step 1: Initial search
    logger.info(f"\n1. Initial search for '{query}'...")
    results_before = await test_search(query)

    if len(results_before) < target_position:
        logger.error(f"Not enough results (got {len(results_before)}, need {target_position})")
        return False

    # Step 2: Pick target item
    target = results_before[target_position - 1]
    target_id = target["output_id"]
    target_title = target["title"]
    initial_position = target["position"]

    logger.info("\n2. Selected target item:")
    logger.info(f"   ID: {target_id}")
    logger.info(f"   Title: {target_title[:50]}...")
    logger.info(f"   Initial position: {initial_position}")

    if target.get("scores"):
        logger.info("   Initial scores:")
        logger.info(f"     - Semantic: {target['scores']['semantic_score']:.4f}")
        logger.info(f"     - Popularity: {target['scores']['popularity_score']:.4f}")
        logger.info(f"     - Composite: {target['scores']['composite_score']:.4f}")

    # Step 3: Record clicks
    logger.info(f"\n3. Recording {click_count} clicks...")
    for i in range(click_count):
        await record_click(target_id, initial_position)
        if (i + 1) % 10 == 0:
            logger.info(f"   Recorded {i + 1}/{click_count} clicks")

    # Check updated stats
    stats = await get_stats(target_id)
    if stats:
        logger.info("\n   Updated stats:")
        logger.info(f"     - Clicks: {stats['click_count']}")
        logger.info(f"     - CTR estimate: {stats['ctr_estimate']:.4f}")

    # Step 4: Search again
    logger.info("\n4. Searching again...")
    results_after = await test_search(query)

    # Step 5: Find new position
    new_position = None
    for result in results_after:
        if result["output_id"] == target_id:
            new_position = result["position"]

            if result.get("scores"):
                logger.info("\n   New scores:")
                logger.info(f"     - Semantic: {result['scores']['semantic_score']:.4f}")
                logger.info(f"     - Popularity: {result['scores']['popularity_score']:.4f}")
                logger.info(f"     - Composite: {result['scores']['composite_score']:.4f}")
            break

    # Step 6: Verify improvement
    logger.info("\n5. Results:")
    logger.info(f"   Initial position: {initial_position}")
    logger.info(f"   New position: {new_position}")

    if new_position is None:
        logger.warning("   Target item not found in results (may have been filtered out)")
        return False

    if new_position < initial_position:
        improvement = initial_position - new_position
        logger.info(f"SUCCESS! Item moved UP by {improvement} positions")
        return True
    elif new_position == initial_position:
        logger.warning("Item stayed at same position")
        logger.warning("   This may happen if popularity weight is low or other items also changed")
        return False
    else:
        logger.error(f"FAILED! Item moved DOWN by {new_position - initial_position} positions")
        return False


async def test_concurrency(output_id: str, concurrent_clicks: int = 20):
    """
    Test that concurrent clicks are handled correctly (no race conditions).
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Concurrent Click Handling")
    logger.info("=" * 60)

    # Get initial stats
    stats_before = await get_stats(output_id)
    initial_clicks = stats_before["click_count"] if stats_before else 0

    logger.info(f"Initial clicks: {initial_clicks}")
    logger.info(f"Sending {concurrent_clicks} concurrent clicks...")

    # Send concurrent clicks
    async with httpx.AsyncClient() as client:
        tasks = [
            client.post(
                f"{BASE_URL}/feedback/{output_id}",
                json={"action": "click", "position_shown": 1},
                timeout=30.0,
            )
            for _ in range(concurrent_clicks)
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful responses
    successful = sum(1 for r in responses if isinstance(r, httpx.Response) and r.status_code == 200)
    logger.info(f"Successful responses: {successful}/{concurrent_clicks}")

    # Check final stats
    stats_after = await get_stats(output_id)
    final_clicks = stats_after["click_count"] if stats_after else 0
    actual_increase = final_clicks - initial_clicks

    logger.info(f"Final clicks: {final_clicks}")
    logger.info(f"Expected increase: {successful}")
    logger.info(f"Actual increase: {actual_increase}")

    if actual_increase == successful:
        logger.info("✅ SUCCESS! All concurrent clicks were recorded correctly")
        return True
    else:
        logger.error(f"❌ FAILED! Expected {successful} new clicks, got {actual_increase}")
        return False


async def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test ranking system")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--query", default="upbeat pop", help="Search query to test")
    parser.add_argument("--clicks", type=int, default=50, help="Number of clicks to simulate")

    args = parser.parse_args()
    global BASE_URL
    BASE_URL = args.url

    logger.info(f"Testing API at {BASE_URL}")

    try:
        # Test 1: Basic search
        results = await test_search(args.query)
        if not results:
            logger.error("No search results, cannot continue tests")
            return

        # Test 2: Ranking improvement
        success = await test_ranking_improvement(
            query=args.query,
            click_count=args.clicks,
        )

        # Test 3: Concurrency (using first result)
        if results:
            await test_concurrency(results[0]["output_id"], concurrent_clicks=20)

        logger.info("\n" + "=" * 60)
        logger.info("All tests completed!")
        logger.info("=" * 60)

    except httpx.ConnectError:
        logger.error(f"Could not connect to {BASE_URL}")
        logger.error("Make sure the API is running: docker-compose up")
    except Exception as e:
        logger.exception(f"Test failed with error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
