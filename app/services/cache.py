"""Redis caching service for query embeddings."""

import hashlib
import logging
from typing import Optional

import numpy as np
import redis.asyncio as redis

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_redis_client: Optional[redis.Redis] = None


async def get_redis_client() -> redis.Redis:
    """Get or create async Redis client (singleton)."""
    global _redis_client
    if _redis_client is None:
        logger.info(f"Connecting to Redis at {settings.redis_url}")
        _redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=False,
        )
        try:
            await _redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, caching disabled")
            _redis_client = None
    return _redis_client


class EmbeddingCache:
    """Cache for query embeddings using Redis."""

    EMBEDDING_TTL = 3600

    def __init__(self):
        self._redis = None
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension

    async def _get_redis(self) -> Optional[redis.Redis]:
        if self._redis is None:
            self._redis = await get_redis_client()
        return self._redis

    def _cache_key(self, query: str) -> str:
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
        return f"emb:{self.model}:{query_hash}"

    async def get(self, query: str) -> Optional[np.ndarray]:
        """Get cached embedding for a query."""
        try:
            redis_client = await self._get_redis()
            if redis_client is None:
                return None

            cache_key = self._cache_key(query)
            cached = await redis_client.get(cache_key)

            if cached is not None:
                embedding = np.frombuffer(cached, dtype=np.float32)
                if len(embedding) == self.dimension:
                    logger.debug(f"Cache HIT: '{query[:30]}...'")
                    return embedding
                await redis_client.delete(cache_key)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        return None

    async def set(self, query: str, embedding: np.ndarray) -> bool:
        """Cache an embedding for a query."""
        try:
            redis_client = await self._get_redis()
            if redis_client is None:
                return False

            cache_key = self._cache_key(query)
            embedding_bytes = embedding.astype(np.float32).tobytes()
            await redis_client.setex(cache_key, self.EMBEDDING_TTL, embedding_bytes)
            logger.debug(f"Cached: '{query[:30]}...'")
            return True
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
            return False

    async def get_or_compute(self, query: str, compute_func) -> np.ndarray:
        """Get from cache or compute if not found."""
        cached = await self.get(query)
        if cached is not None:
            return cached

        logger.debug(f"Cache MISS: '{query[:30]}...'")
        embedding = await compute_func(query)
        await self.set(query, embedding)
        return embedding


def get_embedding_cache() -> EmbeddingCache:
    return EmbeddingCache()
