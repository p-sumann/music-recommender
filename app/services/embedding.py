"""OpenAI embedding service."""

import logging
from typing import List, Optional

import numpy as np
from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_async_client: Optional[AsyncOpenAI] = None
_sync_client: Optional[OpenAI] = None


def get_async_openai_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        logger.info(f"Initializing AsyncOpenAI: {settings.embedding_model}")
        _async_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _async_client


def get_sync_openai_client() -> OpenAI:
    global _sync_client
    if _sync_client is None:
        logger.info(f"Initializing sync OpenAI: {settings.embedding_model}")
        _sync_client = OpenAI(api_key=settings.openai_api_key)
    return _sync_client


class EmbeddingService:
    """OpenAI embedding service (async for API, sync for batch ingestion)."""

    def __init__(self):
        self._async_client = None
        self._sync_client = None
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension

    @property
    def async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = get_async_openai_client()
        return self._async_client

    @property
    def sync_client(self) -> OpenAI:
        if self._sync_client is None:
            self._sync_client = get_sync_openai_client()
        return self._sync_client

    async def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for search query (async)."""
        if not query or not query.strip():
            return np.zeros(self.dimension)
        return await self._embed_with_retry(query)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True
    )
    async def _embed_with_retry(self, text: str) -> np.ndarray:
        response = await self.async_client.embeddings.create(
            model=self.model, input=text, dimensions=self.dimension
        )
        return np.array(response.data[0].embedding)

    def embed_batch(
        self, texts: List[str], batch_size: int = 100, show_progress: bool = True
    ) -> List[Optional[np.ndarray]]:
        """Generate embeddings for batch (sync, for ingestion)."""
        valid_indices, valid_texts = [], []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)

        if not valid_texts:
            return [None] * len(texts)

        all_embeddings = []
        total_batches = (len(valid_texts) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(valid_texts), batch_size):
            batch = valid_texts[batch_idx : batch_idx + batch_size]
            if show_progress:
                logger.info(
                    f"Batch {batch_idx // batch_size + 1}/{total_batches} ({len(batch)} texts)"
                )

            try:
                response = self.sync_client.embeddings.create(
                    model=self.model, input=batch, dimensions=self.dimension
                )
                all_embeddings.extend([np.array(item.embedding) for item in response.data])
            except Exception as e:
                logger.error(f"Batch failed: {e}")
                all_embeddings.extend([None] * len(batch))

        result = [None] * len(texts)
        for idx, embedding in zip(valid_indices, all_embeddings):
            result[idx] = embedding
        return result

    def compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between embeddings."""
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
