"""Application configuration using Pydantic Settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: str = "postgresql+asyncpg://musicgpt:musicgpt_secret_2024@localhost:5432/musicgpt"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # OpenAI
    openai_api_key: str = ""

    # Embedding model (OpenAI)
    embedding_model: str = (
        "text-embedding-3-small"  # or text-embedding-3-large, text-embedding-ada-002
    )
    embedding_dimension: int = 1536  # text-embedding-3-small default

    # Search configuration
    hnsw_ef_search: int = 100
    candidate_pool_size: int = 500
    composite_pool_size: int = 50
    rerank_pool_size: int = 30
    final_result_size: int = 20

    # Scoring weights (must sum to 1.0)
    weight_semantic: float = 0.50
    weight_popularity: float = 0.25
    weight_exploration: float = 0.15
    weight_freshness: float = 0.10

    # Neural reranking (FlashRank)
    enable_neural_rerank: bool = True
    rerank_blend_weight: float = 0.6  # Weight for neural score vs composite

    # Thompson Sampling priors
    thompson_prior_alpha: float = 1.0
    thompson_prior_beta: float = 1.0

    # Position bias propensities (estimated from click logs)
    position_propensities: dict = {
        1: 1.0,
        2: 0.7,
        3: 0.5,
        4: 0.35,
        5: 0.25,
        6: 0.18,
        7: 0.13,
        8: 0.10,
        9: 0.08,
        10: 0.06,
    }
    default_propensity: float = 0.05

    # Freshness decay (days)
    freshness_decay_rate: float = 0.01

    # MMR diversity
    mmr_lambda: float = 0.7  # Balance relevance vs diversity

    # API
    log_level: str = "INFO"
    cors_origins: list = ["*"]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
