"""FastAPI application entry point.

MusicGPT Self-Optimizing Ranking Engine
- Multi-stage ranking pipeline (Retrieval → Ranking → Reranking → Diversity)
- Thread-safe feedback loop with real-time updates
- Position bias correction and Thompson Sampling for exploration
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import close_db, init_db
from app.routers import feedback_router, search_router

# Configure logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting MusicGPT Ranking Engine...")
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    yield

    # Shutdown
    logger.info("Shutting down...")
    await close_db()


app = FastAPI(
    title="MusicGPT Ranking Engine",
    description="""
## Self-Optimizing Music Search Ranking Engine

A multi-stage ranking pipeline inspired by Spotify, TikTok, YouTube, and Netflix.

### Features
- **Semantic Search**: HNSW-indexed vector search for music descriptions
- **Composite Ranking**: Multi-factor scoring (semantic, popularity, exploration, freshness)
- **Position Bias Correction**: IPW debiasing for unbiased learning from clicks
- **Thompson Sampling**: Bayesian exploration/exploitation for cold-start items
- **Neural Reranking**: FlashRank cross-encoder for improved relevance
- **MMR Diversity**: Maximal Marginal Relevance for diverse results

### Pipeline Stages
1. **Retrieval** (500 candidates): HNSW ANN search with optional filters
2. **Ranking** (50 candidates): Composite scoring with debiased popularity
3. **Reranking** (30 candidates): Cross-encoder neural reranking
4. **Diversity** (20 results): MMR with genre slot allocation

### Feedback Loop
Click feedback is recorded and immediately affects rankings:
- Clicks increment popularity scores
- Position data enables bias correction calibration
- Real-time updates without model retraining
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search_router)
app.include_router(feedback_router)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "MusicGPT Ranking Engine",
        "version": "0.1.0",
        "status": "healthy",
        "endpoints": {
            "search": "POST /search",
            "feedback": "POST /feedback/{output_id}",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database": "connected",
        "ranking_engine": "ready",
    }
