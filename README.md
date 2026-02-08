# MusicGPT Ranking Engine

A self-optimizing music search ranking engine with multi-stage retrieval, neural reranking, and real-time feedback learning. Inspired by ranking systems at Spotify, TikTok, YouTube, and Netflix.

## Features

- **Semantic Search** - HNSW-indexed vector search using OpenAI embeddings
- **Multi-Stage Ranking** - Retrieval → Composite Scoring → Neural Reranking → Diversity
- **Self-Optimizing** - Click feedback immediately improves rankings
- **Position Bias Correction** - IPW debiasing for unbiased learning
- **Thompson Sampling** - Bayesian exploration for cold-start items
- **Neural Reranking** - FlashRank cross-encoder for relevance refinement
- **MMR Diversity** - Maximal Marginal Relevance with genre slot allocation

## Architecture

```
Query "upbeat pop"
       │
       ▼
┌──────────────────┐
│  Stage 1: HNSW   │  500 candidates
│  Vector Search   │  ~30ms (cached) / ~500ms (uncached)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Stage 2: Rank   │  50 candidates
│  Composite Score │  ~1ms
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Stage 3: Rerank │  30 candidates
│  FlashRank       │  ~40ms
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Stage 4: MMR    │  20 results
│  Diversity       │  ~10ms
└────────┬─────────┘
         ▼
    Final Results
```

## Scoring Formula

```
composite = 0.50 × semantic + 0.25 × popularity + 0.15 × exploration + 0.10 × freshness
final = 0.60 × neural_rerank + 0.40 × composite
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI API key

### Setup

```bash
# Clone and enter directory
cd ranking-system

# Set your OpenAI API key
export OPENAI_API_KEY=sk-your-key-here

# Start services
docker-compose up -d

# Run migrations
docker-compose exec api alembic upgrade head

# Ingest sample data
docker-compose exec api python scripts/ingest.py --file /app/song_metadata.json
```

### Test

```bash
# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "upbeat pop music", "limit": 5}'

# Record feedback
curl -X POST http://localhost:8000/feedback/{output_id} \
  -H "Content-Type: application/json" \
  -d '{"action": "click", "position_shown": 1}'

# Run test suite
docker-compose exec api python scripts/test_ranking.py
```

## API Endpoints

### POST /search

Search for music with multi-stage ranking.

**Request:**
```json
{
  "query": "upbeat pop music",
  "limit": 20,
  "include_scores": true,
  "filters": {
    "genre": "pop",
    "mood": "happy",
    "bpm_min": 100,
    "bpm_max": 140
  }
}
```

**Response:**
```json
{
  "query": "upbeat pop music",
  "results": [...],
  "total_candidates": 96,
  "retrieval_ms": 35.2,
  "ranking_ms": 0.5,
  "rerank_ms": 42.1,
  "diversity_ms": 8.3,
  "total_ms": 86.1
}
```

### POST /feedback/{output_id}

Record user interaction feedback.

**Request:**
```json
{
  "action": "click",
  "position_shown": 3
}
```

### GET /feedback/{output_id}/stats

Get engagement statistics for an item.

## Configuration

Environment variables (`.env`):

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/musicgpt

# Redis (embedding cache)
REDIS_URL=redis://localhost:6379/0

# OpenAI
OPENAI_API_KEY=sk-your-key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# Scoring weights
WEIGHT_SEMANTIC=0.50
WEIGHT_POPULARITY=0.25
WEIGHT_EXPLORATION=0.15
WEIGHT_FRESHNESS=0.10

# Neural reranking
ENABLE_NEURAL_RERANK=true
RERANK_BLEND_WEIGHT=0.6

# MMR diversity
MMR_LAMBDA=0.7
```

## Project Structure

```
ranking-system/
├── app/
│   ├── core/                 # Core algorithms
│   │   ├── mmr.py           # Maximal Marginal Relevance
│   │   ├── position_bias.py # IPW debiasing
│   │   └── thompson_sampling.py
│   ├── models/              # SQLAlchemy models
│   ├── routers/             # FastAPI routes
│   ├── schemas/             # Pydantic schemas
│   ├── services/            # Business logic
│   │   ├── cache.py         # Redis embedding cache
│   │   ├── embedding.py     # OpenAI embeddings
│   │   ├── retrieval.py     # HNSW search
│   │   ├── ranking.py       # Composite scoring
│   │   ├── neural_reranker.py # FlashRank
│   │   └── diversity.py     # MMR diversification
│   ├── config.py
│   ├── database.py
│   └── main.py
├── alembic/                  # Database migrations
├── scripts/
│   ├── ingest.py            # Data ingestion
│   └── test_ranking.py      # Test suite
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

## Performance

| Scenario | Latency |
|----------|---------|
| Cache HIT (repeated query) | ~80ms |
| Cache MISS (new query) | ~550ms |
| Retrieval (cached) | ~30ms |
| Neural rerank | ~40ms |
| MMR diversity | ~10ms |

## Tech Stack

- **FastAPI** - Async web framework
- **PostgreSQL + pgvector** - Vector database with HNSW index
- **Redis** - Embedding cache
- **SQLAlchemy** - Async ORM
- **OpenAI** - Text embeddings (text-embedding-3-small)
- **FlashRank** - Neural cross-encoder reranking
- **Docker** - Containerization

## License

MIT
