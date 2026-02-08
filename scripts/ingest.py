#!/usr/bin/env python3
"""
Data ingestion script for MusicGPT ranking engine.

Reads song_metadata.json (DynamoDB export format), generates embeddings,
and inserts normalized data into PostgreSQL.

Usage:
    python scripts/ingest.py [--file path/to/song_metadata.json] [--batch-size 32]
"""

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from app.config import get_settings
from app.database import async_session_maker, engine
from app.models import AudioOutput, ItemStatistics, Song
from app.services.embedding import EmbeddingService
from app.utils.dynamo_parser import parse_song_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


async def create_tables():
    """Create database tables if they don't exist."""
    async with engine.begin() as conn:
        # Create pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Import and create all tables
        from app.database import Base

        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")


async def ingest_songs(
    file_path: str,
    batch_size: int = 32,
    skip_existing: bool = True,
):
    """
    Ingest songs from JSON file into database.

    Args:
        file_path: Path to song_metadata.json
        batch_size: Batch size for embedding generation
        skip_existing: Skip songs that already exist in DB
    """
    # Load JSON data
    logger.info(f"Loading data from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    logger.info(f"Loaded {len(raw_data)} items from JSON")

    # Parse DynamoDB format
    logger.info("Parsing DynamoDB format...")
    parsed_songs = []
    for raw_item in raw_data:
        try:
            parsed = parse_song_metadata(raw_item)
            if parsed.get("id") and parsed.get("title"):
                parsed_songs.append(parsed)
        except Exception as e:
            logger.warning(f"Failed to parse item: {e}")
            continue

    logger.info(f"Successfully parsed {len(parsed_songs)} songs")

    # Initialize embedding service
    logger.info("Initializing embedding service...")
    embedding_service = EmbeddingService()

    # Generate embeddings in batches
    logger.info("Generating embeddings...")
    texts_to_embed = [
        song.get("acoustic_prompt_descriptive") or song.get("title", "") for song in parsed_songs
    ]

    embeddings = embedding_service.embed_batch(
        texts_to_embed,
        batch_size=batch_size,
        show_progress=True,
    )

    # Insert into database
    logger.info("Inserting into database...")

    async with async_session_maker() as session:
        inserted_songs = 0
        inserted_outputs = 0
        skipped = 0

        for i, (song_data, embedding) in enumerate(zip(parsed_songs, embeddings)):
            try:
                song_id = uuid.UUID(song_data["id"]) if song_data.get("id") else uuid.uuid4()

                # Check if song exists
                if skip_existing:
                    existing = await session.execute(
                        text("SELECT 1 FROM songs WHERE id = :id"),
                        {"id": str(song_id)},
                    )
                    if existing.scalar():
                        skipped += 1
                        continue

                # Create song
                song = Song(
                    id=song_id,
                    title=song_data["title"],
                    prompt=song_data.get("prompt"),
                    lyrics=song_data.get("lyrics"),
                    acoustic_prompt_descriptive=song_data.get("acoustic_prompt_descriptive"),
                    embedding=embedding.tolist() if embedding is not None else None,
                    bpm=song_data.get("bpm"),
                    musical_key=song_data.get("musical_key"),
                    primary_genre=song_data.get("primary_genre"),
                    primary_mood=song_data.get("primary_mood"),
                    format=song_data.get("format"),
                    primary_context=song_data.get("primary_context"),
                    vocal_gender=song_data.get("vocal_gender"),
                    extended_metadata=song_data.get("extended_metadata", {}),
                )
                session.add(song)
                inserted_songs += 1

                # Create audio outputs
                for output_data in song_data.get("outputs", []):
                    if not output_data.get("audio_url"):
                        continue

                    output = AudioOutput(
                        id=uuid.uuid4(),
                        song_id=song_id,
                        output_number=output_data.get("output_number", 1),
                        audio_url=output_data["audio_url"],
                        sounds_description=output_data.get("sounds_description"),
                    )
                    session.add(output)
                    inserted_outputs += 1

                    # Initialize statistics
                    stats = ItemStatistics(
                        output_id=output.id,
                        impression_count=0,
                        click_count=0,
                        like_count=0,
                        position_sum=0,
                        ctr_estimate=0.5,
                        ctr_variance=0.25,
                    )
                    session.add(stats)

                # Commit in batches
                if (i + 1) % 100 == 0:
                    await session.commit()
                    logger.info(f"Progress: {i + 1}/{len(parsed_songs)} songs processed")

            except Exception as e:
                logger.error(f"Failed to insert song {song_data.get('id')}: {e}")
                await session.rollback()
                continue

        # Final commit
        await session.commit()

        logger.info("Ingestion complete!")
        logger.info(f"  Songs inserted: {inserted_songs}")
        logger.info(f"  Outputs inserted: {inserted_outputs}")
        logger.info(f"  Songs skipped (existing): {skipped}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ingest song data into ranking engine")
    parser.add_argument(
        "--file",
        type=str,
        default="song_metadata.json",
        help="Path to song_metadata.json file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--recreate-tables",
        action="store_true",
        help="Recreate database tables (WARNING: drops existing data)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Don't skip existing songs (may cause duplicates)",
    )

    args = parser.parse_args()

    # Resolve file path
    file_path = Path(args.file)
    if not file_path.is_absolute():
        # Try relative to script, then project root
        script_dir = Path(__file__).parent
        project_root = script_dir.parent

        if (project_root / file_path).exists():
            file_path = project_root / file_path
        elif (script_dir / file_path).exists():
            file_path = script_dir / file_path

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)

    # Create tables
    logger.info("Setting up database...")
    await create_tables()

    # Run ingestion
    await ingest_songs(
        file_path=str(file_path),
        batch_size=args.batch_size,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    asyncio.run(main())
