"""Database initialisation and management utilities."""

import logging

from sqlalchemy import Engine, create_engine, text

from backend.config.settings import DATABASE_URL
from backend.src.data.database.models import (
    Base,
    Division,
    Membership,
    MPPolicySummary,
    Person,
    Vote,
)

logger = logging.getLogger(__name__)


def init_db() -> Engine:
    """Initialise the database engine and ensure pgvector extension exists.

    Returns:
        SQLAlchemy engine connected to the database.
    """
    engine = create_engine(DATABASE_URL, echo=False, future=True)

    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    return engine


def reset_db(drop: bool = True) -> None:
    """Drop and recreate all tables based on current SQLAlchemy models.

    Args:
        drop: If True, drop existing tables before recreating.

    Warning:
        This will delete ALL data in the database when drop=True.
    """
    engine = init_db()

    if drop:
        # Drop all tables
        Base.metadata.drop_all(engine)
        logger.info("All tables dropped.")

    # Recreate all tables
    Base.metadata.create_all(engine)
    logger.info("All tables recreated.")

    # Create indexes for performance
    with engine.connect() as conn:
        # B-tree index on chunk_id for faster joins between embedding and utterance_chunk
        conn.execute(
            text("""
                CREATE INDEX IF NOT EXISTS embedding_chunk_id_idx
                ON embedding (chunk_id)
            """)
        )

        # HNSW index for approximate nearest neighbour search using inner product
        conn.execute(
            text("""
                CREATE INDEX IF NOT EXISTS embedding_embedding_hnsw_ip_idx
                ON embedding
                USING hnsw (embedding vector_ip_ops)
                WITH (m = 16, ef_construction = 64)
            """)
        )

        conn.commit()
        logger.info("Indexes created.")


def reset_metadata_tables() -> None:
    """Drop and recreate metadata tables only (person, membership, division, vote, mp_policy_summary).

    This preserves other tables like utterances and embeddings while allowing
    schema changes to metadata tables.
    """
    engine = init_db()
    metadata_tables = [MPPolicySummary, Vote, Division, Membership, Person]

    # Drop tables in order (respecting foreign key constraints)
    for table in metadata_tables:
        table.__table__.drop(engine, checkfirst=True)
    logger.info("Metadata tables dropped.")

    # Recreate tables
    for table in reversed(metadata_tables):
        table.__table__.create(engine, checkfirst=True)
    logger.info("Metadata tables recreated.")

