"""Database initialisation and management utilities."""

import logging

from sqlalchemy import Engine, create_engine, text

from config.settings import DATABASE_URL
from src.data.database.models import Base

logger = logging.getLogger(__name__)


def init_db() -> Engine:
    """Initialise the database: extensions, tables, and indexes.

    All operations are idempotent — safe to call on every startup.

    Returns:
        SQLAlchemy engine connected to the database.
    """
    engine = create_engine(DATABASE_URL, echo=False, future=True)

    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_textsearch"))
        conn.commit()

    Base.metadata.create_all(engine)

    with engine.connect() as conn:
        conn.execute(
            text("""
                CREATE INDEX IF NOT EXISTS embedding_chunk_id_idx
                ON embedding (chunk_id)
            """)
        )
        conn.execute(
            text("""
                CREATE INDEX IF NOT EXISTS embedding_embedding_hnsw_ip_idx
                ON embedding
                USING hnsw (embedding vector_ip_ops)
                WITH (m = 16, ef_construction = 64)
            """)
        )
        conn.execute(
            text("""
                CREATE INDEX IF NOT EXISTS utterance_chunk_bm25_idx
                ON utterance_chunk
                USING bm25 (chunk_text)
                WITH (text_config = 'english')
            """)
        )
        conn.commit()

    return engine


def reset_db() -> Engine:
    """Drop all tables and recreate from scratch.

    Warning:
        This deletes ALL data. After reset, re-run both ingestion scripts:
        debates first, then metadata (to populate party_at_time).
    """
    engine = create_engine(DATABASE_URL, echo=False, future=True)
    Base.metadata.drop_all(engine)
    logger.info("All tables dropped.")
    return init_db()
