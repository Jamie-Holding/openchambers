import argparse
import logging

from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL_NAME
from src.data.db import init_db, reset_db
from src.data.database.utterance import UtteranceRepository
from src.data.pipelines.debate_pipeline import DebatePipeline
from src.data.transformers.chunking_transformer import ChunkingTransformer
from src.data.transformers.embedding_text_formatter import EmbeddingFormatter
from src.data.transformers.statement_summarizer import StatementSummarizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Hansard debates into the database.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop ALL tables and start fresh. Also clears checkpoint. "
             "Re-run metadata ingestion afterwards.",
    )
    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Clear the pipeline checkpoint so ingestion restarts from the beginning.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the statement summariser cache.",
    )
    parser.add_argument(
        "--start-date",
        default="2025-01-01",
        help="Only ingest debates from this date (YYYY-MM-DD). Default: 2025-01-01.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Only ingest debates up to this date (YYYY-MM-DD). Default: no limit.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of XML files to process per batch. Default: 10.",
    )
    args = parser.parse_args()

    logger.info("Starting debate ingestion pipeline...")

    engine = reset_db() if args.reset else init_db()

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    INCLUDE_CONTEXTS = {
        "include_statement": False,
        "include_main_question": True,
        "include_context_question": True
    }
    statement_summarizer = StatementSummarizer(
        model="gpt-4o-mini",
        summarisation_threshold_chars=500,
        cache_path="data/processed/.statement_summaries_cache.json",
        **INCLUDE_CONTEXTS
    )
    embedding_formatter = EmbeddingFormatter(
        model_name=EMBEDDING_MODEL_NAME,
        max_seq_length=embedding_model.max_seq_length,
        **INCLUDE_CONTEXTS
    )
    chunking_transformer = ChunkingTransformer(
        model_name=EMBEDDING_MODEL_NAME,
        max_seq_length=embedding_model.max_seq_length,
        chunk_size=400,
        overlap=100,
    )

    pipeline = DebatePipeline(
        data_dir="data/hansard/debates",
        repository=UtteranceRepository(engine=engine),
        transformers=[statement_summarizer, embedding_formatter, chunking_transformer],
        embedding_model=embedding_model,
        batch_size=args.batch_size,
        start_date=args.start_date,
        end_date=args.end_date
    )

    if args.reset or args.clear_checkpoint:
        pipeline.clear_checkpoint()

    if args.clear_cache:
        statement_summarizer.clear_cache()

    pipeline.run()

    logger.info("Debate ingestion pipeline completed.")
