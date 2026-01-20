import logging

from sentence_transformers import SentenceTransformer

from backend.config.settings import EMBEDDING_MODEL_NAME
from backend.src.data.db import init_db, reset_db
from backend.src.data.database.utterance import UtteranceRepository
from backend.src.data.pipelines.debate_pipeline import DebatePipeline
from backend.src.data.transformers.chunking_transformer import ChunkingTransformer
from backend.src.data.transformers.embedding_text_formatter import EmbeddingFormatter
from backend.src.data.transformers.statement_summarizer import StatementSummarizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Starting debate ingestion pipeline...")

    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------
    RESET_DATABASE = True  # Set True to drop all tables and start fresh

    # -----------------------------------------------------
    # Initialize the database engine
    # -----------------------------------------------------
    engine = init_db()

    # -----------------------------------------------------
    # Initialize repository
    # -----------------------------------------------------
    utterance_repo = UtteranceRepository(engine=engine)

    # -----------------------------------------------------
    # Initialize embedding model
    # -----------------------------------------------------
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
 
    # -----------------------------------------------------
    # Initialize transformers
    # -----------------------------------------------------
    INCLUDE_CONTEXTS = {
        "include_statement": False,
        "include_main_question": True,
        "include_context_question": True
    }
    statement_summarizer = StatementSummarizer(
        model="gpt-4o-mini",
        token_threshold=100,
        cache_path="backend/data/processed/.statement_summaries_cache.json",
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

    transformers = [
        statement_summarizer,
        embedding_formatter,
        chunking_transformer,
    ]

    # -----------------------------------------------------
    # Initialize pipeline
    # -----------------------------------------------------
    pipeline = DebatePipeline(
        data_dir="backend/data/hansard/debates",
        repository=utterance_repo,
        transformers=transformers,
        embedding_model=embedding_model,
        batch_size=2,
        max_batches=2,  # Process all batches
        start_date="2024-01-01"
    )

    # -----------------------------------------------------
    # Reset DB if needed (also clears checkpoint and summary cache)
    # -----------------------------------------------------
    if RESET_DATABASE:
        reset_db(drop=True)
        pipeline.clear_checkpoint()
        #statement_summarizer.clear_cache()


    # -----------------------------------------------------
    # Run pipeline (automatically resumes from checkpoint)
    # -----------------------------------------------------
    pipeline.run()

    logger.info("Debate ingestion pipeline completed.")
