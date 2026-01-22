import logging

from backend.src.data.db import init_db, reset_db
from backend.src.data.database.metadata import MetadataRepository
from backend.src.data.pipelines.metadata_pipeline import MetadataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Starting metadata ingestion pipeline...")

    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------
    RESET_DATABASE = False  # Set True to drop all tables and start fresh

    # -----------------------------------------------------
    # Initialize the database engine
    # -----------------------------------------------------
    engine = init_db()

    # -----------------------------------------------------
    # Initialize repository
    # -----------------------------------------------------
    metadata_repo = MetadataRepository(engine=engine)

    # -----------------------------------------------------
    # Initialize pipeline
    # -----------------------------------------------------
    pipeline = MetadataPipeline(
        metadata_dir="backend/data/hansard/metadata",
        repository=metadata_repo
    )

    # -----------------------------------------------------
    # Reset DB if needed
    # -----------------------------------------------------
    if RESET_DATABASE:
        reset_db(drop=True)

    # -----------------------------------------------------
    # Run pipeline
    # -----------------------------------------------------
    pipeline.run()

    logger.info("Metadata ingestion pipeline completed.")