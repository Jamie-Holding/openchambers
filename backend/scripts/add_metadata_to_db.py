import logging

from backend.src.data.db import init_db, reset_metadata_tables
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
    # Reset metadata tables and run pipeline
    # -----------------------------------------------------
    reset_metadata_tables()
    pipeline.run()

    logger.info("Metadata ingestion pipeline completed.")