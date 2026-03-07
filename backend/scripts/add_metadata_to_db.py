import logging

from src.data.database.metadata import MetadataRepository
from src.data.db import init_db
from src.data.pipelines.metadata_pipeline import MetadataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Starting metadata ingestion pipeline...")

    engine = init_db()

    pipeline = MetadataPipeline(
        metadata_dir="data/hansard/metadata",
        repository=MetadataRepository(engine=engine),
    )

    pipeline.run()

    logger.info("Metadata ingestion pipeline completed.")
