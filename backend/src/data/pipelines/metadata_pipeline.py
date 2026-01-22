"""Pipeline for loading Hansard metadata into the database."""

import logging

from backend.src.data.database.metadata import MetadataRepository
from backend.src.data.loaders.theyworkforyou.metadata import Metadata

logger = logging.getLogger(__name__)


class MetadataPipeline:
    """Pipeline to load Hansard metadata into the database."""

    def __init__(self, metadata_dir: str, repository: MetadataRepository) -> None:
        """Initialise the metadata pipeline.

        Args:
            metadata_dir: Directory containing TheyWorkForYou metadata files.
            repository: Repository for database interactions.
        """
        self.loader = Metadata(metadata_dir=metadata_dir)
        self.repository = repository

    def run(self) -> None:
        """Execute the metadata pipeline.

        Loads people, memberships, divisions, and votes from files, truncates
        existing tables, inserts new data, and updates party_at_time in utterances.
        """
        # Load metadata
        logger.info("Loading metadata files...")
        people_df = self.loader.load_people()
        memberships_df = self.loader.load_memberships()
        divisions_df = self.loader.load_divisions()
        votes_df = self.loader.load_votes()

        # Insert into database
        logger.info("Truncating metadata tables...")
        self.repository.truncate_tables()

        logger.info(f"Inserting {len(people_df)} people...")
        self.repository.insert_people(people_df)

        logger.info(f"Inserting {len(memberships_df)} memberships...")
        self.repository.insert_memberships(memberships_df)

        logger.info(f"Inserting {len(divisions_df)} divisions...")
        division_key_to_id = self.repository.insert_divisions(divisions_df)

        logger.info(f"Inserting {len(votes_df)} votes...")
        self.repository.insert_votes(votes_df, division_key_to_id)

        logger.info("Updating party_at_time in utterances...")
        self.repository.update_party_at_time()

        logger.info("Metadata pipeline completed successfully.")
