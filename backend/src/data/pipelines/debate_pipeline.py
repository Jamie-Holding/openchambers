"""Pipeline for processing debate XML files into the database."""

import json
import logging
from pathlib import Path
from typing import Any

from backend.src.data.loaders.theyworkforyou.debates import Debates

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = "backend/data/processed/.debate_pipeline_checkpoint.json"


class DebatePipeline:
    """Batch pipeline for processing debate XML files into the database.

    Applies transformers, generates embeddings, and supports checkpointing
    for resumable runs.
    """

    def __init__(
        self,
        data_dir: str,
        repository: Any,
        transformers: list[Any] | None,
        embedding_model: Any = None,
        batch_size: int = 50,
        max_batches: int | None = None,
        start_date: str | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        """Initialise the debate pipeline.

        Args:
            data_dir: Directory containing debate XML files.
            repository: Repository for database operations.
            transformers: List of transformers to apply to each batch.
            embedding_model: Model for generating embeddings.
            batch_size: Number of files per batch.
            max_batches: Maximum batches to process (None for unlimited).
            start_date: Only process files from this date (YYYY-MM-DD).
            checkpoint_path: Path for checkpoint file.
        """
        self.data_dir = data_dir
        self.loader = Debates(source_path=data_dir, start_date=start_date)
        self.repository = repository
        self.transformers = transformers or []
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.start_date = start_date
        self.max_batches = max_batches
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
        self._processed_files = self._load_checkpoint()

    def _load_checkpoint(self) -> set[str]:
        """Load the set of already-processed files from checkpoint.

        Returns:
            Set of file paths that have already been processed.
        """
        checkpoint = Path(self.checkpoint_path)
        if checkpoint.exists():
            try:
                data = json.loads(checkpoint.read_text())
                processed = set(data.get("processed_files", []))
                logger.info(f"Loaded checkpoint: {len(processed)} files already processed")
                return processed
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
        return set()

    def _save_checkpoint(self) -> None:
        """Save the set of processed files to checkpoint."""
        checkpoint = Path(self.checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({"processed_files": list(self._processed_files)}))

    def clear_checkpoint(self) -> None:
        """Clear the checkpoint file to start fresh."""
        checkpoint = Path(self.checkpoint_path)
        if checkpoint.exists():
            checkpoint.unlink()
            logger.info("Checkpoint cleared.")
        self._processed_files = set()

    def run(self) -> None:
        """Run the pipeline with error handling and checkpointing.

        Raises:
            Exception: Re-raises any exception from batch processing after
                saving checkpoint.
        """
        batches_processed = 0
        total_utterances = 0
        failed_batches = 0

        for batch_number, batch_df in enumerate(self.loader.iter_batches(self.batch_size)):
            if self.max_batches is not None and batches_processed >= self.max_batches:
                break

            # Get unique files in this batch to check against checkpoint.
            batch_files = set(batch_df["xml_path"].unique())

            # Skip if all files in batch already processed.
            if batch_files.issubset(self._processed_files):
                logger.debug(f"Skipping batch {batch_number} (already processed)")
                continue

            # Filter out already-processed files from this batch.
            new_files = batch_files - self._processed_files
            batch_df = batch_df[batch_df["xml_path"].isin(new_files)]

            if batch_df.empty:
                continue

            logger.info(f"Processing batch {batch_number} ({len(batch_df)} utterances from {len(new_files)} files)")

            try:
                # Apply transformers.
                logger.debug("Applying transformers...")
                for transformer in self.transformers:
                    batch_df = transformer.transform(batch_df)

                # Generate embeddings for chunks.
                logger.debug(f"Generating embeddings for {len(batch_df)} chunks...")
                texts = batch_df["chunk_embedding_text"].tolist()
                embeddings = self.embedding_model.encode(
                    texts, batch_size=8, convert_to_numpy=True
                )
                batch_df["embedding"] = embeddings.tolist()

                # Write to db.
                logger.debug("Writing batch to database...")
                self.repository.insert_batch_with_chunks(
                    batch_df, self.embedding_model.__class__.__name__
                )

                # Mark files as processed and save checkpoint.
                self._processed_files.update(new_files)
                self._save_checkpoint()

                batches_processed += 1
                total_utterances += len(batch_df)
                logger.info(f"Finished batch {batch_number}")

            except Exception as e:
                failed_batches += 1
                logger.error(f"Batch {batch_number} failed: {e}")
                logger.info("Checkpoint saved. Resume by re-running the pipeline.")
                raise

        logger.info(
            f"Pipeline complete: {batches_processed} batches, "
            f"{total_utterances} utterances, {failed_batches} failures"
        )
