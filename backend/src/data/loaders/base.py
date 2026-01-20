"""Base class for batch-based data loaders."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

import pandas as pd


class BaseLoader(ABC):
    """Abstract base class for all data loaders with batch-based loading."""

    def __init__(self, source_path: str) -> None:
        """Initialise the loader.

        Args:
            source_path: Path to the data source directory or file.
        """
        self.source_path = source_path

    @abstractmethod
    def load_batch(self, batch_number: int, batch_size: int) -> pd.DataFrame:
        """Load a single batch of data.

        Args:
            batch_number: Zero-indexed batch number to load.
            batch_size: Number of records per batch.

        Returns:
            DataFrame containing the batch data, or empty DataFrame if no
            more batches remain.
        """
        pass

    def iter_batches(self, batch_size: int) -> Iterator[pd.DataFrame]:
        """Yield batches one at a time.

        Args:
            batch_size: Number of records per batch.

        Yields:
            DataFrames containing batch data until exhausted.
        """
        batch_number = 0
        while True:
            df = self.load_batch(batch_number, batch_size)
            if df.empty:
                break
            yield df
            batch_number += 1
