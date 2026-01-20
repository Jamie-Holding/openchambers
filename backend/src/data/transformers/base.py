"""Base class for DataFrame transformers."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseTransformer(ABC):
    """Abstract base class for DataFrame transformers."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation to the input DataFrame.

        Args:
            df: Input DataFrame to transform.

        Returns:
            Transformed DataFrame.
        """
        pass
