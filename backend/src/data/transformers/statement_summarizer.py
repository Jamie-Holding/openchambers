"""Transformer for summarizing long text fields using OpenAI."""

import asyncio
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer

from backend.src.data.transformers.base import BaseTransformer

logger = logging.getLogger(__name__)

# Silence verbose OpenAI/httpx logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class StatementSummarizer(BaseTransformer):
    """Summarizes long text fields using OpenAI to reduce token count.

    Uses a persistent JSON cache to avoid re-summarizing duplicate text.
    Cache is saved after each API call for crash safety.

    Attributes:
        client: OpenAI client for API calls.
        tokenizer: Tokenizer for counting tokens.
        model: OpenAI model name to use.
        token_threshold: Summarize text exceeding this token count.
        target_tokens: Target length for summaries.
        cache_path: Path to the persistent cache file.
    """

    DEFAULT_CACHE_PATH = "backend/data/processed/.statement_summaries_cache.json"
    FIELDS_TO_SUMMARIZE = {
        "statement_text": "statement",
        "question_text": "question",
        "context_question_text": "question",
    }

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        token_threshold: int = 100,
        target_tokens: int = 50,
        cache_path: str | None = None,
        include_statement: bool = False,
        include_main_question: bool = True,
        include_context_question: bool = True,
        max_concurrent: int = 10,
    ) -> None:
        """Initialise the statement summarizer.

        Args:
            model: OpenAI model name to use for summarization.
            token_threshold: Summarize text exceeding this token count.
            target_tokens: Target length for summaries.
            cache_path: Path to the persistent cache file.
            include_statement: Whether to summarize statement_text fields.
            include_main_question: Whether to summarize question_text fields.
            include_context_question: Whether to summarize context_question_text fields.
            max_concurrent: Maximum concurrent API requests (for rate limiting).
        """
        self.client = AsyncOpenAI()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = model
        self.token_threshold = token_threshold
        self.target_tokens = target_tokens
        self.cache_path = cache_path or self.DEFAULT_CACHE_PATH
        self.include_statement = include_statement
        self.include_main_question = include_main_question
        self.include_context_question = include_context_question
        self.max_concurrent = max_concurrent
        self._cache = self._load_cache()

    def _load_cache(self) -> dict[str, str]:
        """Load existing summaries from disk.

        Returns:
            Dictionary mapping text hashes to summaries.
        """
        path = Path(self.cache_path)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                logger.info(f"Loaded summary cache: {len(data)} entries")
                return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load summary cache: {e}. Starting fresh.")
        return {}

    def _save_cache(self) -> None:
        """Persist cache to disk."""
        path = Path(self.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._cache))

    def _hash_text(self, text: str) -> str:
        """Create a hash key for the text.

        Args:
            text: Text to hash.

        Returns:
            SHA256 hash of the text.
        """
        return hashlib.sha256(text.encode()).hexdigest()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize long text fields in the DataFrame.

        Args:
            df: DataFrame containing text fields.

        Returns:
            DataFrame with long text fields summarized.
        """
        return asyncio.run(self._transform_async(df))

    async def _transform_async(self, df: pd.DataFrame) -> pd.DataFrame:
        """Async implementation of transform."""
        stats = {"cache_hits": 0, "api_calls": 0, "skipped": 0}
        semaphore = asyncio.Semaphore(self.max_concurrent)

        for field, text_type in self.FIELDS_TO_SUMMARIZE.items():
            # Skip fields based on include flags
            if field == "statement_text" and not self.include_statement:
                continue
            if field == "question_text" and not self.include_main_question:
                continue
            if field == "context_question_text" and not self.include_context_question:
                continue

            if field in df.columns:
                texts = df[field].tolist()
                results = await self._process_batch(texts, text_type, semaphore, stats, field)
                df[field] = results

        logger.info(
            f"Text summarization: {stats['api_calls']} API calls, "
            f"{stats['cache_hits']} cache hits, {stats['skipped']} under threshold"
        )
        return df

    async def _process_batch(
        self,
        texts: list[str],
        text_type: str,
        semaphore: asyncio.Semaphore,
        stats: dict,
        field: str,
    ) -> list[str]:
        """Process a batch of texts concurrently."""
        tasks = [
            self._process_text(text, text_type, semaphore, stats)
            for text in texts
        ]
        results = await tqdm_asyncio.gather(*tasks, desc=f"Summarizing {field}")
        self._save_cache()
        return results

    async def _process_text(
        self,
        text: str,
        text_type: str,
        semaphore: asyncio.Semaphore,
        stats: dict,
    ) -> str:
        """Process a single text field, summarizing if needed."""
        if pd.isna(text) or not text:
            return text

        token_count = len(self.tokenizer.encode(text))
        if token_count <= self.token_threshold:
            stats["skipped"] += 1
            return text

        text_hash = self._hash_text(text)
        if text_hash in self._cache:
            stats["cache_hits"] += 1
            return self._cache[text_hash]

        # Call OpenAI with rate limiting
        async with semaphore:
            summary = await self._summarize(text, text_type)
        self._cache[text_hash] = summary
        stats["api_calls"] += 1
        return summary

    async def _summarize(self, text: str, text_type: str) -> str:
        """Call OpenAI to summarize the text."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"Summarize this parliamentary {text_type} in 1-2 sentences, "
                    "preserving key facts and positions.",
                },
                {"role": "user", "content": text},
            ],
            max_tokens=120,
        )
        output_text = response.choices[0].message.content
        return output_text

    def clear_cache(self) -> None:
        """Clear the cache file for fresh runs."""
        path = Path(self.cache_path)
        if path.exists():
            path.unlink()
            logger.info("Summary cache cleared.")
        self._cache = {}
