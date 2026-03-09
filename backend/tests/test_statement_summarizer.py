"""Unit tests for StatementSummarizer."""

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.data.transformers.statement_summarizer import StatementSummarizer

SHORT_TEXT = "A short question about housing."
LONG_TEXT = "x" * 600  # Exceeds default 500-char threshold
MOCK_SUMMARY = "Summarized text about housing."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def summarizer(tmp_path):
    """Create a StatementSummarizer with mocked OpenAI client and temp cache."""
    cache_path = str(tmp_path / "test_cache.json")
    with patch.object(StatementSummarizer, "__init__", lambda self, **kw: None):
        s = StatementSummarizer()
    s.client = AsyncMock()
    s.model = "gpt-4o-mini"
    s.summarisation_threshold_chars = 500
    s.target_tokens = 50
    s.cache_path = cache_path
    s.include_statement = False
    s.include_main_question = True
    s.include_context_question = True
    s.max_concurrent = 10
    s._cache = {}

    # Mock OpenAI response structure
    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=MOCK_SUMMARY))]
    )
    s.client.chat.completions.create = AsyncMock(return_value=mock_response)
    return s


def _make_df(**kwargs):
    """Build a single-row DataFrame with statement/question fields."""
    defaults = {
        "statement_text": None,
        "question_text": None,
        "context_question_text": None,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# _hash_text
# ---------------------------------------------------------------------------


class TestHashText:
    def test_returns_sha256(self, summarizer):
        text = "test input"
        expected = hashlib.sha256(text.encode()).hexdigest()
        assert summarizer._hash_text(text) == expected

    def test_different_inputs_different_hashes(self, summarizer):
        assert summarizer._hash_text("a") != summarizer._hash_text("b")

    def test_same_input_same_hash(self, summarizer):
        assert summarizer._hash_text("abc") == summarizer._hash_text("abc")


# ---------------------------------------------------------------------------
# Cache persistence
# ---------------------------------------------------------------------------


class TestCache:
    def test_save_and_load(self, summarizer):
        summarizer._cache = {"hash1": "summary1", "hash2": "summary2"}
        summarizer._save_cache()

        loaded = summarizer._load_cache()
        assert loaded == {"hash1": "summary1", "hash2": "summary2"}

    def test_load_missing_file_returns_empty(self, summarizer, tmp_path):
        summarizer.cache_path = str(tmp_path / "nonexistent" / "cache.json")
        assert summarizer._load_cache() == {}

    def test_load_corrupt_json_returns_empty(self, summarizer, tmp_path):
        corrupt_file = tmp_path / "corrupt.json"
        corrupt_file.write_text("{invalid json")
        summarizer.cache_path = str(corrupt_file)
        assert summarizer._load_cache() == {}

    def test_clear_cache_removes_file_and_dict(self, summarizer):
        summarizer._cache = {"key": "val"}
        summarizer._save_cache()
        assert Path(summarizer.cache_path).exists()

        summarizer.clear_cache()
        assert summarizer._cache == {}
        assert not Path(summarizer.cache_path).exists()


# ---------------------------------------------------------------------------
# _process_text
# ---------------------------------------------------------------------------


class TestProcessText:
    @pytest.mark.asyncio
    async def test_short_text_skipped(self, summarizer):
        stats = {"cache_hits": 0, "api_calls": 0, "skipped": 0}
        semaphore = asyncio.Semaphore(10)

        result = await summarizer._process_text(
            SHORT_TEXT, "question", semaphore, stats
        )

        assert result == SHORT_TEXT
        assert stats["skipped"] == 1
        summarizer.client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_text_calls_api(self, summarizer):
        stats = {"cache_hits": 0, "api_calls": 0, "skipped": 0}
        semaphore = asyncio.Semaphore(10)

        result = await summarizer._process_text(LONG_TEXT, "question", semaphore, stats)

        assert result == MOCK_SUMMARY
        assert stats["api_calls"] == 1
        summarizer.client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_cached_text_skips_api(self, summarizer):
        text_hash = summarizer._hash_text(LONG_TEXT)
        summarizer._cache[text_hash] = "cached summary"
        stats = {"cache_hits": 0, "api_calls": 0, "skipped": 0}
        semaphore = asyncio.Semaphore(10)

        result = await summarizer._process_text(LONG_TEXT, "question", semaphore, stats)

        assert result == "cached summary"
        assert stats["cache_hits"] == 1
        summarizer.client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_input_returns_none(self, summarizer):
        stats = {"cache_hits": 0, "api_calls": 0, "skipped": 0}
        semaphore = asyncio.Semaphore(10)

        result = await summarizer._process_text(None, "question", semaphore, stats)

        assert result is None
        summarizer.client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_string_returns_empty(self, summarizer):
        stats = {"cache_hits": 0, "api_calls": 0, "skipped": 0}
        semaphore = asyncio.Semaphore(10)

        result = await summarizer._process_text("", "question", semaphore, stats)

        assert result == ""
        summarizer.client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_api_result_is_cached(self, summarizer):
        stats = {"cache_hits": 0, "api_calls": 0, "skipped": 0}
        semaphore = asyncio.Semaphore(10)

        await summarizer._process_text(LONG_TEXT, "question", semaphore, stats)

        text_hash = summarizer._hash_text(LONG_TEXT)
        assert text_hash in summarizer._cache
        assert summarizer._cache[text_hash] == MOCK_SUMMARY


# ---------------------------------------------------------------------------
# transform — full pipeline
# ---------------------------------------------------------------------------


class TestTransform:
    def test_summarizes_long_question_text(self, summarizer):
        df = _make_df(question_text=LONG_TEXT)
        result = summarizer.transform(df)
        assert result.iloc[0]["question_text"] == MOCK_SUMMARY

    def test_preserves_short_question_text(self, summarizer):
        df = _make_df(question_text=SHORT_TEXT)
        result = summarizer.transform(df)
        assert result.iloc[0]["question_text"] == SHORT_TEXT

    def test_summarizes_long_context_question_text(self, summarizer):
        df = _make_df(context_question_text=LONG_TEXT)
        result = summarizer.transform(df)
        assert result.iloc[0]["context_question_text"] == MOCK_SUMMARY

    def test_none_fields_unchanged(self, summarizer):
        df = _make_df(question_text=None, context_question_text=None)
        result = summarizer.transform(df)
        assert pd.isna(result.iloc[0]["question_text"])
        assert pd.isna(result.iloc[0]["context_question_text"])

    def test_cache_persisted_after_transform(self, summarizer):
        df = _make_df(question_text=LONG_TEXT)
        summarizer.transform(df)

        cache_data = json.loads(Path(summarizer.cache_path).read_text())
        text_hash = summarizer._hash_text(LONG_TEXT)
        assert cache_data[text_hash] == MOCK_SUMMARY


# ---------------------------------------------------------------------------
# Include flags
# ---------------------------------------------------------------------------


class TestIncludeFlags:
    def test_include_statement_false_skips_statement_text(self, summarizer):
        summarizer.include_statement = False
        df = _make_df(statement_text=LONG_TEXT)
        result = summarizer.transform(df)
        assert result.iloc[0]["statement_text"] == LONG_TEXT
        summarizer.client.chat.completions.create.assert_not_called()

    def test_include_statement_true_summarizes_statement_text(self, summarizer):
        summarizer.include_statement = True
        df = _make_df(statement_text=LONG_TEXT)
        result = summarizer.transform(df)
        assert result.iloc[0]["statement_text"] == MOCK_SUMMARY

    def test_include_main_question_false_skips_question_text(self, summarizer):
        summarizer.include_main_question = False
        df = _make_df(question_text=LONG_TEXT)
        result = summarizer.transform(df)
        assert result.iloc[0]["question_text"] == LONG_TEXT
        summarizer.client.chat.completions.create.assert_not_called()

    def test_include_main_question_true_summarizes_question_text(self, summarizer):
        summarizer.include_main_question = True
        df = _make_df(question_text=LONG_TEXT)
        result = summarizer.transform(df)
        assert result.iloc[0]["question_text"] == MOCK_SUMMARY

    def test_include_context_question_false_skips_context_question_text(
        self, summarizer
    ):
        summarizer.include_context_question = False
        df = _make_df(context_question_text=LONG_TEXT)
        result = summarizer.transform(df)
        assert result.iloc[0]["context_question_text"] == LONG_TEXT
        summarizer.client.chat.completions.create.assert_not_called()

    def test_include_context_question_true_summarizes_context_question_text(
        self, summarizer
    ):
        summarizer.include_context_question = True
        df = _make_df(context_question_text=LONG_TEXT)
        result = summarizer.transform(df)
        assert result.iloc[0]["context_question_text"] == MOCK_SUMMARY
