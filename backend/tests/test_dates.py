"""Unit and integration tests for the dates module."""

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.src.chatbot.dates import parse_dates, parse_dates_regex
from backend.src.chatbot.messages.resolve import DATE_NOT_UNDERSTOOD
from backend.src.chatbot.schemas import DateRange

MOCK_TODAY = date(2025, 3, 15)
MOCK_TODAY_STR = "2025-03-15"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeDate(date):
    """date subclass with a controllable today()."""

    @classmethod
    def today(cls):
        return MOCK_TODAY


@pytest.fixture
def freeze_date(monkeypatch):
    """Freeze date.today() in the dates module to MOCK_TODAY."""
    monkeypatch.setattr("backend.src.chatbot.dates.date", FakeDate)


@pytest.fixture
def mock_llm(monkeypatch):
    """Replace the llm in dates module with a mock.

    Returns the AsyncMock parser whose .ainvoke() you can configure.
    """
    mock_parser = AsyncMock()
    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_parser
    monkeypatch.setattr("backend.src.chatbot.dates.fast_llm", mock_llm_instance)
    return mock_parser


# ---------------------------------------------------------------------------
# parse_dates_regex - pure functions, frozen date
# ---------------------------------------------------------------------------

class TestParseDatesRegex:

    def test_in_year(self, freeze_date):
        assert parse_dates_regex("in 2025") == ("2025-01-01", "2025-12-31")

    def test_bare_year(self, freeze_date):
        assert parse_dates_regex("2025") == ("2025-01-01", "2025-12-31")

    def test_last_6_months(self, freeze_date):
        # 2025-03-15 minus 180 days = 2024-09-16
        result = parse_dates_regex("last 6 months")
        assert result == ("2024-09-16", MOCK_TODAY_STR)

    def test_last_1_year(self, freeze_date):
        result = parse_dates_regex("last 1 year")
        assert result == ("2024-03-15", MOCK_TODAY_STR)

    def test_since_month_with_year(self, freeze_date):
        result = parse_dates_regex("since march 2024")
        assert result == ("2024-03-01", MOCK_TODAY_STR)

    def test_since_month_without_year(self, freeze_date):
        result = parse_dates_regex("since january")
        assert result == ("2025-01-01", MOCK_TODAY_STR)

    def test_in_month_year(self, freeze_date):
        assert parse_dates_regex("in march 2024") == ("2024-03-01", "2024-03-31")

    def test_in_february_leap_year(self, freeze_date):
        assert parse_dates_regex("in february 2024") == ("2024-02-01", "2024-02-29")

    def test_unrecognised_returns_none(self, freeze_date):
        assert parse_dates_regex("nonsense") is None

    def test_empty_string_returns_none(self, freeze_date):
        assert parse_dates_regex("") is None


# ---------------------------------------------------------------------------
# parse_dates - async orchestrator
# ---------------------------------------------------------------------------

class TestParseDates:

    @pytest.mark.asyncio
    async def test_none_input(self):
        result = await parse_dates(None)
        assert result == (None, None, None)

    @pytest.mark.asyncio
    async def test_regex_match_skips_llm(self, freeze_date, mock_llm):
        date_from, date_to, ask_msg = await parse_dates("in 2025")
        assert date_from == "2025-01-01"
        assert date_to == "2025-12-31"
        assert ask_msg is None
        mock_llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_regex_miss_llm_success(self, freeze_date, mock_llm):
        mock_llm.ainvoke.return_value = DateRange(
            date_from="2024-01-01", date_to="2024-03-31"
        )
        date_from, date_to, ask_msg = await parse_dates("between january and march 2024")
        assert date_from == "2024-01-01"
        assert date_to == "2024-03-31"
        assert ask_msg is None

    @pytest.mark.asyncio
    async def test_both_fail_returns_ask_message(self, freeze_date, mock_llm):
        mock_llm.ainvoke.side_effect = Exception("LLM failed")
        date_from, date_to, ask_msg = await parse_dates("gibberish")
        assert date_from is None
        assert date_to is None
        assert ask_msg == DATE_NOT_UNDERSTOOD.format(date_text="gibberish")


# ---------------------------------------------------------------------------
# Integration tests (real LLM - run with: pytest -m integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestParseDatesIntegration:

    @pytest.mark.asyncio
    async def test_llm_fallback_parses_complex_expression(self):
        date_from, date_to, ask_msg = await parse_dates(
            "between january and march 2024"
        )
        assert ask_msg is None
        assert date_from is not None
        assert date_to is not None
        assert date_from.startswith("2024-01")
        assert date_to.startswith("2024-03")
