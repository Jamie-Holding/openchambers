"""Unit and integration tests for the resolve node."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from backend.src.chatbot.messages.resolve import PERSON_DISAMBIGUATION, PERSON_NOT_FOUND
from backend.src.chatbot.nodes.resolve import (
    _merge_context,
    _resolve_people,
    _resolve_person_disambiguation,
    resolve_node,
)
from backend.src.chatbot.schemas import ActiveContext, ContextUpdate

SINGLE_MATCH = [{"person_id": 101, "display_name": "Keir Starmer", "current_party": "labour"}]
DIFFERENT_NAMES_MATCH = [
    {"person_id": 201, "display_name": "David Davis", "current_party": "conservative"},
    {"person_id": 202, "display_name": "David T.C. Davies", "current_party": "conservative"},
]
SAME_NAME_MATCH = [
    {"person_id": 301, "display_name": "John Smith", "current_party": "labour"},
    {"person_id": 302, "display_name": "John Smith", "current_party": "conservative"},
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_hansard_tool(monkeypatch):
    """Replace hansard_tool in the resolve module with a mock."""
    mock_tool = MagicMock()
    monkeypatch.setattr("backend.src.chatbot.nodes.resolve.hansard_tool", mock_tool)
    return mock_tool


@pytest.fixture
def mock_parse_dates(monkeypatch):
    """Replace parse_dates in the resolve module with a mock."""
    mock_fn = AsyncMock(return_value=(None, None, None))
    monkeypatch.setattr("backend.src.chatbot.nodes.resolve.parse_dates", mock_fn)
    return mock_fn


def _make_state(message="test", **overrides):
    """Build a minimal AgentState dict for testing."""
    state = {
        "messages": [HumanMessage(content=message)],
        "active_context": ActiveContext(),
        "last_turn_was_ai_question": False,
        "user_intent": "new_query",
        "context_update": {},
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# _merge_context
# ---------------------------------------------------------------------------

class TestMergeContext:

    def test_new_query_replaces_entirely(self):
        existing = ActiveContext(
            person_names=["Old Person"],
            parties=["labour"],
            search_query="old topic",
        )
        update = ContextUpdate(
            person_names=["New Person"],
            parties=["conservative"],
            search_query="new topic",
            date_text="in 2025",
        )
        result = _merge_context(existing, update, "new_query")
        assert result.person_names == ["New Person"]
        assert result.parties == ["conservative"]
        assert result.search_query == "new topic"
        assert result.date_text == "in 2025"
        assert result.person_ids == []

    def test_refine_query_merges_changed_fields(self):
        existing = ActiveContext(
            person_names=["Keir Starmer"],
            person_ids=[101],
            parties=["labour"],
            date_text="in 2025",
            date_from="2025-01-01",
            date_to="2025-12-31",
            search_query="housing",
        )
        update = ContextUpdate(date_text="in 2024")
        result = _merge_context(existing, update, "refine_query")
        assert result.person_names == ["Keir Starmer"]
        assert result.person_ids == [101]
        assert result.parties == ["labour"]
        assert result.search_query == "housing"
        assert result.date_text == "in 2024"
        assert result.date_from is None
        assert result.date_to is None

    def test_refine_query_resets_person_ids_when_names_change(self):
        existing = ActiveContext(person_names=["Old"], person_ids=[99])
        update = ContextUpdate(person_names=["New Person"])
        result = _merge_context(existing, update, "refine_query")
        assert result.person_names == ["New Person"]
        assert result.person_ids == []

    def test_refine_query_no_changes_keeps_existing(self):
        existing = ActiveContext(
            person_names=["Keir Starmer"],
            person_ids=[101],
            search_query="housing",
        )
        update = ContextUpdate()
        result = _merge_context(existing, update, "refine_query")
        assert result.person_names == ["Keir Starmer"]
        assert result.person_ids == [101]
        assert result.search_query == "housing"

    def test_answer_to_question_returns_unchanged(self):
        existing = ActiveContext(person_names=["Keir Starmer"], search_query="housing")
        update = ContextUpdate(person_names=["Different"])
        result = _merge_context(existing, update, "answer_to_question")
        assert result.person_names == ["Keir Starmer"]
        assert result.search_query == "housing"


# ---------------------------------------------------------------------------
# _resolve_people
# ---------------------------------------------------------------------------

class TestResolvePeople:

    def test_single_match_sets_person_id(self, mock_hansard_tool):
        mock_hansard_tool.list_people.return_value = SINGLE_MATCH
        ctx = ActiveContext(person_names=["Keir Starmer"])
        result, ask_msg = _resolve_people(ctx)
        assert result.person_ids == [101]
        assert ask_msg is None

    def test_no_matches_returns_ask_message(self, mock_hansard_tool):
        mock_hansard_tool.list_people.return_value = []
        ctx = ActiveContext(person_names=["Nonexistent MP"])
        result, ask_msg = _resolve_people(ctx)
        assert result.person_ids == []
        assert "Nonexistent MP" in ask_msg

    def test_multiple_matches_returns_disambiguation(self, mock_hansard_tool):
        mock_hansard_tool.list_people.return_value = DIFFERENT_NAMES_MATCH
        ctx = ActiveContext(person_names=["David Davis"])
        result, ask_msg = _resolve_people(ctx)
        assert result.person_ids == []
        assert "David Davis" in ask_msg
        assert "David T.C. Davies" in ask_msg

    def test_no_person_names_passes_through(self, mock_hansard_tool):
        ctx = ActiveContext()
        result, ask_msg = _resolve_people(ctx)
        assert ask_msg is None
        mock_hansard_tool.list_people.assert_not_called()

    def test_already_resolved_passes_through(self, mock_hansard_tool):
        ctx = ActiveContext(person_names=["Keir Starmer"], person_ids=[101])
        result, ask_msg = _resolve_people(ctx)
        assert ask_msg is None
        mock_hansard_tool.list_people.assert_not_called()


# ---------------------------------------------------------------------------
# _resolve_person_disambiguation
# ---------------------------------------------------------------------------

class TestResolvePersonDisambiguation:

    def test_selects_by_number(self, mock_hansard_tool):
        mock_hansard_tool.list_people.return_value = DIFFERENT_NAMES_MATCH
        ctx = ActiveContext(person_names=["David Davis"])
        result, ask_msg = _resolve_person_disambiguation(ctx, "1")
        assert result.person_ids == [201]
        assert result.person_names == ["David Davis"]
        assert ask_msg is None

    def test_selects_second_option(self, mock_hansard_tool):
        mock_hansard_tool.list_people.return_value = DIFFERENT_NAMES_MATCH
        ctx = ActiveContext(person_names=["David Davis"])
        result, ask_msg = _resolve_person_disambiguation(ctx, "2")
        assert result.person_ids == [202]
        assert result.person_names == ["David T.C. Davies"]
        assert ask_msg is None

    def test_selects_by_unique_name(self, mock_hansard_tool):
        mock_hansard_tool.list_people.return_value = DIFFERENT_NAMES_MATCH
        ctx = ActiveContext(person_names=["David Davis"])
        result, ask_msg = _resolve_person_disambiguation(ctx, "T.C.")
        assert result.person_ids == [202]
        assert ask_msg is None

    def test_ambiguous_text_reasks(self, mock_hansard_tool):
        mock_hansard_tool.list_people.return_value = DIFFERENT_NAMES_MATCH
        ctx = ActiveContext(person_names=["David Davis"])
        result, ask_msg = _resolve_person_disambiguation(ctx, "David")
        assert result.person_ids == []
        assert "David Davis" in ask_msg

    def test_same_name_text_match_reasks(self, mock_hansard_tool):
        """When all matches share the same name, text matching can't disambiguate."""
        mock_hansard_tool.list_people.return_value = SAME_NAME_MATCH
        ctx = ActiveContext(person_names=["John Smith"])
        result, ask_msg = _resolve_person_disambiguation(ctx, "John Smith")
        assert result.person_ids == []
        assert ask_msg is not None
        assert "John Smith" in ask_msg

    def test_same_name_number_still_works(self, mock_hansard_tool):
        """Even with identical names, number selection resolves correctly."""
        mock_hansard_tool.list_people.return_value = SAME_NAME_MATCH
        ctx = ActiveContext(person_names=["John Smith"])
        result, ask_msg = _resolve_person_disambiguation(ctx, "2")
        assert result.person_ids == [302]
        assert ask_msg is None

    def test_no_unresolved_names_passes_through(self, mock_hansard_tool):
        ctx = ActiveContext(person_names=["Keir Starmer"], person_ids=[101])
        result, ask_msg = _resolve_person_disambiguation(ctx, "1")
        assert ask_msg is None
        mock_hansard_tool.list_people.assert_not_called()


# ---------------------------------------------------------------------------
# resolve_node - full node
# ---------------------------------------------------------------------------

class TestResolveNode:

    @pytest.mark.asyncio
    async def test_happy_path(self, mock_hansard_tool, mock_parse_dates):
        mock_hansard_tool.list_people.return_value = SINGLE_MATCH
        mock_parse_dates.return_value = ("2025-01-01", "2025-12-31", None)

        result = await resolve_node(_make_state(
            context_update={"person_names": ["Keir Starmer"], "date_text": "in 2025"},
        ))

        assert result["last_turn_was_ai_question"] is False
        ctx = result["active_context"]
        assert ctx.person_ids == [101]
        assert ctx.date_from == "2025-01-01"
        assert ctx.date_to == "2025-12-31"

    @pytest.mark.asyncio
    async def test_person_not_found_asks(self, mock_hansard_tool, mock_parse_dates):
        mock_hansard_tool.list_people.return_value = []

        result = await resolve_node(_make_state(
            context_update={"person_names": ["Nobody"]},
        ))

        assert result["last_turn_was_ai_question"] is True
        assert isinstance(result["messages"][0], AIMessage)
        assert "Nobody" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_disambiguation_asks(self, mock_hansard_tool, mock_parse_dates):
        mock_hansard_tool.list_people.return_value = DIFFERENT_NAMES_MATCH

        result = await resolve_node(_make_state(
            context_update={"person_names": ["David Davis"]},
        ))

        assert result["last_turn_was_ai_question"] is True
        assert "David T.C. Davies" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_answer_to_question_resolves(self, mock_hansard_tool, mock_parse_dates):
        mock_hansard_tool.list_people.return_value = DIFFERENT_NAMES_MATCH
        mock_parse_dates.return_value = (None, None, None)

        result = await resolve_node(_make_state(
            message="1",
            user_intent="answer_to_question",
            active_context=ActiveContext(person_names=["David Davis"]),
        ))

        assert result["last_turn_was_ai_question"] is False
        assert result["active_context"].person_ids == [201]


# ---------------------------------------------------------------------------
# Integration tests (real DB - run with: pytest -m integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestResolveNodeIntegration:

    @pytest.mark.asyncio
    async def test_resolves_real_mp(self):
        result = await resolve_node(_make_state(
            context_update={"person_names": ["Keir Starmer"], "search_query": "housing"},
        ))
        assert result["last_turn_was_ai_question"] is False
        assert len(result["active_context"].person_ids) == 1
        assert result["active_context"].person_ids[0] > 0
