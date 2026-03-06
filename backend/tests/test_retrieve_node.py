"""Unit and integration tests for the retrieve node."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from src.chatbot.nodes.retrieve import _build_filter_combos, retrieve_node
from src.chatbot.schemas import ActiveContext

MOCK_QUOTE = {"speech_id": 1, "text": "We need more housing.", "date": "2025-03-01"}
MOCK_QUOTE_2 = {"speech_id": 2, "text": "Education is key.", "date": "2025-04-01"}
MOCK_VOTE = {"person_id": 101, "policy_name": "Health", "mp_stance_label": "voted for"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_hansard_tool(monkeypatch):
    """Replace hansard_tool in the retrieve module with a mock."""
    mock_tool = MagicMock()
    mock_tool.fetch.return_value = [MOCK_QUOTE]
    mock_tool.get_mp_voting_record.return_value = [MOCK_VOTE]
    monkeypatch.setattr("src.chatbot.nodes.retrieve.hansard_tool", mock_tool)
    return mock_tool


def _make_state(**overrides):
    """Build a minimal AgentState dict for testing."""
    state = {
        "messages": [HumanMessage(content="test")],
        "active_context": ActiveContext(),
        "last_turn_was_ai_question": False,
        "need_votes": False,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# _build_filter_combos
# ---------------------------------------------------------------------------

class TestBuildFilterCombos:

    def test_no_filters(self):
        ctx = ActiveContext()
        combos = _build_filter_combos(ctx)
        assert combos == [{"person_id": None, "party": None}]

    def test_single_person_id(self):
        ctx = ActiveContext(person_ids=[101])
        combos = _build_filter_combos(ctx)
        assert combos == [{"person_id": 101, "party": None}]

    def test_single_party(self):
        ctx = ActiveContext(parties=["labour"])
        combos = _build_filter_combos(ctx)
        assert combos == [{"person_id": None, "party": "labour"}]

    def test_cross_product(self):
        ctx = ActiveContext(person_ids=[101, 102], parties=["labour", "conservative"])
        combos = _build_filter_combos(ctx)
        assert len(combos) == 4
        assert {"person_id": 101, "party": "labour"} in combos
        assert {"person_id": 102, "party": "conservative"} in combos


# ---------------------------------------------------------------------------
# retrieve_node
# ---------------------------------------------------------------------------

class TestRetrieveNode:

    @pytest.mark.asyncio
    async def test_happy_path(self, mock_hansard_tool):
        result = await retrieve_node(_make_state(
            active_context=ActiveContext(
                person_ids=[101],
                search_query="housing",
                date_from="2025-01-01",
                date_to="2025-12-31",
            ),
        ))

        assert len(result["retrieval_result"]["quotes"]) == 1
        assert result["retrieval_result"]["quotes"][0] == MOCK_QUOTE
        assert result["retrieval_result"]["votes"] == []
        mock_hansard_tool.fetch.assert_called_once_with(
            query="housing",
            date_from="2025-01-01",
            date_to="2025-12-31",
            person_id=101,
            party=None,
        )

    @pytest.mark.asyncio
    async def test_with_votes(self, mock_hansard_tool):
        result = await retrieve_node(_make_state(
            active_context=ActiveContext(person_ids=[101], search_query="health"),
            need_votes=True,
        ))

        assert len(result["retrieval_result"]["quotes"]) == 1
        assert len(result["retrieval_result"]["votes"]) == 1
        assert result["retrieval_result"]["votes"][0] == MOCK_VOTE
        mock_hansard_tool.get_mp_voting_record.assert_called_once_with(
            person_id=101, search_term="health",
        )

    @pytest.mark.asyncio
    async def test_no_search_query_skips_quotes(self, mock_hansard_tool):
        result = await retrieve_node(_make_state(
            active_context=ActiveContext(person_ids=[101]),
        ))

        assert result["retrieval_result"]["quotes"] == []
        mock_hansard_tool.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_search_query_skips_votes(self, mock_hansard_tool):
        result = await retrieve_node(_make_state(
            active_context=ActiveContext(person_ids=[101]),
            need_votes=True,
        ))

        assert result["retrieval_result"]["votes"] == []
        mock_hansard_tool.get_mp_voting_record.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_person_ids_fetches_unfiltered(self, mock_hansard_tool):
        result = await retrieve_node(_make_state(
            active_context=ActiveContext(search_query="housing"),
        ))

        assert len(result["retrieval_result"]["quotes"]) == 1
        mock_hansard_tool.fetch.assert_called_once_with(
            query="housing",
            date_from=None,
            date_to=None,
            person_id=None,
            party=None,
        )

    @pytest.mark.asyncio
    async def test_multiple_person_ids_calls_fetch_per_id(self, mock_hansard_tool):
        mock_hansard_tool.fetch.side_effect = [[MOCK_QUOTE], [MOCK_QUOTE_2]]

        result = await retrieve_node(_make_state(
            active_context=ActiveContext(
                person_ids=[101, 102], search_query="housing",
            ),
        ))

        assert len(result["retrieval_result"]["quotes"]) == 2
        assert mock_hansard_tool.fetch.call_count == 2

    @pytest.mark.asyncio
    async def test_multiple_parties_calls_fetch_per_party(self, mock_hansard_tool):
        mock_hansard_tool.fetch.side_effect = [[MOCK_QUOTE], [MOCK_QUOTE_2]]

        result = await retrieve_node(_make_state(
            active_context=ActiveContext(
                parties=["labour", "conservative"], search_query="housing",
            ),
        ))

        assert len(result["retrieval_result"]["quotes"]) == 2
        assert mock_hansard_tool.fetch.call_count == 2

    @pytest.mark.asyncio
    async def test_votes_need_person_ids(self, mock_hansard_tool):
        """need_votes=True but no person_ids - votes should be empty."""
        result = await retrieve_node(_make_state(
            active_context=ActiveContext(search_query="health"),
            need_votes=True,
        ))

        assert result["retrieval_result"]["votes"] == []
        mock_hansard_tool.get_mp_voting_record.assert_not_called()


# ---------------------------------------------------------------------------
# Integration tests (real DB - run with: pytest -m integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRetrieveNodeIntegration:

    @pytest.mark.asyncio
    async def test_real_fetch(self):
        result = await retrieve_node(_make_state(
            active_context=ActiveContext(
                search_query="housing",
                parties=["labour"],
                date_from="2025-01-01",
                date_to="2025-12-31",
            ),
        ))
        quotes = result["retrieval_result"]["quotes"]
        assert len(quotes) > 0
        assert "text" in quotes[0]

    @pytest.mark.asyncio
    async def test_real_votes(self):
        result = await retrieve_node(_make_state(
            active_context=ActiveContext(
                person_ids=[25353],
                search_query="health",
            ),
            need_votes=True,
        ))
        votes = result["retrieval_result"]["votes"]
        assert len(votes) > 0
        assert "policy_name" in votes[0]
