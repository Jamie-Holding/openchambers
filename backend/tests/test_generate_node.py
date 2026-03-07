"""Unit and integration tests for the generate node."""

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.chatbot.messages.generate import NO_RESULTS
from src.chatbot.nodes.generate import (
    _prepare_messages,
    _truncate_ai_message,
    generate_node,
)
from src.chatbot.schemas import ActiveContext

MOCK_QUOTE = {
    "date": "2025-03-01",
    "text": "We need more housing.",
    "speaker": {"name": "Chris Hinchliff", "office": None},
    "party": "labour",
    "speech_id": 1,
    "context": {
        "topic": "Housing",
        "department": None,
        "session": None,
        "main_question": None,
        "context_question": None,
    },
}
MOCK_VOTE = {
    "person_id": 101,
    "policy_name": "Health Funding",
    "policy_description": "Votes on health funding",
    "context_description": "health funding",
    "mp_stance_label": "consistently voted for",
    "mp_policy_alignment_score": 0.95,
    "num_votes_same": 10,
    "num_strong_votes_same": 5,
    "num_votes_different": 1,
    "num_strong_votes_different": 0,
    "num_votes_absent": 0,
    "num_strong_votes_absent": 0,
    "num_votes_abstain": 0,
    "num_strong_votes_abstain": 0,
    "total_votes": 11,
    "total_opportunities": 11,
    "percent_aligned": 90.9,
    "percent_opposed": 9.1,
    "percent_absent": 0.0,
    "percent_abstain": 0.0,
}

FULL_AI_RESPONSE = (
    "SUMMARY:\n"
    "- Housing is important.\n"
    "- Labour wants more homes.\n"
    "\nEVIDENCE:\n"
    "[QUOTE]\n"
    "Who/when: Chris Hinchliff — 2025-03-01 — labour\n"
    "Where: Housing\n"
    "Point: Housing is key.\n"
    'Quote: "We need more housing."'
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm(monkeypatch):
    """Replace the llm in generate module with a mock."""
    mock_llm_instance = AsyncMock()
    mock_llm_instance.ainvoke.return_value = AIMessage(
        content="SUMMARY:\n- Mocked response."
    )
    monkeypatch.setattr("src.chatbot.nodes.generate.llm", mock_llm_instance)
    return mock_llm_instance


def _make_state(message="What about housing?", **overrides):
    """Build a minimal AgentState dict for testing."""
    state = {
        "messages": [HumanMessage(content=message)],
        "active_context": ActiveContext(),
        "last_turn_was_ai_question": False,
        "retrieval_result": {"quotes": [], "votes": []},
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# _truncate_ai_message
# ---------------------------------------------------------------------------


class TestTruncateAiMessage:
    def test_strips_evidence_section(self):
        result = _truncate_ai_message(FULL_AI_RESPONSE)
        assert "SUMMARY:" in result
        assert "EVIDENCE:" not in result
        assert "[QUOTE]" not in result

    def test_preserves_summary_content(self):
        result = _truncate_ai_message(FULL_AI_RESPONSE)
        assert "Housing is important." in result
        assert "Labour wants more homes." in result

    def test_no_evidence_marker_returns_unchanged(self):
        short_msg = "Just a simple response."
        assert _truncate_ai_message(short_msg) == short_msg


# ---------------------------------------------------------------------------
# _prepare_messages
# ---------------------------------------------------------------------------


class TestPrepareMessages:
    def test_truncates_ai_messages(self):
        state = _make_state(
            messages=[
                HumanMessage(content="What about housing?"),
                AIMessage(content=FULL_AI_RESPONSE),
                HumanMessage(content="Tell me more"),
            ]
        )
        result = _prepare_messages(state)
        ai_msg = result[1]
        assert isinstance(ai_msg, AIMessage)
        assert "EVIDENCE:" not in ai_msg.content
        assert "SUMMARY:" in ai_msg.content

    def test_preserves_human_messages(self):
        state = _make_state(
            messages=[
                HumanMessage(content="First question"),
                HumanMessage(content="Second question"),
            ]
        )
        result = _prepare_messages(state)
        assert result[0].content == "First question"
        assert result[1].content == "Second question"


# ---------------------------------------------------------------------------
# generate_node
# ---------------------------------------------------------------------------


class TestGenerateNode:
    @pytest.mark.asyncio
    async def test_happy_path(self, mock_llm):
        result = await generate_node(
            _make_state(
                retrieval_result={"quotes": [MOCK_QUOTE], "votes": []},
            )
        )

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_quotes_in_prompt(self, mock_llm):
        await generate_node(
            _make_state(
                retrieval_result={"quotes": [MOCK_QUOTE], "votes": []},
            )
        )

        system_msg = mock_llm.ainvoke.call_args[0][0][0].content
        assert "Chris Hinchliff" in system_msg
        assert "We need more housing." in system_msg

    @pytest.mark.asyncio
    async def test_votes_in_prompt(self, mock_llm):
        await generate_node(
            _make_state(
                retrieval_result={"quotes": [], "votes": [MOCK_VOTE]},
            )
        )

        system_msg = mock_llm.ainvoke.call_args[0][0][0].content
        assert "Health Funding" in system_msg
        assert "consistently voted for" in system_msg

    @pytest.mark.asyncio
    async def test_no_results_skips_llm(self, mock_llm):
        result = await generate_node(
            _make_state(
                retrieval_result={"quotes": [], "votes": []},
            )
        )

        assert result["messages"][0].content == NO_RESULTS
        mock_llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_retrieval_result_skips_llm(self, mock_llm):
        result = await generate_node(_make_state(retrieval_result=None))

        assert result["messages"][0].content == NO_RESULTS
        mock_llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_conversation_history_included(self, mock_llm):
        await generate_node(
            _make_state(
                messages=[
                    HumanMessage(content="First question"),
                    AIMessage(content=FULL_AI_RESPONSE),
                    HumanMessage(content="Follow up"),
                ],
                retrieval_result={"quotes": [MOCK_QUOTE], "votes": []},
            )
        )

        call_messages = mock_llm.ainvoke.call_args[0][0]
        # system + 3 conversation messages
        assert len(call_messages) == 4
        assert call_messages[1].content == "First question"
        assert "EVIDENCE:" not in call_messages[2].content
        assert call_messages[3].content == "Follow up"


# ---------------------------------------------------------------------------
# Integration tests (real LLM - run with: pytest -m integration)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGenerateNodeIntegration:
    @pytest.mark.asyncio
    async def test_real_llm_produces_summary_and_evidence(self):
        result = await generate_node(
            _make_state(
                retrieval_result={"quotes": [MOCK_QUOTE], "votes": []},
            )
        )
        content = result["messages"][0].content
        assert "SUMMARY:" in content
        assert "EVIDENCE:" in content
