"""Unit and integration tests for the classify node."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from backend.src.chatbot.nodes.classify import (
    MAX_MESSAGES,
    Classification,
    classify_node,
)
from backend.src.chatbot.state import ActiveContext, ContextUpdate

MOCK_PARTIES = ["conservative", "labour", "liberal-democrat", "scottish-national-party"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm(monkeypatch):
    """Replace the llm in classify module with a mock.

    Returns the AsyncMock classifier whose .ainvoke() you can configure.
    """
    mock_classifier = AsyncMock()
    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_classifier
    monkeypatch.setattr("backend.src.chatbot.nodes.classify.llm", mock_llm_instance)
    return mock_classifier


@pytest.fixture
def mock_hansard_tool(monkeypatch):
    """Replace the hansard_tool in classify module with a mock."""
    mock_tool = MagicMock()
    mock_tool.list_parties.return_value = MOCK_PARTIES
    monkeypatch.setattr(
        "backend.src.chatbot.nodes.classify.hansard_tool", mock_tool
    )
    return mock_tool


def _make_state(message, **overrides):
    """Build a minimal AgentState dict for testing."""
    state = {
        "messages": [HumanMessage(content=message)],
        "active_context": ActiveContext(),
        "last_turn_was_ai_question": False,
    }
    state.update(overrides)
    return state


def _default_classification(**overrides):
    """Build a Classification with sensible defaults."""
    kwargs = {
        "user_intent": "new_query",
        "context_update": ContextUpdate(),
        "need_votes": False,
    }
    kwargs.update(overrides)
    return Classification(**kwargs)


# ---------------------------------------------------------------------------
# Unit tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestClassifyNode:

    @pytest.mark.asyncio
    async def test_returns_correct_shape(self, mock_llm, mock_hansard_tool):
        mock_llm.ainvoke.return_value = _default_classification(
            user_intent="new_query",
            context_update=ContextUpdate(
                person_names=["Keir Starmer"],
                parties=["labour"],
                date_text="last 6 months",
                search_query="housing",
            ),
            need_votes=False,
        )

        result = await classify_node(_make_state("What has Keir Starmer said about housing?"))

        assert result["user_intent"] == "new_query"
        assert result["need_votes"] is False
        assert result["context_update"]["search_query"] == "housing"
        assert result["context_update"]["parties"] == ["labour"]
        assert result["context_update"]["person_names"] == ["Keir Starmer"]
        assert result["context_update"]["date_text"] == "last 6 months"

    @pytest.mark.asyncio
    async def test_context_update_is_plain_dict(self, mock_llm, mock_hansard_tool):
        mock_llm.ainvoke.return_value = _default_classification()

        result = await classify_node(_make_state("test"))

        assert isinstance(result["context_update"], dict)

    @pytest.mark.asyncio
    async def test_message_truncation(self, mock_llm, mock_hansard_tool):
        mock_llm.ainvoke.return_value = _default_classification()
        messages = [HumanMessage(content=f"Message {i}") for i in range(12)]

        await classify_node(_make_state("ignored", messages=messages))

        call_args = mock_llm.ainvoke.call_args[0][0]
        # 1 system message + last MAX_MESSAGES user messages
        assert len(call_args) == 1 + MAX_MESSAGES
        assert call_args[1].content == "Message 4"
        assert call_args[-1].content == "Message 11"

    @pytest.mark.asyncio
    async def test_prompt_includes_parties(self, mock_llm, mock_hansard_tool):
        mock_llm.ainvoke.return_value = _default_classification()

        await classify_node(_make_state("test"))

        system_msg = mock_llm.ainvoke.call_args[0][0][0].content
        for party in MOCK_PARTIES:
            assert party in system_msg

    @pytest.mark.asyncio
    async def test_prompt_includes_active_context(self, mock_llm, mock_hansard_tool):
        mock_llm.ainvoke.return_value = _default_classification()
        active_context = ActiveContext(
            person_names=["Keir Starmer"],
            parties=["labour"],
            date_text="last month",
        )

        await classify_node(_make_state("test", active_context=active_context))

        system_msg = mock_llm.ainvoke.call_args[0][0][0].content
        assert "Keir Starmer" in system_msg
        assert "last month" in system_msg

    @pytest.mark.asyncio
    async def test_prompt_includes_last_turn_flag(self, mock_llm, mock_hansard_tool):
        mock_llm.ainvoke.return_value = _default_classification(
            user_intent="answer_to_question"
        )

        await classify_node(
            _make_state("Keir Starmer", last_turn_was_ai_question=True)
        )

        system_msg = mock_llm.ainvoke.call_args[0][0][0].content
        assert "Last turn was AI question: True" in system_msg


# ---------------------------------------------------------------------------
# Integration tests (real LLM — run with: pytest -m integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestClassifyNodeIntegration:

    @pytest.mark.asyncio
    async def test_classifies_new_query(self):
        result = await classify_node(
            _make_state("What has Labour said about housing in 2025?")
        )
        assert result["user_intent"] == "new_query"
        assert result["need_votes"] is False
        ctx = result["context_update"]
        assert any("labour" in p.lower() for p in ctx["parties"])
        assert "housing" in ctx["search_query"].lower()
        assert "2025" in ctx["date_text"]

    @pytest.mark.asyncio
    async def test_classifies_refine_query(self):
        active_context = ActiveContext(
            parties=["labour"],
            search_query="housing",
            date_text="in 2025",
        )
        result = await classify_node(
            _make_state(
                "What about in 2024 instead?",
                active_context=active_context,
            )
        )
        assert result["user_intent"] == "refine_query"
        assert "2024" in result["context_update"]["date_text"]

    @pytest.mark.asyncio
    async def test_classifies_answer_to_question(self):
        result = await classify_node(
            _make_state(
                "Keir Starmer",
                last_turn_was_ai_question=True,
                messages=[
                    AIMessage(content="Did you mean Keir Starmer or Ian Murray?"),
                    HumanMessage(content="Keir Starmer"),
                ],
            )
        )
        assert result["user_intent"] == "answer_to_question"
        assert "Keir Starmer" in result["context_update"]["person_names"]

    @pytest.mark.asyncio
    async def test_detects_voting_query(self):
        result = await classify_node(
            _make_state("How did Keir Starmer vote on the Rwanda bill?")
        )
        assert result["need_votes"] is True
        assert "Keir Starmer" in result["context_update"]["person_names"]
        assert "rwanda" in result["context_update"]["search_query"].lower()
