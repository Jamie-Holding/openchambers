"""Synthesize a response from retrieval results."""

import json

from langchain_core.messages import AIMessage, SystemMessage

from src.chatbot.messages.generate import NO_RESULTS
from src.chatbot.prompts.generate import GENERATE_PROMPT
from src.chatbot.state import AgentState
from src.chatbot.utils import llm

MAX_MESSAGES = 8


def _truncate_ai_message(content: str) -> str:
    """Keep just the SUMMARY section from a previous AI response."""
    marker = "\nEVIDENCE:"
    idx = content.find(marker)
    if idx != -1:
        return content[:idx].strip()
    return content


def _prepare_messages(state: AgentState) -> list:
    """Build a recent message list with truncated AI messages."""
    messages = []
    for msg in state["messages"][-MAX_MESSAGES:]:
        if isinstance(msg, AIMessage) and msg.content:
            messages.append(AIMessage(content=_truncate_ai_message(msg.content)))
        else:
            messages.append(msg)
    return messages


async def generate_node(state: AgentState) -> dict:
    """Synthesize a response from retrieval results using the LLM."""
    retrieval_result = state.get("retrieval_result") or {}
    quotes = retrieval_result.get("quotes", [])
    votes = retrieval_result.get("votes", [])

    if not quotes and not votes:
        return {"messages": [AIMessage(content=NO_RESULTS)]}

    system = SystemMessage(
        content=GENERATE_PROMPT.format(
            quotes_json=json.dumps(quotes, indent=2) if quotes else "None",
            votes_json=json.dumps(votes, indent=2) if votes else "None",
        )
    )

    recent_messages = _prepare_messages(state)
    response = await llm.ainvoke([system, *recent_messages])

    return {"messages": [response]}
